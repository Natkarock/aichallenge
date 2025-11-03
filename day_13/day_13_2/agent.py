# agent.py — live logs (Android LiveData–style) for mid-function updates
import asyncio
import contextvars
import queue as threading_queue
import threading
from typing import Annotated, Sequence, TypedDict, Dict, Any, List
import operator
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

load_dotenv()

# ---------------- Live logger (context-local), usable from ANY node mid-execution ----------------
_event_queue_ctx: contextvars.ContextVar["asyncio.Queue[dict] | None"] = (
    contextvars.ContextVar("event_q", default=None)
)


async def alog(text: str):
    """Async-safe log: await inside your async node functions to push a line immediately to UI."""
    q = _event_queue_ctx.get()
    if q is not None:
        await q.put({"type": "log", "text": text})


def log(text: str):
    """Sync-safe log: can be used even in sync parts; schedules an async put into the current loop."""
    q = _event_queue_ctx.get()
    if q is not None:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(
                asyncio.create_task, q.put({"type": "log", "text": text})
            )


# ---------------- Agent state with logs accumulator (for end-of-node summaries) ------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    logs: Annotated[List[str], operator.add]


def emit(text: str) -> Dict[str, Any]:
    """Adds a line to the state's log list (captured on node end)."""
    return {"logs": [text]}


# ---------------- MCP tools loading (robust one-by-one) ----------------
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


llm_with_tools = ChatOpenAI(
        model="openai:gpt-4.1-mini",
    )


async def get_all_tools():
    servers = {
        "weather_mcp": {
            "transport": "streamable_http",
            "url": "https://weathermcp-natkarock.amvera.io/mcp",
        },
        "image-gen": {
            "transport": "stdio",
            "command": "npx",
            "args": ["@gongrzhe/image-gen-server"],
            "env": {
                "REPLICATE_API_TOKEN": REPLICATE_API_TOKEN,
                "MODEL": os.environ.get("REPLICATE_MODEL"),
            },
        },
        "markdown2pdf": {
            "transport": "stdio",
            # Используй один из вариантов запуска — например node build/index.js:
            "command": "node",
            "args": ["/Users/user/MEProjects/markdown2pdf-mcp/build/index.js"],
            "env": {
                "M2P_OUTPUT_DIR": "/Users/user/MEProjects/aichallenge/agent/day_12/weather_image_mcp_agent_v3-2",
                "M2P_VERBOSE": "true",
                # При необходимости:
                # "M2P_CHROME_PATH": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                # "M2P_LOAD_TIMEOUT_MS": "120000",
                # "M2P_RENDER_TIMEOUT_MS": "20000",
            },
        },
    }
    client = MultiServerMCPClient(servers)
    tools: List = []
    for name in servers.keys():
        try:
            async with client.session(name) as session:
                t = await load_mcp_tools(session)
                tools.extend(t)
        except Exception as e:
            print(f"[MCP] Failed to load tools from '{name}': {e}")
            # не роняем загрузку остальных
            continue
    return tools


# ---------------- Wrapper that adds a log line without touching node logic ----------------
def with_log(fn, message: str):
    async def _wrapped(state: AgentState) -> AgentState:
        # allow user code to stream mid-function logs:
        # they can call: await alog("...progress...") or log("...progress...")
        out = await fn(state)
        if not isinstance(out, dict):
            out = {}
        # also add a summarized line at node end
        out.update(emit(message))
        return out

    return _wrapped


async def call_weather(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "Ты моя система. У тебя есть MCP-погода"
            "Алгоритм: (1) при необходимости вызови погоду (2) Выведи информацию о погоде в читаемом виде"
        )
    )
    messages = [system_prompt] + list(state["messages"])
    # create_agent ожидает {"messages": [...]}
    agent_input = {"messages": messages}
    print("Генерация погоды")
    agent_output = await llm_with_tools.ainvoke(agent_input)
    # agent_output уже dict {"messages":[...]} — вернём его напрямую
    return agent_output


async def call_image_generation(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "Ты моя система. У тебя есть MCP-генерация сообщений"
            "На вход получена информация о названии города и погоде в нем"
            "Алгоритм: (1) Сгенерируй изображения о погоде. Промт для генерации сделай на английском языке. (2) Выведи информацию о погоде и полученных ссылках на изображения о погоде"
        )
    )
    messages = [system_prompt] + list(state["messages"])
    # create_agent ожидает {"messages": [...]}
    agent_input = {"messages": messages}
    print("Генерация изображений")
    agent_output = await llm_with_tools.ainvoke(agent_input)
    # agent_output уже dict {"messages":[...]} — вернём его напрямую
    return agent_output


async def call_pdf_to_markdown(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "Ты моя система. У тебя есть MCP-генерация Pdf из markdown"
            "На вход получена информация о названии города и погоде в нем. Также получен список изображений"
            "Алгоритм: (1) Сформируй подробную информацию о городе или стране с достопримечательностями в формате Markdown"
            "(2)из markdown сгенерируй pdf и сохрани в файл через mcp"
            "(5)затем обязательно верни финальный объект WeatherInfo И БОЛЬШЕ НИЧЕГО И БОЛЬШЕ НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ."
            "В WeatherInfo(json) помести краткое описание погоды , список URL изображений, путь к PDF файлу (description, images, path). Финальный description должен быть на языке введенного сообщения"
        )
    )
    messages = [system_prompt] + list(state["messages"])
    # create_agent ожидает {"messages": [...]}
    agent_input = {"messages": messages}
    print("Генерация pdf файлов")
    agent_output = await llm_with_tools.ainvoke(agent_input)
    # agent_output уже dict {"messages":[...]} — вернём его напрямую
    return agent_output


async def call_final_answer(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "Ты моя система. У тебя есть информация о погоде, сгенерированные изображения и путь к pdf файлу"
            "На вход получена информация о названии города и погоде в нем. Также получен список изображений и путь к pdf файлу"
            "(1)Верни финальный объект WeatherInfo И БОЛЬШЕ НИЧЕГО И БОЛЬШЕ НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ."
            "В WeatherInfo(json) помести краткое описание погоды , список URL изображений, путь к PDF файлу (description, images, path). Финальный description должен быть на языке введенного сообщения"
        )
    )
    messages = [system_prompt] + list(state["messages"])
    # create_agent ожидает {"messages": [...]}
    agent_input = {"messages": messages}
    print("Генерация финального ответа")
    agent_output = await llm_with_tools.ainvoke(agent_input)
    # agent_output уже dict {"messages":[...]} — вернём его напрямую
    return agent_output


# ---------------- Build graph using your existing node functions ----------------
def _build_graph(all_tools):
    # IMPORTANT: we do NOT change your business logic; we assume these functions exist in this file:
    # - call_weather(state) -> AgentState
    # - call_image_generation(state) -> AgentState
    # - call_pdf_to_markdown(state) -> AgentState
    # - call_final_answer(state) -> AgentState
    # If they are in another module, import them above and keep the same names.

    # Create ToolNode (don't fail graph on tool errors)
    tool_node = ToolNode(tools=all_tools, handle_tool_errors=True)

    graph = StateGraph(AgentState)

    # Wrap your nodes with with_log(...) so we append a line when a node ends,
    # and you can stream mid-function via alog()/log().
    graph.add_node("call_weather", with_log(call_weather, "🌤️ Шаг 1/4: погода получена"))
    graph.add_node(
        "call_image_generation",
        with_log(call_image_generation, "🖼️ Шаг 2/4: изображение готово"),
    )
    graph.add_node(
        "call_pdf_to_markdown",
        with_log(call_pdf_to_markdown, "📄 Шаг 3/4: PDF сформирован"),
    )
    graph.add_node(
        "call_final_answer",
        with_log(call_final_answer, "✅ Шаг 4/4: финальный ответ сформирован"),
    )

    graph.add_node("tools", tool_node)

    # Keep your edges/order:
    graph.add_edge(START, "call_weather")
    graph.add_edge("call_weather", "tools")
    graph.add_edge("tools", "call_image_generation")
    graph.add_edge("call_image_generation", "tools")
    graph.add_edge("tools", "call_pdf_to_markdown")
    graph.add_edge("call_pdf_to_markdown", "tools")
    graph.add_edge("tools", "call_final_answer")
    graph.add_edge("call_final_answer", END)

    return graph.compile()


# ---------------- Legacy API: return final result (no streaming) ----------------
async def run_with_message(message: str) -> Dict[str, Any]:
    all_tools = await get_all_tools()
    app = _build_graph(all_tools)
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=message)], "logs": []}
    )
    return result


# ---------------- Live streaming API: yield events during execution ----------------
def run_with_message_events(message: str):
    """Sync generator for Streamlit: emits {'type':'log','text':...} and then {'type':'final','result':...}.

    Use alog()/log() in your nodes to stream mid-function progress to UI immediately.
    """
    thread_q: "threading_queue.Queue[dict | None]" = threading_queue.Queue(maxsize=500)

    async def producer():
        # internal async queue visible to nodes via context var
        async_q: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=1000)
        token = _event_queue_ctx.set(async_q)
        try:
            all_tools = await get_all_tools()
            app = _build_graph(all_tools)

            # run the graph in background
            task = asyncio.create_task(
                app.ainvoke({"messages": [HumanMessage(content=message)], "logs": []})
            )

            # forward any logs coming from nodes immediately
            async def forwarder():
                while True:
                    item = await async_q.get()
                    thread_q.put(item)  # send to Streamlit thread
                    async_q.task_done()

            fwd = asyncio.create_task(forwarder())

            # also emit node start/end via astream_events for nice UX
            async for ev in app.astream_events(
                {"messages": [HumanMessage(content=message)], "logs": []}, version="v1"
            ):
                if ev.get("event") == "on_node_start":
                    node = ev.get("name", "")
                    thread_q.put({"type": "log", "text": f"▶️ {node}…"})
                if ev.get("event") == "on_node_end" and ev.get("name") == "graph":
                    # graph finished; result is in ev['data']['output'] too, but we also wait for task
                    pass

            result = await task
            fwd.cancel()
            thread_q.put({"type": "final", "result": result})
        finally:
            _event_queue_ctx.reset(token)
            thread_q.put(None)

    def runner():
        asyncio.run(producer())

    threading.Thread(target=runner, daemon=True).start()

    while True:
        item = thread_q.get()
        if item is None:
            break
        yield item


# --------------- NOTE for your node functions ---------------
# Inside your existing nodes you can now do:
#
#   from agent import alog, log
#   await alog("Запрашиваю геокодинг…")       # mid-function, async
#   log("Получены координаты: 45.0, 38.9")    # mid-function, sync-safe
#
# These lines will appear in Streamlit immediately, before the node finishes.
