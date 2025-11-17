import asyncio
from typing import Annotated, Sequence, TypedDict
from typing import Any, Dict, Callable, Optional, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from faker import Faker
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import create_agent
from typing import List, Dict, Any


load_dotenv()

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


_LIVE_EMITTER: Optional[Callable[[str], None]] = None


def set_live_emitter(emit: Callable[[str], None]) -> None:
    """
    Register a synchronous callback that will receive short log strings.
    main.py injects this to update the UI in near real-time.
    """
    global _LIVE_EMITTER
    _LIVE_EMITTER = emit


def _live_log(msg: str) -> None:
    # Keep original behavior (stdout) AND forward to UI if hooked.
    print(msg, flush=True)
    if _LIVE_EMITTER:
        try:
            _LIVE_EMITTER(msg)
        except Exception:
            # Don't let UI issues break the agent flow.
            pass


class WeatherInfo(BaseModel):
    """Contact information for a person."""

    weather_description: str = Field(description="The description of weather")
    weather_urls: List[str] = Field(
        description="List of urls for weather. May be empty"
    )
    weather_pdf: str = Field(description="Path to pdf")


class AgentState(TypedDict):
    """Состояние агента, содержащее последовательность сообщений."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


async def get_all_tools():
    """Получение всех инструментов: ваших + MCP"""
    # Настройка MCP клиента

    mcp_client = MultiServerMCPClient(
        {
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
                "command": "node",
                "args": ["/Users/user/MEProjects/markdown2pdf-mcp/build/index.js"],
                "env": {
                    "M2P_OUTPUT_DIR": os.environ.get("PATH_TO_PDF"),
                    "M2P_VERBOSE": "true",
                },
            },
        }
    )

    # Получаем MCP инструменты
    mcp_tools = await mcp_client.get_tools()

    # Объединяем ваши инструменты с MCP инструментами
    return mcp_tools


llm = ChatOpenAI(model="gpt-4.1-mini")


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    print("Надо продолжать")
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


async def run_with_message(message: str) -> Dict[str, Any]:

    # Получаем все инструменты и MCP клиент
    all_tools = await get_all_tools()

    # Привязываем инструменты к LLM
    llm_with_tools = create_agent(
        model="openai:gpt-4.1-mini",
        tools=all_tools,
    )

    async def call_weather(state: AgentState) -> AgentState:
        _live_log("Генерация погоды")

        system_prompt = SystemMessage(
            content=(
                "Ты моя система. У тебя есть MCP-погода"
                "Алгоритм: (1) при необходимости вызови погоду (2) Выведи информацию о погоде в читаемом виде"
            )
        )
        messages = [system_prompt] + list(state["messages"])
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        return agent_output  # already {"messages":[...]}

    async def call_image_generation(state: AgentState) -> AgentState:
        _live_log("Генерация изображений")

        system_prompt = SystemMessage(
            content=(
                "Ты моя система. У тебя есть MCP-генерация сообщений"
                "На вход получена информация о названии города и погоде в нем"
                "Алгоритм: (1) Сгенерируй изображения о погоде. Промт для генерации сделай на английском языке. (2) Выведи информацию о погоде и полученных ссылках на изображения о погоде"
            )
        )

        messages = [system_prompt] + list(state["messages"])
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        return agent_output

    async def call_pdf_to_markdown(state: AgentState) -> AgentState:
        _live_log("Генерация pdf файлов")

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
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        return agent_output

    async def call_final_answer(state: AgentState) -> AgentState:
        _live_log("Генерация финального ответа")

        system_prompt = SystemMessage(
            content=(
                "Ты моя система. У тебя есть информация о погоде, сгенерированные изображения и путь к pdf файлу"
                "На вход получена информация о названии города и погоде в нем. Также получен список изображений и путь к pdf файлу"
                "(1)Верни финальный объект WeatherInfo И БОЛЬШЕ НИЧЕГО И БОЛЬШЕ НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ."
                "В WeatherInfo(json) помести краткое описание погоды , список URL изображений, путь к PDF файлу (description, images, path). Финальный description должен быть на языке введенного сообщения"
            )
        )

        messages = [system_prompt] + list(state["messages"])
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        return agent_output

    async def model_call_with_tools(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content=(
                "Ты моя система. У тебя есть MCP-погода и генерация изображений."
                "Алгоритм: (1) при необходимости вызови погоду, (2) при необходимости сгенерируй изображение, промт для изображения на английском языке"
                "(3)Далее сформируй подробную информацию о городе или стране с достопримечательностями в формате Markdown (4)из markdown сгенерируй pdf и сохрани в файл, если есть такой инструмент"
                "(5)затем обязательно верни финальный объект WeatherInfo И БОЛЬШЕ НИЧЕГО И БОЛЬШЕ НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ."
                "В WeatherInfo(json) помести краткое описание погоды , список URL изображений, путь к PDF файлу (description, images, path). Финальный description должен быть на языке введенного сообщения"
            )
        )
        messages = [system_prompt] + list(state["messages"])
        # create_agent ожидает {"messages": [...]}
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        # agent_output уже dict {"messages":[...]} — вернём его напрямую
        print("Вызов model_call_with_tools")
        return agent_output

    # Создание графа
    graph = StateGraph(AgentState)

    graph.add_node("call_weather", call_weather)
    graph.add_node("call_image_generation", call_image_generation)
    graph.add_node("call_pdf_to_markdown", call_pdf_to_markdown)
    graph.add_node("call_final_answer", call_final_answer)

    # Настройка потока
    graph.add_edge(START, "call_weather")
    graph.add_edge("call_weather", "call_image_generation")
    graph.add_edge("call_image_generation", "call_pdf_to_markdown")
    graph.add_edge("call_pdf_to_markdown", "call_final_answer")
    graph.add_edge("call_final_answer", END)

    # Компиляция и запуск
    app = graph.compile()

    result = await app.ainvoke({"messages": [HumanMessage(content=message)]})

    # Показываем результат
    print("=== Полная история сообщений ===")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}. {type(msg).__name__}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   Tool calls: {msg.tool_calls}")

    # Финальный ответ
    final_message = result["messages"][-1]
    print(f"\n=== Финальный ответ ===")
    print(final_message.content)

    return result
