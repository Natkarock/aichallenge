import asyncio
from typing import Annotated, Sequence, TypedDict
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


class WeatherInfo(BaseModel):
    """Contact information for a person."""
    weather_description: str = Field(description="The description of weather")
    weather_urls: List[str] = Field(description="List of urls for weather. May be empty")


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
                    "MODEL": os.environ.get("REPLICATE_MODEL")
                },
            },
            "markdown2pdf": {
                "transport": "stdio",
                "command": "node",
                "args": ["/Users/user/MEProjects/markdown2pdf-mcp/build/index.js"],
                "env": {
                    "M2P_OUTPUT_DIR": os.environ.get("PATH_TO_PDF"),
                    "M2P_VERBOSE": "true",
                }
          }
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
    
    async def model_call_with_tools(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content=(
                "Ты моя система. У тебя есть MCP-погода и генерация изображений."
                "Алгоритм: (1) при необходимости вызови погоду, (2) при необходимости сгенерируй изображение, промт для изображения на английском языке"
                "(3)Далее сформируй подробную информацию о городе или стране с достопримечательностями в формате Markdown (4)из markdown сгенерируй pdf и сохрани в файл, если есть такой инструмент"
                "(5)затем обязательно верни финальный объект WeatherInfo И БОЛЬШЕ НИЧЕГО И БОЛЬШЕ НЕ ВЫЗЫВАЙ ИНСТРУМЕНТЫ."
                "В WeatherInfo(json) помести краткое описание погоды и список URL изображений (description, images). Финальный description должен быть на языке введенного сообщения"
           )
       )
        messages = [system_prompt] + list(state["messages"])
        # create_agent ожидает {"messages": [...]}
        agent_input = {"messages": messages}
        agent_output = await llm_with_tools.ainvoke(agent_input)
        # agent_output уже dict {"messages":[...]} — вернём его напрямую
        return agent_output

    # Создание графа
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call_with_tools)

    # ToolNode с всеми инструментами (ваши + MCP)
    graph.add_node("tools", ToolNode(tools=all_tools))

    # Настройка потока
    graph.add_edge(START, "our_agent")
    graph.add_conditional_edges(
        "our_agent",  # От какого узла
        should_continue,  # Функция-решатель
        {  # Карта решений
            "continue": "tools",  # Если "continue" → идем в "tools"
            "end": END,  # Если "end" → завершаем
        },
    )
    graph.add_edge("tools", "our_agent")

    # Компиляция и запуск
    app = graph.compile()

    result = await app.ainvoke(
        {"messages": [HumanMessage(content=message)]}
    )
    
        # Показываем результат
    print("=== Полная история сообщений ===")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}. {type(msg).__name__}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   Tool calls: {msg.tool_calls}")
            
    return result
  

