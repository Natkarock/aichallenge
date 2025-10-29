import asyncio
from fastmcp.client import Client

async def main():
    url = "http://0.0.0.0:3333/mcp"

    # Универсальное подключение
    async with Client(url) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        
        print(f"📋 Доступно инструментов: {len(tools)}")
        for tool in tools:
            print(f"  • {tool.name}")
      
        res = await client.call_tool("weather_forecast", {"input": {"city": "Махачкала", "days": 5}})
        data = res.data  # WeatherOutput
        print(data.city, data.country_code, data.timezone)
        print(data.current.temperature_c, data.current.weather_text)
        for day in data.daily:
            print(day.date, day.t_min_c, day.t_max_c, day.weather_text)

if __name__ == "__main__":
    asyncio.run(main())

