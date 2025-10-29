from fastmcp import FastMCP
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, AnyUrl, Field
from jinja2 import Environment, FileSystemLoader, select_autoescape
import httpx
from slugify import slugify
from pathlib import Path
import json
from typing import Optional

mcp = FastMCP("spec-assembler")

TEMPLATES_DIR = Path(__file__).parent / "routes" / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(enabled_extensions=("j2",)))


# ---------- MODELS ----------
class MakeSpecInput(BaseModel):
    template_id: Literal["mobile_app", "web_service", "library", "data_pipeline"]
    inputs: Dict[str, Any] = {}


class MakeSpecOutput(BaseModel):
    filename: str
    markdown: str


class ChecklistInput(BaseModel):
    purpose: str


class CriteriaInput(BaseModel):
    style: Literal["gherkin", "bulleted"] = "gherkin"
    items: List[str]


class OpenAPISummaryInput(BaseModel):
    url: AnyUrl


# --- WEATHER TOOL (Open-Meteo, без ключей) -----------------------------------

class WeatherInput(BaseModel):
    city: str = Field(..., description="Название города на любом языке")
    days: int = Field(3, ge=1, le=16, description="Сколько дней прогноза (1..16)")
    timezone: Optional[str] = Field(None, description="IANA таймзона (например, 'Europe/Amsterdam'). Если не указана — auto")

class WeatherCurrent(BaseModel):
    temperature_c: float
    wind_speed_ms: Optional[float] = None
    weather_code: Optional[int] = None
    weather_text: Optional[str] = None

class WeatherDaily(BaseModel):
    date: str
    t_min_c: Optional[float] = None
    t_max_c: Optional[float] = None
    precip_prob_max_pct: Optional[int] = None
    weather_code: Optional[int] = None
    weather_text: Optional[str] = None

class WeatherOutput(BaseModel):
    query: str
    city: str
    country_code: Optional[str] = None
    latitude: float
    longitude: float
    timezone: str
    current: Optional[WeatherCurrent] = None
    daily: list[WeatherDaily] = []

_WMO_TEXT = {
    0: "Ясно",
    1: "Преимущественно ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Изморозь",
    51: "Морось слабая", 53: "Морось", 55: "Морось сильная",
    56: "Ледяная морось слабая", 57: "Ледяная морось сильная",
    61: "Дождь слабый", 63: "Дождь", 65: "Дождь сильный",
    66: "Ледяной дождь слабый", 67: "Ледяной дождь сильный",
    71: "Снег слабый", 73: "Снег", 75: "Снег сильный",
    77: "Снежные зерна",
    80: "Ливни слабые", 81: "Ливни", 82: "Ливни сильные",
    85: "Снегопад ливневый слабый", 86: "Снегопад ливневый сильный",
    95: "Гроза", 96: "Гроза с градом", 99: "Гроза с сильным градом",
}

def _wmo_text(code: Optional[int]) -> Optional[str]:
    return _WMO_TEXT.get(code) if code is not None else None

@mcp.tool()
def weather_forecast(input: WeatherInput) -> WeatherOutput:
    """
    Прогноз погоды по названию города (без ключей API).
    1) Геокодирование через Open-Meteo Geocoding
    2) Прогноз current + daily через Open-Meteo Forecast
    """
    city_q = input.city.strip()
    if not city_q:
        raise ValueError("city is empty")

    # 1) Геокодирование
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_q, "count": 1, "language": "ru", "format": "json"}
    with httpx.Client(timeout=20) as cx:
        gr = cx.get(geocode_url, params=params)
        gr.raise_for_status()
        gj = gr.json()

    results = (gj or {}).get("results") or []
    if not results:
        raise ValueError(f"Город не найден: {city_q}")

    best = results[0]
    lat = float(best["latitude"])
    lon = float(best["longitude"])
    disp_name = best.get("name") or city_q
    country_code = best.get("country_code")

    # 2) Прогноз
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    tz = input.timezone or "auto"
    f_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "forecast_days": input.days,
        "current": "temperature_2m,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
    }

    with httpx.Client(timeout=20) as cx:
        fr = cx.get(forecast_url, params=f_params)
        fr.raise_for_status()
        fj = fr.json()

    # current
    curr = None
    if "current" in fj:
        c = fj["current"]
        curr = WeatherCurrent(
            temperature_c=c.get("temperature_2m"),
            wind_speed_ms=(
                None if c.get("wind_speed_10m") is None
                else float(c.get("wind_speed_10m")) / 3.6  # км/ч → м/с (если вернётся в км/ч)
            ),
            weather_code=c.get("weather_code"),
            weather_text=_wmo_text(c.get("weather_code")),
        )

    # daily
    daily = []
    d = fj.get("daily") or {}
    dates = d.get("time") or []
    tmax = d.get("temperature_2m_max") or []
    tmin = d.get("temperature_2m_min") or []
    pprob = d.get("precipitation_probability_max") or []
    wcode = d.get("weather_code") or []
    for i, dt in enumerate(dates):
        daily.append(WeatherDaily(
            date=dt,
            t_min_c=(tmin[i] if i < len(tmin) else None),
            t_max_c=(tmax[i] if i < len(tmax) else None),
            precip_prob_max_pct=(pprob[i] if i < len(pprob) else None),
            weather_code=(wcode[i] if i < len(wcode) else None),
            weather_text=_wmo_text(wcode[i] if i < len(wcode) else None),
        ))

    return WeatherOutput(
        query=city_q,
        city=disp_name,
        country_code=country_code,
        latitude=lat,
        longitude=lon,
        timezone=fj.get("timezone") or tz,
        current=curr,
        daily=daily,
    )

    

# ---------- ENTRY ----------
if __name__ == "__main__":
    mcp.run(transport="http", port=3333, host="0.0.0.0")
