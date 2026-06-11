"""
app.py
======
BTC CVD Dashboard — Cumulative Volume Delta (Binance USDT-M Futures).

Аналитический дашборд (НЕ торговый бот): показывает, кто прямо сейчас
двигает цену рыночными ордерами — агрессивные покупатели или продавцы.

    Delta = объем маркет-покупок − объем маркет-продаж (за бар);
    CVD   = накопленная сумма дельты.

Главный сигнал — дивергенция цены и CVD:
    цена обновила минимум, CVD — нет  -> продавцы выдыхаются (бычья);
    цена обновила максимум, CVD — нет -> рост без покупок (медвежья).

Анализ ведется на четырех таймфреймах (как в RSI-дашборде):
    15m — ранний сигнал потока
    1h  — подтверждение импульса
    4h  — основной контекст
    1d  — глобальный поток

Скрипт полностью автономен: один файл, никаких импортов из других
модулей проекта. API-ключи не нужны — только публичные эндпоинты.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="BTC CVD Dashboard",
    page_icon="📶",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
FAPI = "https://fapi.binance.com"

# Таймфреймы и их роли (от младшего к старшему)
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
TF_ROLES = {
    "15m": "Ранний сигнал потока",
    "1h": "Подтверждение импульса",
    "4h": "Основной контекст",
    "1d": "Глобальный поток",
}
TF_COLORS = {"15m": "#60a5fa", "1h": "#a78bfa", "4h": "#f59e0b", "1d": "#ef4444"}

DEFAULT_LIMIT = 500     # свечей на таймфрейм
DIV_WINDOW = 80         # окно поиска экстремумов для дивергенций, баров
DIV_RECENT = 8          # «свежий» экстремум = в последних N барах
PRESSURE_BARS = 20      # окно расчета доли агрессивных покупок

COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"

# Цвета светофора — те же, что в соседних дашбордах
TRAFFIC_COLORS = {
    "green": ("#16a34a", "🟢", "Поток подтверждает цену"),
    "yellow": ("#d97706", "🟡", "Дивергенция на младших ТФ"),
    "red": ("#dc2626", "🔴", "Поток против цены"),
}


class DataSourceError(Exception):
    """Единый тип ошибки загрузки данных — UI ловит именно его."""


# ---------------------------------------------------------------------------
# Загрузка данных: Binance Futures REST + офлайн demo-fallback
# ---------------------------------------------------------------------------
def _fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Klines Binance Futures. Ключевое поле — taker buy volume (индекс 9):
    объем, купленный агрессорами (маркет-байи). Именно из него считается дельта.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(f"{FAPI}/fapi/v1/klines", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataSourceError(f"Ошибка запроса klines {interval}: {exc}") from exc
    if isinstance(data, dict):
        raise DataSourceError(f"Binance вернул ошибку: {data}")

    df = pd.DataFrame(
        [(row[0], row[4], row[5], row[9]) for row in data],
        columns=["timestamp", "close", "volume", "taker_buy"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("close", "volume", "taker_buy"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise DataSourceError("Получен пустой набор свечей.")

    # Дельта бара: покупки − продажи = taker_buy − (volume − taker_buy)
    df["delta"] = 2 * df["taker_buy"] - df["volume"]
    df["cvd"] = df["delta"].cumsum()  # CVD от начала загруженного окна
    return df


_TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _demo_frames(limit: int) -> dict[str, pd.DataFrame]:
    """
    Синтетические данные, если биржа недоступна (сеть, гео-блокировка).
    Доля покупок слабо привязана к доходности, поэтому CVD в целом следует
    за ценой, но периодически расходится — дивергенции есть что показать.
    """
    rng = np.random.default_rng(seed=int(time.time() // 3600))
    frames: dict[str, pd.DataFrame] = {}
    for tf, minutes in _TF_MINUTES.items():
        end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
        idx = pd.date_range(end=end, periods=limit, freq=f"{minutes}min", tz="UTC")

        returns = rng.normal(0, 0.004 * np.sqrt(minutes / 60), limit)
        close = 100_000 * np.exp(np.cumsum(returns))
        volume = rng.uniform(200, 2000, limit)
        buy_share = np.clip(0.5 + 8 * returns + rng.normal(0, 0.06, limit), 0.15, 0.85)

        df = pd.DataFrame({
            "timestamp": idx,
            "close": close,
            "volume": volume,
            "taker_buy": volume * buy_share,
        })
        df["delta"] = 2 * df["taker_buy"] - df["volume"]
        df["cvd"] = df["delta"].cumsum()
        frames[tf] = df
    return frames


@st.cache_data(ttl=120, show_spinner="Загружаю поток ордеров с биржи...")
def load_all(symbol: str, limit: int = DEFAULT_LIMIT) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Кэшированная загрузка всех таймфреймов (ttl=120с — свежо и не дергает
    биржу на каждый клик). Если Binance недоступен — офлайн demo-данные.
    """
    sym = symbol.replace("/", "").upper()
    try:
        return {tf: _fetch_klines(sym, tf, limit) for tf in TIMEFRAMES}, "binance-futures"
    except DataSourceError:
        return _demo_frames(limit), "demo"


# ---------------------------------------------------------------------------
# Анализ: статусы по таймфреймам, дивергенции, общий вердикт
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """Качественная оценка статуса: подпись + цвет + эмодзи + уровень тревоги."""
    label: str
    color: str
    emoji: str
    level: str  # green / yellow / red — вклад в общий светофор


STATUS_META = {
    "confirm_up": Zone("Рост подтвержден покупками", COLOR_UP, "✅", "green"),
    "confirm_down": Zone("Падение подтверждено продажами", COLOR_DOWN, "🔻", "green"),
    "bull_div": Zone("Бычья дивергенция", "#60a5fa", "🐂", "yellow"),
    "bear_div": Zone("Медвежья дивергенция", "#f59e0b", "🐻", "yellow"),
    "neutral": Zone("Поток нейтрален", "#9ca3af", "➖", "green"),
}

STATUS_EXPLAIN = {
    "confirm_up": "Свежий максимум цены подкреплен агрессивными покупками — движение «честное».",
    "confirm_down": "Свежий минимум цены подкреплен агрессивными продажами — давление вниз реальное.",
    "bull_div": "Цена обновила минимум, а CVD — нет: продавцы выдыхаются, назревает отскок/разворот вверх.",
    "bear_div": "Цена обновила максимум, а CVD — нет: рост без поддержки покупок, риск разворота вниз.",
    "neutral": "Свежих экстремумов нет — поток без явного сигнала.",
}


@dataclass
class TFAnalysis:
    """Результат анализа одного таймфрейма."""
    timeframe: str
    role: str
    status: str          # ключ STATUS_META
    zone: Zone
    buy_share: float     # доля агрессивных покупок за PRESSURE_BARS баров, %
    delta_window: float  # суммарная дельта за это же окно, BTC
    cvd_slope: str       # ↑ / ↓ / → — направление CVD за окно
    explanation: str


@dataclass
class Report:
    """Итог анализа — все, что нужно UI, в одном месте."""
    per_timeframe: dict[str, TFAnalysis]
    price_now: float
    price_change_24h: float
    delta_24h: float       # суммарная дельта за сутки (по 15m), BTC
    traffic_light: str
    verdict_title: str
    headline: str
    explanation_lines: list[str]


def _detect_status(close: pd.Series, cvd: pd.Series,
                   window: int = DIV_WINDOW, recent: int = DIV_RECENT) -> str:
    """
    Сравнивает свежие экстремумы цены и CVD в окне `window` баров.
    Экстремум «свежий», если попал в последние `recent` баров.
    Дивергенция = цена обновила экстремум, а CVD свой — нет.
    """
    c = close.tail(window).to_numpy()
    v = cvd.tail(window).to_numpy()
    fresh = len(c) - recent
    p_low, p_high = c.argmin() >= fresh, c.argmax() >= fresh
    v_low, v_high = v.argmin() >= fresh, v.argmax() >= fresh

    if p_low and not v_low:
        return "bull_div"
    if p_high and not v_high:
        return "bear_div"
    if p_high and v_high:
        return "confirm_up"
    if p_low and v_low:
        return "confirm_down"
    return "neutral"


def _analyze_tf(tf: str, df: pd.DataFrame) -> TFAnalysis:
    """Полный разбор одного таймфрейма: статус, давление покупок, дельта."""
    status = _detect_status(df["close"], df["cvd"])

    tail = df.tail(PRESSURE_BARS)
    buy_share = float(tail["taker_buy"].sum() / tail["volume"].sum() * 100)
    delta_window = float(tail["delta"].sum())

    cvd_chg = float(df["cvd"].iloc[-1] - df["cvd"].iloc[-PRESSURE_BARS])
    threshold = float(df["delta"].abs().tail(PRESSURE_BARS).mean())  # шумовой порог
    cvd_slope = "↑" if cvd_chg > threshold else ("↓" if cvd_chg < -threshold else "→")

    return TFAnalysis(
        timeframe=tf,
        role=TF_ROLES[tf],
        status=status,
        zone=STATUS_META[status],
        buy_share=buy_share,
        delta_window=delta_window,
        cvd_slope=cvd_slope,
        explanation=STATUS_EXPLAIN[status],
    )


def compute_report(frames: dict[str, pd.DataFrame]) -> Report:
    """Сводит таймфреймы в общий вердикт со светофором."""
    per_tf = {tf: _analyze_tf(tf, frames[tf]) for tf in TIMEFRAMES}

    df15 = frames["15m"]
    price_now = float(df15["close"].iloc[-1])
    lookback = min(96, len(df15) - 1)  # 96 свечей по 15 минут ~ сутки
    price_change_24h = (price_now / float(df15["close"].iloc[-1 - lookback]) - 1) * 100
    delta_24h = float(df15["delta"].tail(lookback).sum())

    # Вердикт: дивергенция на старших ТФ важнее всего (4h — контекст, 1d — глобально)
    divs = {"bull_div", "bear_div"}
    senior_div = [per_tf[tf] for tf in ("4h", "1d") if per_tf[tf].status in divs]
    junior_div = [per_tf[tf] for tf in ("15m", "1h") if per_tf[tf].status in divs]

    if senior_div:
        a = senior_div[-1]  # приоритет 1d
        direction = "вверх" if a.status == "bull_div" else "вниз"
        traffic = "red"
        verdict_title = f"{a.zone.label} на {a.timeframe}"
        headline = (f"Поток ордеров на старшем ТФ расходится с ценой — "
                    f"высокая вероятность разворота {direction}.")
    elif junior_div:
        a = junior_div[-1]
        traffic = "yellow"
        verdict_title = f"Ранняя {a.zone.label.lower()} на {a.timeframe}"
        headline = ("Младший ТФ показывает дивергенцию — ждать подтверждения "
                    "на 4h/1d, пока это только ранний сигнал.")
    else:
        statuses = {a.status for a in per_tf.values()}
        traffic = "green"
        if statuses == {"confirm_up"}:
            verdict_title = "Рост подтвержден потоком"
            headline = "Все таймфреймы: новые максимумы цены выкупаются агрессорами."
        elif statuses == {"confirm_down"}:
            verdict_title = "Падение подтверждено потоком"
            headline = "Все таймфреймы: новые минимумы цены продавлены агрессорами."
        else:
            verdict_title = "Дивергенций нет"
            headline = "Поток ордеров не противоречит цене — явных разворотных сигналов нет."

    side = "покупателей" if delta_24h > 0 else "продавцов"
    lines = [
        f"Дельта за 24ч: {delta_24h:+,.0f} BTC — суммарный перевес агрессивных {side}.",
        *(f"{a.timeframe} ({a.role.lower()}): {a.zone.emoji} {a.zone.label.lower()}, "
          f"CVD {a.cvd_slope}, покупки {a.buy_share:.1f}% объема. {a.explanation}"
          for a in per_tf.values()),
    ]

    return Report(
        per_timeframe=per_tf,
        price_now=price_now,
        price_change_24h=price_change_24h,
        delta_24h=delta_24h,
        traffic_light=traffic,
        verdict_title=verdict_title,
        headline=headline,
        explanation_lines=lines,
    )


# ---------------------------------------------------------------------------
# UI-компоненты
# ---------------------------------------------------------------------------
def render_header(symbol: str, report: Report, source_name: str) -> None:
    """Верхняя панель: цена, вердикт по потоку, светофор."""
    color, emoji, traffic_text = TRAFFIC_COLORS[report.traffic_light]

    col_price, col_signal, col_light = st.columns([1.2, 1.4, 1])
    with col_price:
        st.metric(
            label=f"{symbol} · источник: {source_name}",
            value=f"${report.price_now:,.0f}",
            delta=f"{report.price_change_24h:+.2f}% за 24ч",
        )
    with col_signal:
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 8px 14px;
                        background: rgba(128,128,128,0.08); border-radius: 6px;">
                <div style="font-size: 1.4rem; font-weight: 700;">{report.verdict_title}</div>
                <div style="opacity: 0.85;">{report.headline}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_light:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top: 4px;">
                <div style="font-size: 3rem; line-height: 1;">{emoji}</div>
                <div style="font-weight: 600; color: {color};">{traffic_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if source_name == "demo":
        st.warning("⚠️ Показаны СИНТЕТИЧЕСКИЕ демо-данные (биржа недоступна). "
                   "Не используйте их для анализа реального рынка.")


def _card_html(a: TFAnalysis) -> str:
    """HTML карточки таймфрейма — в стиле карточек RSI-дашборда."""
    share_color = COLOR_UP if a.buy_share >= 50 else COLOR_DOWN
    return f"""
    <div style="border: 1px solid rgba(128,128,128,0.25); border-top: 5px solid {a.zone.color};
                border-radius: 10px; padding: 14px; height: 100%;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="font-size:1.2rem; font-weight:700;">{a.timeframe}</span>
            <span style="opacity:0.7; font-size:0.8rem;">CVD {a.cvd_slope}</span>
        </div>
        <div style="opacity:0.75; font-size:0.85rem; margin-bottom:6px;">{a.role}</div>
        <div style="font-size:1.5rem; font-weight:700; color:{share_color};">
            {a.buy_share:.1f}% <span style="font-size:0.9rem; opacity:0.8;">покупки</span>
        </div>
        <div style="margin: 4px 0;">{a.zone.emoji} {a.zone.label}</div>
        <div style="font-size:0.8rem; opacity:0.8; margin-top:6px;">
            Дельта за {PRESSURE_BARS} баров: {a.delta_window:+,.0f} BTC
        </div>
        <div style="font-size:0.85rem; margin-top:8px; opacity:0.9;">{a.explanation}</div>
    </div>
    """


def render_timeframe_cards(report: Report) -> None:
    """Ряд карточек 15m / 1h / 4h / 1d."""
    st.subheader("Таймфреймы")
    cols = st.columns(len(report.per_timeframe))
    for col, a in zip(cols, report.per_timeframe.values()):
        with col:
            st.markdown(_card_html(a), unsafe_allow_html=True)


def render_cvd_grid(frames: dict[str, pd.DataFrame],
                    report: Report, bars: int = 150) -> None:
    """
    CVD и цена по таймфреймам — сетка 2x2, каждому ТФ своя панель.

    Как и в RSI-дашборде, общая ось времени невозможна: окна ТФ покрывают
    разные периоды. В каждой панели CVD (цветная линия, левая ось) наложен
    на цену (серая линия, правая ось) — дивергенция видна как «ножницы».
    """
    titles = [
        f"<b>{tf}</b> · {report.per_timeframe[tf].zone.emoji} "
        f"{report.per_timeframe[tf].zone.label}"
        for tf in TIMEFRAMES
    ]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=titles,
        specs=[[{"secondary_y": True}] * 2] * 2,
        vertical_spacing=0.22, horizontal_spacing=0.09,
    )

    for i, tf in enumerate(TIMEFRAMES):
        row, col = i // 2 + 1, i % 2 + 1
        df = frames[tf].tail(bars)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["cvd"], name=f"CVD {tf}",
            mode="lines", line=dict(color=TF_COLORS[tf], width=2),
            showlegend=False,
        ), row=row, col=col, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["close"], name=f"Цена {tf}",
            mode="lines", line=dict(color="#9ca3af", width=1.2),
            opacity=0.7, showlegend=False,
        ), row=row, col=col, secondary_y=True)

        fig.layout.annotations[i].update(
            font=dict(size=13, color=TF_COLORS[tf]),
        )
        fig.update_yaxes(showgrid=False, row=row, col=col)

    fig.update_layout(
        title=dict(text="CVD (цветная) и цена (серая) по таймфреймам", y=0.98),
        height=460, margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Дивергенция видна как «ножницы»: цена идет на новый экстремум, "
               "а CVD свой не обновляет. Абсолютный уровень CVD не важен — "
               "он считается от начала окна; важна форма относительно цены.")


def render_delta_chart(df: pd.DataFrame, timeframe: str, bars: int = 100) -> None:
    """Дельта по барам выбранного ТФ: кто бил по рынку в каждый момент."""
    tail = df.tail(bars)
    colors = [COLOR_UP if d >= 0 else COLOR_DOWN for d in tail["delta"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=tail["timestamp"], y=tail["delta"], name="Дельта",
        marker_color=colors, opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=tail["timestamp"], y=tail["close"], name="Цена",
        mode="lines", line=dict(color="#9ca3af", width=1.2), opacity=0.7,
    ), secondary_y=True)

    fig.update_yaxes(title_text="Дельта, BTC", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Цена, $", secondary_y=True, showgrid=False)
    fig.update_layout(
        title=f"Дельта по барам · {timeframe}",
        height=460, margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Зеленый бар — агрессивные покупки преобладали, красный — продажи. "
               "Серия крупных красных баров без падения цены = продажи поглощаются "
               "лимитными покупателями (скрытая сила).")


def render_explanation(report: Report) -> None:
    """Блок «что происходит человеческим языком»."""
    st.subheader("Объяснение")
    color, _, _ = TRAFFIC_COLORS[report.traffic_light]
    st.markdown(
        f"<div style='border-left: 6px solid {color}; padding-left: 12px; "
        f"font-weight: 600;'>{report.headline}</div>",
        unsafe_allow_html=True,
    )
    for line in report.explanation_lines:
        st.markdown(f"- {line}")
    st.caption(
        "CVD показывает ПОТОК АГРЕССИВНЫХ ОРДЕРОВ, а не точку входа. "
        "Любая сделка требует собственного риск-менеджмента. Это не финансовый совет."
    )


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.title("📶 CVD Dashboard")
        st.caption("Поток агрессивных ордеров по BTC. "
                   "Только аналитика — без автоторговли.")

        symbol = st.text_input("Контракт (USDT-M)", value="BTCUSDT").strip().upper()
        delta_tf = st.selectbox("Таймфрейм графика дельты", TIMEFRAMES, index=1)

        if st.button("🔄 Обновить данные", use_container_width=True):
            load_all.clear()  # сброс кэша -> следующий вызов пойдет на биржу

        st.divider()
        st.markdown(
            """
            **Как читать светофор**

            🟢 — поток подтверждает движение цены
            🟡 — дивергенция на младших ТФ, ждать
            🔴 — дивергенция на старших ТФ: поток против цены
            """
        )
        st.caption("Не является финансовым советом.")

    frames, source_name = load_all(symbol)
    try:
        report = compute_report(frames)
    except (ValueError, KeyError, IndexError) as exc:
        st.error(f"Ошибка расчета индикаторов: {exc}")
        st.stop()

    render_header(symbol, report, source_name)
    st.divider()

    render_timeframe_cards(report)
    st.divider()

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        render_cvd_grid(frames, report)
    with col_right:
        render_delta_chart(frames[delta_tf], delta_tf)

    render_explanation(report)


if __name__ == "__main__":
    main()
