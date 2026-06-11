"""
app.py
======
BTC Funding Rate & Open Interest Dashboard (Binance USDT-M Futures).

Аналитический дашборд (НЕ торговый бот): показывает два ключевых
индикатора деривативного рынка и объясняет их человеческим языком:

    Funding Rate  — кто платит за удержание позиции (перекос лонг/шорт);
    Open Interest — сколько денег сидит в открытых позициях (поток капитала).

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
    page_title="BTC Funding & OI Dashboard",
    page_icon="💸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
FAPI = "https://fapi.binance.com"

# Пороги funding rate (ставка за 8 часов, доля: 0.0001 = 0.01%)
FUNDING_BASELINE = 0.0001   # «нейтральная» ставка Binance
FUNDING_HIGH = 0.0003       # перегрев лонгов
FUNDING_EXTREME = 0.001     # экстремальный перекос — риск лонг-сквиза

# Пороги изменения Open Interest за 24 часа, %
OI_NOTABLE = 5.0
OI_STRONG = 12.0

# Периоды истории OI, поддерживаемые Binance (и совпадающие с интервалами klines)
OI_PERIODS = ["15m", "30m", "1h", "2h", "4h", "1d"]

# Цвета светофора — те же, что в btc_rsi_dashboard
TRAFFIC_COLORS = {
    "green": ("#16a34a", "🟢", "Рынок сбалансирован"),
    "yellow": ("#d97706", "🟡", "Перекос — осторожно"),
    "red": ("#dc2626", "🔴", "Перегрев — риск сквиза"),
}

COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_OI = "#60a5fa"
COLOR_PRICE = "#f59e0b"


class DataSourceError(Exception):
    """Единый тип ошибки загрузки данных — UI ловит именно его."""


# ---------------------------------------------------------------------------
# Загрузка данных: Binance Futures REST + офлайн demo-fallback
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Все данные одного обновления: funding, OI, цена + метка источника."""
    funding_df: pd.DataFrame    # timestamp, rate — история funding (шаг 8ч)
    oi_df: pd.DataFrame         # timestamp, oi, oi_usd — история Open Interest
    price_df: pd.DataFrame      # timestamp, close — цена тем же периодом, что OI
    funding_now: float          # текущая ставка (доля, не %)
    next_funding_ts: pd.Timestamp
    mark_price: float
    source_name: str


def _get_json(path: str, params: dict) -> list | dict:
    """GET к публичному API Binance Futures с приведением ошибок к одному типу."""
    try:
        resp = requests.get(f"{FAPI}{path}", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataSourceError(f"Ошибка запроса {path}: {exc}") from exc
    if isinstance(data, dict) and "code" in data and "msg" in data:
        raise DataSourceError(f"Binance вернул ошибку: {data}")
    return data


def _fetch_binance(symbol: str, period: str) -> Snapshot:
    """Реальные данные Binance USDT-M Futures (BTCUSDT и т.п., без слэша)."""
    sym = symbol.replace("/", "").upper()

    # Текущая ставка и mark price
    premium = _get_json("/fapi/v1/premiumIndex", {"symbol": sym})

    # История funding: 3 выплаты в сутки, 240 записей ~ 80 дней
    raw_funding = _get_json("/fapi/v1/fundingRate", {"symbol": sym, "limit": 240})
    funding_df = pd.DataFrame(raw_funding)
    funding_df["timestamp"] = pd.to_datetime(funding_df["fundingTime"], unit="ms", utc=True)
    funding_df["rate"] = pd.to_numeric(funding_df["fundingRate"], errors="coerce")
    funding_df = funding_df[["timestamp", "rate"]].dropna().sort_values("timestamp")

    # История Open Interest (Binance хранит максимум ~30 дней)
    raw_oi = _get_json("/futures/data/openInterestHist",
                       {"symbol": sym, "period": period, "limit": 500})
    oi_df = pd.DataFrame(raw_oi)
    oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
    oi_df["oi"] = pd.to_numeric(oi_df["sumOpenInterest"], errors="coerce")
    oi_df["oi_usd"] = pd.to_numeric(oi_df["sumOpenInterestValue"], errors="coerce")
    oi_df = oi_df[["timestamp", "oi", "oi_usd"]].dropna().sort_values("timestamp")

    # Цена тем же периодом, что OI, — чтобы наложить на один график
    raw_klines = _get_json("/fapi/v1/klines",
                           {"symbol": sym, "interval": period, "limit": 500})
    price_df = pd.DataFrame(
        [(row[0], row[4]) for row in raw_klines], columns=["timestamp", "close"]
    )
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True)
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df = price_df.dropna().sort_values("timestamp")

    if funding_df.empty or oi_df.empty or price_df.empty:
        raise DataSourceError("Получен пустой набор данных.")

    return Snapshot(
        funding_df=funding_df,
        oi_df=oi_df,
        price_df=price_df,
        funding_now=float(premium["lastFundingRate"]),
        next_funding_ts=pd.to_datetime(int(premium["nextFundingTime"]), unit="ms", utc=True),
        mark_price=float(premium["markPrice"]),
        source_name="binance-futures",
    )


_PERIOD_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def _demo_snapshot(period: str) -> Snapshot:
    """
    Синтетические данные, если биржа недоступна (сеть, гео-блокировка).
    Интерфейс всегда можно открыть; в UI явно помечается как DEMO.
    """
    rng = np.random.default_rng(seed=int(time.time() // 3600))
    now = pd.Timestamp.now(tz="UTC").floor("min")

    # Funding: шум вокруг базовой ставки с редкими «перекосами»
    n_f = 90
    f_idx = pd.date_range(end=now.floor("8h"), periods=n_f, freq="8h", tz="UTC")
    rates = FUNDING_BASELINE + rng.normal(0, 0.00015, n_f) + 0.0004 * np.sin(
        np.linspace(0, 6, n_f))
    funding_df = pd.DataFrame({"timestamp": f_idx, "rate": rates})

    # Цена: геометрическое случайное блуждание вокруг 100k
    minutes = _PERIOD_MINUTES.get(period, 60)
    n = 400
    p_idx = pd.date_range(end=now.floor(f"{minutes}min"), periods=n,
                          freq=f"{minutes}min", tz="UTC")
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.004 * np.sqrt(minutes / 60), n)))
    price_df = pd.DataFrame({"timestamp": p_idx, "close": close})

    # OI: случайное блуждание, слабо скоррелированное с ценой
    oi = 80_000 * np.exp(np.cumsum(
        rng.normal(0, 0.003, n) + 0.15 * np.diff(np.log(close), prepend=np.log(close[0]))))
    oi_df = pd.DataFrame({"timestamp": p_idx, "oi": oi, "oi_usd": oi * close})

    return Snapshot(
        funding_df=funding_df,
        oi_df=oi_df,
        price_df=price_df,
        funding_now=float(rates[-1]),
        next_funding_ts=now.ceil("8h"),
        mark_price=float(close[-1]),
        source_name="demo",
    )


@st.cache_data(ttl=120, show_spinner="Загружаю funding и Open Interest...")
def load_snapshot(symbol: str, period: str) -> Snapshot:
    """
    Кэшированная загрузка (ttl=120с — свежо и не дергает биржу на каждый клик).
    Если Binance недоступен — офлайн demo-данные, дашборд не «умирает».
    """
    try:
        return _fetch_binance(symbol, period)
    except DataSourceError:
        return _demo_snapshot(period)


# ---------------------------------------------------------------------------
# Анализ: зоны, режим рынка, светофор
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """Качественная оценка одного индикатора: подпись + цвет + эмодзи."""
    label: str
    color: str
    emoji: str
    level: str  # green / yellow / red — вклад в общий светофор


@dataclass
class Report:
    """Итог анализа — все, что нужно UI, в одном месте."""
    funding_now: float
    funding_avg_7d: float
    funding_zone: Zone
    oi_now: float
    oi_usd_now: float
    oi_change_24h: float
    oi_zone: Zone
    price_now: float
    price_change_24h: float
    regime_title: str
    regime_desc: str
    traffic_light: str
    headline: str
    explanation_lines: list[str]


def _classify_funding(rate: float) -> Zone:
    """Зона funding rate: знак и величина перекоса."""
    if rate >= FUNDING_EXTREME:
        return Zone("Экстремально высокий", COLOR_DOWN, "🔥", "red")
    if rate >= FUNDING_HIGH:
        return Zone("Повышенный — лонги переплачивают", "#d97706", "⚠️", "yellow")
    if rate >= -FUNDING_BASELINE:
        return Zone("Нормальный", COLOR_UP, "✅", "green")
    if rate >= -FUNDING_HIGH:
        return Zone("Отрицательный — шорты платят", "#d97706", "⚠️", "yellow")
    return Zone("Сильно отрицательный", COLOR_DOWN, "🔥", "red")


def _classify_oi_change(change_pct: float) -> Zone:
    """Зона изменения OI за 24ч: скорость притока/оттока капитала."""
    if abs(change_pct) >= OI_STRONG:
        direction = "приток" if change_pct > 0 else "отток"
        return Zone(f"Резкий {direction} капитала", COLOR_DOWN, "🔥", "red")
    if abs(change_pct) >= OI_NOTABLE:
        direction = "растет" if change_pct > 0 else "снижается"
        return Zone(f"OI заметно {direction}", "#d97706", "⚠️", "yellow")
    return Zone("OI стабилен", COLOR_UP, "✅", "green")


def _value_24h_ago(df: pd.DataFrame, column: str) -> float:
    """Значение колонки ~24 часа назад (ближайшая запись к этой отметке)."""
    target = df["timestamp"].iloc[-1] - pd.Timedelta(hours=24)
    idx = (df["timestamp"] - target).abs().idxmin()
    return float(df.loc[idx, column])


def _detect_regime(price_chg: float, oi_chg: float) -> tuple[str, str]:
    """
    Классическая матрица «цена x OI»: куда идут деньги.
    Пороги ±0.3% (цена) и ±1% (OI) отсекают шум боковика.
    """
    price_up, price_down = price_chg > 0.3, price_chg < -0.3
    oi_up, oi_down = oi_chg > 1.0, oi_chg < -1.0

    if price_up and oi_up:
        return ("Рост на новых деньгах",
                "Цена и OI растут: в рынок заходят новые лонги — тренд подтвержден капиталом.")
    if price_up and oi_down:
        return ("Рост на закрытии шортов",
                "Цена растет, а OI падает: рост питается выкупом шортов — без новых денег он хрупкий.")
    if price_down and oi_up:
        return ("Наращивание шортов",
                "Цена падает, а OI растет: открываются новые шорты — давление вниз усиливается.")
    if price_down and oi_down:
        return ("Разгрузка лонгов",
                "Цена и OI падают: лонги закрываются/ликвидируются — продавцы выдыхаются по мере падения OI.")
    return ("Нет выраженного потока капитала",
            "Связка «цена x OI» за сутки не сложилась: значимого притока "
            "или оттока денег в позиции нет.")


def compute_report(snap: Snapshot) -> Report:
    """Сводит funding, OI и цену в единый отчет со светофором."""
    funding_avg_7d = float(snap.funding_df["rate"].tail(21).mean())  # 21 выплата = 7 дней
    funding_zone = _classify_funding(snap.funding_now)

    oi_now = float(snap.oi_df["oi"].iloc[-1])
    oi_usd_now = float(snap.oi_df["oi_usd"].iloc[-1])
    oi_change_24h = (oi_now / _value_24h_ago(snap.oi_df, "oi") - 1) * 100
    oi_zone = _classify_oi_change(oi_change_24h)

    price_now = float(snap.price_df["close"].iloc[-1])
    price_change_24h = (price_now / _value_24h_ago(snap.price_df, "close") - 1) * 100

    regime_title, regime_desc = _detect_regime(price_change_24h, oi_change_24h)

    # Светофор: худшая из двух зон; одновременный перекос обеих — всегда red
    levels = {"green": 0, "yellow": 1, "red": 2}
    worst = max(funding_zone.level, oi_zone.level, key=lambda lv: levels[lv])
    if funding_zone.level != "green" and oi_zone.level != "green":
        worst = "red"

    if worst == "red":
        headline = "Сильный перекос позиций — высокая вероятность резкого движения (сквиза)."
    elif worst == "yellow":
        headline = "Рынок начинает перекашиваться — следить за funding и OI внимательнее."
    else:
        headline = "Funding у базовой ставки, OI без рывков — деривативы спокойны."

    annual = snap.funding_now * 3 * 365 * 100
    lines = [
        f"Funding сейчас {snap.funding_now * 100:+.4f}% за 8ч ({annual:+.1f}% годовых) — "
        f"{funding_zone.label.lower()}. "
        + ("Лонги платят шортам: преобладают покупатели с плечом."
           if snap.funding_now > 0 else
           "Шорты платят лонгам: преобладают продавцы с плечом."),
        f"Средний funding за 7 дней: {funding_avg_7d * 100:+.4f}% — "
        + ("текущая ставка выше средней, перекос нарастает."
           if abs(snap.funding_now) > abs(funding_avg_7d) * 1.3
           else "текущая ставка в рамках недельной нормы."),
        f"Open Interest: {oi_now:,.0f} BTC (${oi_usd_now / 1e9:,.2f} млрд), "
        f"изменение за 24ч: {oi_change_24h:+.2f}% — {oi_zone.label}.",
        f"Режим «{regime_title}»: {regime_desc}",
    ]
    if worst == "red":
        side = "лонг" if snap.funding_now > 0 else "шорт"
        lines.append(f"⚠️ Толпа перекошена в {side}: именно против нее чаще всего случается сквиз.")

    return Report(
        funding_now=snap.funding_now,
        funding_avg_7d=funding_avg_7d,
        funding_zone=funding_zone,
        oi_now=oi_now,
        oi_usd_now=oi_usd_now,
        oi_change_24h=oi_change_24h,
        oi_zone=oi_zone,
        price_now=price_now,
        price_change_24h=price_change_24h,
        regime_title=regime_title,
        regime_desc=regime_desc,
        traffic_light=worst,
        headline=headline,
        explanation_lines=lines,
    )


# ---------------------------------------------------------------------------
# UI-компоненты
# ---------------------------------------------------------------------------
def render_header(symbol: str, snap: Snapshot, report: Report) -> None:
    """Верхняя панель: цена, режим рынка, светофор."""
    color, emoji, traffic_text = TRAFFIC_COLORS[report.traffic_light]

    col_price, col_signal, col_light = st.columns([1.2, 1.4, 1])
    with col_price:
        st.metric(
            label=f"{symbol} · источник: {snap.source_name}",
            value=f"${report.price_now:,.0f}",
            delta=f"{report.price_change_24h:+.2f}% за 24ч",
        )
    with col_signal:
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 8px 14px;
                        background: rgba(128,128,128,0.08); border-radius: 6px;">
                <div style="font-size: 1.4rem; font-weight: 700;">{report.regime_title}</div>
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

    if snap.source_name == "demo":
        st.warning("⚠️ Показаны СИНТЕТИЧЕСКИЕ демо-данные (биржа недоступна). "
                   "Не используйте их для анализа реального рынка.")


def _indicator_card(title: str, value: str, zone: Zone, sub: str, note: str) -> str:
    """HTML карточки индикатора — в стиле карточек таймфреймов RSI-дашборда."""
    return f"""
    <div style="border: 1px solid rgba(128,128,128,0.25); border-top: 5px solid {zone.color};
                border-radius: 10px; padding: 14px; height: 100%;">
        <div style="font-size:1.05rem; font-weight:700;">{title}</div>
        <div style="font-size:2rem; font-weight:700; color:{zone.color}; margin: 4px 0;">
            {value}
        </div>
        <div style="margin: 4px 0;">{zone.emoji} {zone.label}</div>
        <div style="font-size:0.85rem; opacity:0.8;">{sub}</div>
        <div style="font-size:0.8rem; opacity:0.7; margin-top:6px;">{note}</div>
    </div>
    """


def render_indicator_cards(snap: Snapshot, report: Report) -> None:
    """Две карточки: Funding Rate и Open Interest — суть дашборда одним взглядом."""
    seconds_left = (snap.next_funding_ts - pd.Timestamp.now(tz="UTC")).total_seconds()
    minutes_left = max(0, int(seconds_left // 60))
    countdown = f"{minutes_left // 60}ч {minutes_left % 60:02d}м"
    annual = report.funding_now * 3 * 365 * 100

    col_f, col_oi = st.columns(2)
    with col_f:
        st.markdown(_indicator_card(
            "Funding Rate (8ч)",
            f"{report.funding_now * 100:+.4f}%",
            report.funding_zone,
            f"≈ {annual:+.1f}% годовых · среднее за 7д: {report.funding_avg_7d * 100:+.4f}%",
            f"Следующая выплата через {countdown}. Положительный — платят лонги, "
            f"отрицательный — шорты.",
        ), unsafe_allow_html=True)
    with col_oi:
        st.markdown(_indicator_card(
            "Open Interest",
            f"{report.oi_now:,.0f} BTC",
            report.oi_zone,
            f"${report.oi_usd_now / 1e9:,.2f} млрд · за 24ч: {report.oi_change_24h:+.2f}%",
            "Объем открытых позиций. Рост OI = в рынок заходят новые деньги, "
            "падение = позиции закрываются.",
        ), unsafe_allow_html=True)


def render_funding_chart(funding_df: pd.DataFrame, bars: int = 90) -> None:
    """История funding rate: бары по знаку + базовая и «перегретая» отметки."""
    df = funding_df.tail(bars)
    pct = df["rate"] * 100
    colors = [COLOR_UP if r >= 0 else COLOR_DOWN for r in df["rate"]]

    fig = go.Figure(go.Bar(x=df["timestamp"], y=pct, marker_color=colors, name="Funding"))
    fig.add_hline(y=FUNDING_BASELINE * 100, line_dash="dash", line_color="#9ca3af",
                  opacity=0.7, annotation_text="базовая 0.01%",
                  annotation_font_size=11)
    for level in (FUNDING_HIGH, -FUNDING_HIGH):
        fig.add_hline(y=level * 100, line_dash="dot", line_color="#d97706", opacity=0.6)
    fig.update_layout(
        title=f"Funding Rate, % за 8ч · последние {bars} выплат (~{bars // 3} дней)",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False, yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Зеленые бары — платят лонги, красные — шорты. Пунктир 0.01% — "
               "нейтральная ставка; точечные ±0.03% — зона перегрева.")


def render_oi_chart(oi_df: pd.DataFrame, price_df: pd.DataFrame, period: str) -> None:
    """Open Interest (заливка) + цена (линия) на общей оси времени."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=oi_df["timestamp"], y=oi_df["oi"], name="Open Interest",
        mode="lines", line=dict(color=COLOR_OI, width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.15)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=price_df["timestamp"], y=price_df["close"], name="Цена",
        mode="lines", line=dict(color=COLOR_PRICE, width=1.5),
    ), secondary_y=True)

    fig.update_yaxes(title_text="OI, BTC", secondary_y=False,
                     rangemode="tozero", showgrid=False)
    fig.update_yaxes(title_text="Цена, $", secondary_y=True, showgrid=False)
    fig.update_layout(
        title=f"Open Interest и цена · период {period}",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Расхождение линий — главный сигнал: цена вверх при падающем OI — "
               "рост без новых денег; цена вниз при растущем OI — давят свежие шорты.")


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
        "Funding и OI показывают ПОЗИЦИОНИРОВАНИЕ рынка, а не точку входа. "
        "Любая сделка требует собственного риск-менеджмента. Это не финансовый совет."
    )


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.title("💸 Funding & OI")
        st.caption("Позиционирование рынка BTC по деривативам. "
                   "Только аналитика — без автоторговли.")

        symbol = st.text_input("Контракт (USDT-M)", value="BTCUSDT").strip().upper()
        period = st.selectbox("Период графика OI", OI_PERIODS, index=2)

        if st.button("🔄 Обновить данные", use_container_width=True):
            load_snapshot.clear()  # сброс кэша -> следующий вызов пойдет на биржу

        st.divider()
        st.markdown(
            """
            **Как читать светофор**

            🟢 — funding у нормы, OI стабилен
            🟡 — перекос нарастает, следить
            🔴 — сильный перекос, риск сквиза
            """
        )
        st.caption("Не является финансовым советом.")

    snap = load_snapshot(symbol, period)
    try:
        report = compute_report(snap)
    except (ValueError, KeyError, IndexError) as exc:
        st.error(f"Ошибка расчета индикаторов: {exc}")
        st.stop()

    render_header(symbol, snap, report)
    st.divider()

    render_indicator_cards(snap, report)
    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        render_funding_chart(snap.funding_df)
    with col_right:
        render_oi_chart(snap.oi_df, snap.price_df, period)

    render_explanation(report)


if __name__ == "__main__":
    main()
