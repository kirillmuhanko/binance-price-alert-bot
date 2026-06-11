"""
app.py
======
BTC Volume Profile Dashboard (Binance USDT-M Futures).

Аналитический дашборд (НЕ торговый бот): распределяет наторгованный объем
не по времени, а ПО ЦЕНЕ, и показывает, где сидят сильные уровни:

    POC (Point of Control) — цена с максимальным объемом, «магнит» рынка;
    Value Area (VAH..VAL)  — диапазон, где прошло 70% объема;
    HVN / LVN              — узлы высокого/низкого объема: высокие тормозят
                             цену, низкие («вакуум») цена пролетает насквозь.

Профиль строится на трех окнах: 7 / 30 / 90 дней — краткосрочные,
среднесрочные и глобальные уровни.

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
    page_title="BTC Volume Profile Dashboard",
    page_icon="🏔️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
FAPI = "https://fapi.binance.com"

# Окна профиля: подбираем интервал свечей так, чтобы баров было 500-750
WINDOWS = {
    "7д": {"interval": "15m", "bars": 672, "role": "Краткосрочные уровни"},
    "30д": {"interval": "1h", "bars": 720, "role": "Среднесрочные уровни"},
    "90д": {"interval": "4h", "bars": 540, "role": "Глобальные уровни"},
}
MAIN_WINDOW = "30д"        # окно, по которому ставится светофор
PROFILE_BINS = 120         # ценовых корзин в профиле
VALUE_AREA_SHARE = 0.70    # доля объема в Value Area
EDGE_TOLERANCE = 0.01      # «у границы VA» = в пределах 1% от нее
VACUUM_RATIO = 0.40        # бин тоньше 40% медианы профиля = вакуум

WINDOW_COLORS = {"7д": "#60a5fa", "30д": "#f59e0b", "90д": "#ef4444"}

COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_POC = "#f59e0b"

# Цвета светофора — те же, что в соседних дашбордах
TRAFFIC_COLORS = {
    "green": ("#16a34a", "🟢", "Рынок в балансе"),
    "yellow": ("#d97706", "🟡", "Цена вне зоны стоимости"),
    "red": ("#dc2626", "🔴", "Объемный вакуум"),
}


class DataSourceError(Exception):
    """Единый тип ошибки загрузки данных — UI ловит именно его."""


# ---------------------------------------------------------------------------
# Загрузка данных: Binance Futures REST + офлайн demo-fallback
# ---------------------------------------------------------------------------
def _fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """OHLCV с Binance Futures: high/low нужны, чтобы размазать объем по цене."""
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    try:
        resp = requests.get(f"{FAPI}/fapi/v1/klines", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise DataSourceError(f"Ошибка запроса klines {interval}: {exc}") from exc
    if isinstance(data, dict):
        raise DataSourceError(f"Binance вернул ошибку: {data}")

    df = pd.DataFrame(
        [row[:6] for row in data],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise DataSourceError("Получен пустой набор свечей.")
    return df


_INTERVAL_MINUTES = {"15m": 15, "1h": 60, "4h": 240}


def _demo_frames() -> dict[str, pd.DataFrame]:
    """
    Синтетические данные, если биржа недоступна (сеть, гео-блокировка).
    Цена подолгу «пасется» у круглых уровней, чтобы профиль имел узлы объема.
    """
    rng = np.random.default_rng(seed=int(time.time() // 3600))
    frames: dict[str, pd.DataFrame] = {}
    for label, cfg in WINDOWS.items():
        minutes = _INTERVAL_MINUTES[cfg["interval"]]
        n = cfg["bars"]
        end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
        idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")

        # Случайное блуждание с притяжением к «уровню» — образуются узлы объема
        step = 0.004 * np.sqrt(minutes / 60)
        log_p = np.log(100_000) + np.zeros(n)
        anchor = np.log(100_000)
        for i in range(1, n):
            if rng.random() < 0.01:  # изредка уровень-якорь смещается
                anchor += rng.normal(0, 0.05)
            log_p[i] = log_p[i - 1] + rng.normal(0, step) + 0.02 * (anchor - log_p[i - 1])
        close = np.exp(log_p)

        spread = np.abs(rng.normal(0, step, n)) * close
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        frames[label] = pd.DataFrame({
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(200, 2000, n),
        })
    return frames


@st.cache_data(ttl=300, show_spinner="Строю профили объема...")
def load_all(symbol: str) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Кэшированная загрузка всех окон (ttl=300с — уровни меняются медленно).
    Если Binance недоступен — офлайн demo-данные, дашборд не «умирает».
    """
    sym = symbol.replace("/", "").upper()
    try:
        frames = {
            label: _fetch_klines(sym, cfg["interval"], cfg["bars"])
            for label, cfg in WINDOWS.items()
        }
        return frames, "binance-futures"
    except DataSourceError:
        return _demo_frames(), "demo"


# ---------------------------------------------------------------------------
# Анализ: профиль, POC, Value Area, позиция цены
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """Качественная оценка позиции цены: подпись + цвет + эмодзи + уровень."""
    label: str
    color: str
    emoji: str
    level: str  # green / yellow / red — вклад в общий светофор


POSITION_META = {
    "inside": Zone("В зоне стоимости", COLOR_UP, "⚖️", "green"),
    "above": Zone("Выше зоны стоимости", "#60a5fa", "⬆️", "yellow"),
    "below": Zone("Ниже зоны стоимости", "#f59e0b", "⬇️", "yellow"),
}


@dataclass
class Profile:
    """Профиль объема одного окна + позиция текущей цены относительно него."""
    label: str
    role: str
    interval: str
    centers: np.ndarray   # центры ценовых корзин
    volumes: np.ndarray   # объем в каждой корзине
    poc: float
    vah: float
    val: float
    position: str         # ключ POSITION_META
    zone: Zone
    dist_poc_pct: float   # где POC относительно текущей цены, % (+ выше / − ниже)
    vacuum: bool          # цена в корзине с аномально тонким объемом


@dataclass
class Report:
    """Итог анализа — все, что нужно UI, в одном месте."""
    per_window: dict[str, Profile]
    price_now: float
    price_change_24h: float
    resistance: tuple[float, str] | None  # ближайший уровень сверху (цена, имя)
    support: tuple[float, str] | None     # ближайший уровень снизу
    traffic_light: str
    verdict_title: str
    headline: str
    explanation_lines: list[str]


def _build_histogram(df: pd.DataFrame, bins: int = PROFILE_BINS
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Профиль объема: объем каждой свечи равномерно размазывается по корзинам,
    которые накрывает ее диапазон low..high.
    """
    lo, hi = float(df["low"].min()), float(df["high"].max())
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    volumes = np.zeros(bins)

    for low, high, vol in zip(df["low"].values, df["high"].values, df["volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, high, side="left") - 1, 0, bins - 1))
        volumes[i0:i1 + 1] += vol / (i1 - i0 + 1)
    return centers, volumes


def _value_area(volumes: np.ndarray, poc_idx: int,
                share: float = VALUE_AREA_SHARE) -> tuple[int, int]:
    """
    Value Area: от POC жадно расширяемся в сторону большего соседнего
    объема, пока не наберем долю `share` от всего профиля.
    """
    total = volumes.sum()
    lo = hi = poc_idx
    acc = volumes[poc_idx]
    while acc < share * total:
        up = volumes[hi + 1] if hi + 1 < len(volumes) else -1.0
        dn = volumes[lo - 1] if lo - 1 >= 0 else -1.0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += dn
    return lo, hi


def _analyze_window(label: str, df: pd.DataFrame, price_now: float) -> Profile:
    """Полный разбор одного окна: гистограмма, POC, VA, позиция цены."""
    centers, volumes = _build_histogram(df)
    poc_idx = int(volumes.argmax())
    lo_idx, hi_idx = _value_area(volumes, poc_idx)
    poc, val, vah = float(centers[poc_idx]), float(centers[lo_idx]), float(centers[hi_idx])

    if price_now > vah:
        position = "above"
    elif price_now < val:
        position = "below"
    else:
        position = "inside"

    # Вакуум: объем в корзине текущей цены аномально тонкий
    price_idx = int(np.clip(np.searchsorted(centers, price_now), 0, len(centers) - 1))
    vacuum = bool(volumes[price_idx] < VACUUM_RATIO * np.median(volumes))

    return Profile(
        label=label,
        role=WINDOWS[label]["role"],
        interval=WINDOWS[label]["interval"],
        centers=centers,
        volumes=volumes,
        poc=poc,
        vah=vah,
        val=val,
        position=position,
        zone=POSITION_META[position],
        dist_poc_pct=(poc / price_now - 1) * 100,
        vacuum=vacuum,
    )


def _nearest_levels(per_window: dict[str, Profile], price_now: float
                    ) -> tuple[tuple[float, str] | None, tuple[float, str] | None]:
    """Ближайшие сопротивление (сверху) и поддержка (снизу) из POC/VAH/VAL всех окон."""
    levels = []
    for label, p in per_window.items():
        levels += [(p.poc, f"POC {label}"), (p.vah, f"VAH {label}"), (p.val, f"VAL {label}")]
    above = [lv for lv in levels if lv[0] > price_now]
    below = [lv for lv in levels if lv[0] < price_now]
    resistance = min(above, key=lambda lv: lv[0]) if above else None
    support = max(below, key=lambda lv: lv[0]) if below else None
    return resistance, support


def compute_report(frames: dict[str, pd.DataFrame]) -> Report:
    """Сводит профили всех окон в общий вердикт со светофором."""
    df7 = frames["7д"]
    price_now = float(df7["close"].iloc[-1])
    lookback = min(96, len(df7) - 1)  # 96 свечей по 15 минут ~ сутки
    price_change_24h = (price_now / float(df7["close"].iloc[-1 - lookback]) - 1) * 100

    per_window = {label: _analyze_window(label, frames[label], price_now)
                  for label in WINDOWS}
    resistance, support = _nearest_levels(per_window, price_now)

    # Светофор — по основному окну (30д): баланс / выход из VA / вакуум
    main = per_window[MAIN_WINDOW]
    near_edge = (abs(price_now / main.vah - 1) <= EDGE_TOLERANCE
                 or abs(price_now / main.val - 1) <= EDGE_TOLERANCE)

    if main.position == "inside":
        traffic = "green"
        verdict_title = "Рынок в балансе"
        headline = (f"Цена внутри зоны стоимости 30д (${main.val:,.0f}–${main.vah:,.0f}): "
                    f"рядом много объема, движения ограничены уровнями.")
    elif main.vacuum:
        traffic = "red"
        verdict_title = "Цена в объемном вакууме"
        headline = ("Цена вне зоны стоимости 30д и в области тонкого объема — "
                    "тормозить ее нечему, вероятны быстрые движения до ближайшего узла.")
    else:
        traffic = "yellow"
        direction = "выше" if main.position == "above" else "ниже"
        verdict_title = f"Выход {direction} зоны стоимости"
        edge_note = " Цена у самой границы — решается возврат или закрепление." if near_edge else ""
        headline = (f"Цена {direction} Value Area 30д: если закрепится — это принятие "
                    f"нового диапазона, если вернется — ложный выход.{edge_note}")

    lines = []
    if resistance:
        lines.append(f"Ближайшее сопротивление: ${resistance[0]:,.0f} "
                     f"({resistance[1]}, {(resistance[0] / price_now - 1) * 100:+.1f}%).")
    if support:
        lines.append(f"Ближайшая поддержка: ${support[0]:,.0f} "
                     f"({support[1]}, {(support[0] / price_now - 1) * 100:+.1f}%).")
    for p in per_window.values():
        vacuum_note = " ⚠️ цена в тонком объеме." if p.vacuum else ""
        lines.append(
            f"{p.label} ({p.role.lower()}): {p.zone.emoji} {p.zone.label.lower()}, "
            f"POC ${p.poc:,.0f} ({p.dist_poc_pct:+.1f}% от текущей цены), "
            f"VA ${p.val:,.0f}–${p.vah:,.0f}.{vacuum_note}"
        )

    return Report(
        per_window=per_window,
        price_now=price_now,
        price_change_24h=price_change_24h,
        resistance=resistance,
        support=support,
        traffic_light=traffic,
        verdict_title=verdict_title,
        headline=headline,
        explanation_lines=lines,
    )


# ---------------------------------------------------------------------------
# UI-компоненты
# ---------------------------------------------------------------------------
def render_header(symbol: str, report: Report, source_name: str) -> None:
    """Верхняя панель: цена, вердикт по уровням, светофор."""
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


def _card_html(p: Profile) -> str:
    """HTML карточки окна профиля — в стиле карточек соседних дашбордов."""
    vacuum_note = ("<div style='color:#dc2626; font-size:0.8rem; margin-top:4px;'>"
                   "🕳️ Цена в тонком объеме</div>" if p.vacuum else "")
    return f"""
    <div style="border: 1px solid rgba(128,128,128,0.25); border-top: 5px solid {p.zone.color};
                border-radius: 10px; padding: 14px; height: 100%;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="font-size:1.2rem; font-weight:700;">{p.label}</span>
            <span style="opacity:0.7; font-size:0.8rem;">свечи {p.interval}</span>
        </div>
        <div style="opacity:0.75; font-size:0.85rem; margin-bottom:6px;">{p.role}</div>
        <div style="font-size:1.5rem; font-weight:700; color:{COLOR_POC};">
            ${p.poc:,.0f} <span style="font-size:0.85rem; opacity:0.8;">POC</span>
        </div>
        <div style="margin: 4px 0;">{p.zone.emoji} {p.zone.label}</div>
        <div style="font-size:0.8rem; opacity:0.8; margin-top:6px;">
            VA: ${p.val:,.0f} – ${p.vah:,.0f}<br>
            До POC: {p.dist_poc_pct:+.1f}%
        </div>
        {vacuum_note}
    </div>
    """


def render_window_cards(report: Report) -> None:
    """Ряд карточек 7д / 30д / 90д."""
    st.subheader("Окна профиля")
    cols = st.columns(len(report.per_window))
    for col, p in zip(cols, report.per_window.values()):
        with col:
            st.markdown(_card_html(p), unsafe_allow_html=True)


def render_main_profile(df: pd.DataFrame, p: Profile, price_now: float) -> None:
    """
    Главный график: цена во времени (слева) + горизонтальный профиль объема
    (справа) на ОБЩЕЙ ценовой оси. POC и Value Area размечены в обеих панелях.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.72, 0.28], horizontal_spacing=0.02)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["close"], name="Цена",
        mode="lines", line=dict(color="#9ca3af", width=1.4),
    ), row=1, col=1)

    # Корзины Value Area подсвечены, POC — отдельным цветом
    in_va = (p.centers >= p.val) & (p.centers <= p.vah)
    colors = np.where(in_va, "rgba(96,165,250,0.75)", "rgba(96,165,250,0.25)")
    colors[int(p.volumes.argmax())] = COLOR_POC
    fig.add_trace(go.Bar(
        x=p.volumes, y=p.centers, orientation="h", name="Профиль",
        marker_color=colors.tolist(),
    ), row=1, col=2)

    for col in (1, 2):
        fig.add_hline(y=p.poc, line_color=COLOR_POC, line_width=1.5,
                      opacity=0.9, row=1, col=col)
        fig.add_hline(y=p.vah, line_dash="dash", line_color="#60a5fa",
                      opacity=0.7, row=1, col=col)
        fig.add_hline(y=p.val, line_dash="dash", line_color="#60a5fa",
                      opacity=0.7, row=1, col=col)
        fig.add_hline(y=price_now, line_dash="dot", line_color="#e5e7eb",
                      opacity=0.9, row=1, col=col)

    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        title=f"Профиль объема · окно {p.label} (свечи {p.interval})",
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False, bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Оранжевая линия — POC ${p.poc:,.0f}; синий пунктир — границы Value Area "
               f"(${p.val:,.0f}–${p.vah:,.0f}); белый точечный — текущая цена. "
               "Яркие бары = Value Area (70% объема). Толстые «полки» тормозят цену, "
               "тонкие участки она проходит быстро.")


def render_profiles_overlay(report: Report) -> None:
    """Профили всех окон одной картинкой: формы нормированы, ось цены общая."""
    fig = go.Figure()
    for label, p in report.per_window.items():
        fig.add_trace(go.Scatter(
            x=p.volumes / p.volumes.max(), y=p.centers, name=label,
            mode="lines", line=dict(color=WINDOW_COLORS[label], width=2),
        ))
    fig.add_hline(y=report.price_now, line_dash="dot", line_color="#e5e7eb",
                  opacity=0.9, annotation_text="цена", annotation_font_size=11)

    fig.update_layout(
        title="Сравнение окон (объем нормирован)",
        height=480, margin=dict(l=10, r=10, t=40, b=40),
        xaxis_title="Объем, доля от максимума",
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Совпадение «горбов» разных окон = особенно сильный уровень: "
               "его видят и краткосрочные, и долгосрочные участники.")


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
        "Профиль объема показывает УРОВНИ, где сосредоточен интерес рынка, "
        "а не точки входа. Любая сделка требует собственного риск-менеджмента. "
        "Это не финансовый совет."
    )


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.title("🏔️ Volume Profile")
        st.caption("Уровни объема по BTC. Только аналитика — без автоторговли.")

        symbol = st.text_input("Контракт (USDT-M)", value="BTCUSDT").strip().upper()
        main_window = st.selectbox("Окно главного графика", list(WINDOWS), index=1)

        if st.button("🔄 Обновить данные", use_container_width=True):
            load_all.clear()  # сброс кэша -> следующий вызов пойдет на биржу

        st.divider()
        st.markdown(
            """
            **Как читать светофор**

            🟢 — цена в зоне стоимости (баланс)
            🟡 — цена вышла из зоны стоимости
            🔴 — цена в объемном вакууме
            """
        )
        st.caption("Не является финансовым советом.")

    frames, source_name = load_all(symbol)
    try:
        report = compute_report(frames)
    except (ValueError, KeyError, IndexError) as exc:
        st.error(f"Ошибка расчета профиля: {exc}")
        st.stop()

    render_header(symbol, report, source_name)
    st.divider()

    render_window_cards(report)
    st.divider()

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        render_main_profile(frames[main_window], report.per_window[main_window],
                            report.price_now)
    with col_right:
        render_profiles_overlay(report)

    render_explanation(report)


if __name__ == "__main__":
    main()
