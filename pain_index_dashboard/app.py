"""
app.py
======
BTC Position Pain Index Dashboard (Binance USDT-M Futures).

Аналитический дашборд (НЕ торговый бот). Авторский индикатор:
по истории Open Interest восстанавливается СРЕДНЯЯ ЦЕНА ВХОДА всех
открытых фьючерсных позиций — аналог on-chain Realized Price, но для перпов.

    OI вырос  -> открылись новые позиции по текущей цене (новый «транш»);
    OI упал   -> позиции закрылись пропорционально (распределение входов
                 сохраняет форму, средний вход не меняется).

Из реконструкции считаются:
    Рыночный безубыток — цена, при которой средняя позиция в нуле;
    Индекс боли        — насколько глубоко открытые позиции под водой;
    Детектор капитуляции — боль высокая И OI резко падает: больные позиции
                 выбрасывают, исторически рядом формируются экстремумы.

Допущение (честно): Binance отдает историю OI ~30 дней, поэтому позиции
старше окна заякорены на цене его начала; закрытия считаются
пропорциональными. Для BTC-перпа, где OI оборачивается за дни, это
рабочая точность.

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

st.set_page_config(
    page_title="BTC Pain Index Dashboard",
    page_icon="🩸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
FAPI = "https://fapi.binance.com"

# Периоды реконструкции (= интервал истории OI и свечей), окно ~500 баров
PERIODS = {"30m": 30, "1h": 60, "2h": 120, "4h": 240}
DEFAULT_PERIOD = "1h"

# Пороги взвешенной боли позиций, % от цены входа
PAIN_NOTABLE = 3.0    # боль заметна — топливо копится
PAIN_HIGH = 7.0       # боль высокая
EPISODE_PAIN = 5.0    # порог боли для эпизода капитуляции
FLUSH_OI_DROP = -5.0  # падение OI за 24ч, % — «флаш» позиций

PROFILE_BINS = 60     # корзин в профиле входов позиций

COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_BREAKEVEN = "#f59e0b"

# Цвета светофора — те же, что в соседних дашбордах
TRAFFIC_COLORS = {
    "green": ("#16a34a", "🟢", "Рынок у безубытка"),
    "yellow": ("#d97706", "🟡", "Боль накапливается"),
    "red": ("#dc2626", "🔴", "Капитуляция в процессе"),
}


class DataSourceError(Exception):
    """Единый тип ошибки загрузки данных — UI ловит именно его."""


# ---------------------------------------------------------------------------
# Загрузка данных: Binance Futures REST + офлайн demo-fallback
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Сырье одного обновления: OI + цена на общей сетке, текущий funding."""
    df: pd.DataFrame      # timestamp, close, oi
    funding_now: float
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
    """История OI + свечи того же интервала, сведенные на одну сетку времени."""
    sym = symbol.replace("/", "").upper()

    raw_oi = _get_json("/futures/data/openInterestHist",
                       {"symbol": sym, "period": period, "limit": 500})
    oi_df = pd.DataFrame(raw_oi)
    oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
    oi_df["oi"] = pd.to_numeric(oi_df["sumOpenInterest"], errors="coerce")
    oi_df = oi_df[["timestamp", "oi"]].dropna().sort_values("timestamp")

    raw_klines = _get_json("/fapi/v1/klines",
                           {"symbol": sym, "interval": period, "limit": 500})
    price_df = pd.DataFrame(
        [(row[0], row[4]) for row in raw_klines], columns=["timestamp", "close"]
    )
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True)
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df = price_df.dropna().sort_values("timestamp")

    # Метки OI стоят на закрытии периода, klines — на открытии: сводим nearest
    df = pd.merge_asof(
        oi_df, price_df, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta(minutes=PERIODS[period]),
    ).dropna().reset_index(drop=True)
    if len(df) < 50:
        raise DataSourceError("Слишком мало совмещенных точек OI и цены.")

    premium = _get_json("/fapi/v1/premiumIndex", {"symbol": sym})
    return Snapshot(
        df=df,
        funding_now=float(premium["lastFundingRate"]),
        source_name="binance-futures",
    )


def _demo_snapshot(period: str) -> Snapshot:
    """
    Синтетические данные, если биржа недоступна (сеть, гео-блокировка).
    OI падает на сильных движениях цены — детектору капитуляций есть что ловить.
    """
    rng = np.random.default_rng(seed=int(time.time() // 3600))
    minutes = PERIODS[period]
    n = 500
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")

    returns = rng.normal(0, 0.004 * np.sqrt(minutes / 60), n)
    close = 100_000 * np.exp(np.cumsum(returns))
    # OI: дрейф вверх в спокойные периоды, резкие сбросы на сильных движениях
    oi_ret = 0.0015 - 1.5 * np.abs(returns) + rng.normal(0, 0.004, n)
    oi = 80_000 * np.exp(np.cumsum(oi_ret))

    df = pd.DataFrame({"timestamp": idx, "close": close, "oi": oi})
    return Snapshot(df=df, funding_now=0.00018, source_name="demo")


@st.cache_data(ttl=300, show_spinner="Восстанавливаю входы позиций из OI...")
def load_snapshot(symbol: str, period: str) -> Snapshot:
    """
    Кэшированная загрузка (ttl=300с — реконструкция меняется медленно).
    Если Binance недоступен — офлайн demo-данные, дашборд не «умирает».
    """
    try:
        return _fetch_binance(symbol, period)
    except DataSourceError:
        return _demo_snapshot(period)


# ---------------------------------------------------------------------------
# Реконструкция позиций и анализ
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """Качественная оценка: подпись + цвет + эмодзи + уровень тревоги."""
    label: str
    color: str
    emoji: str
    level: str  # green / yellow / red


@dataclass
class Report:
    """Итог анализа — все, что нужно UI, в одном месте."""
    df: pd.DataFrame                # + колонки breakeven, long_pain, short_pain, capit
    tranche_prices: np.ndarray      # входы открытых траншей на конец окна
    tranche_sizes: np.ndarray
    price_now: float
    price_change_24h: float
    breakeven: float
    price_vs_breakeven_pct: float
    long_pain: float                # взвешенная боль лонгов, %
    short_pain: float
    long_under_share: float         # доля OI, где лонги в минусе, %
    crowd_side: str                 # «лонги» / «шорты» — кто переполнен (по funding)
    crowd_pain: float
    oi_change_24h: float
    episodes: list[tuple[pd.Timestamp, pd.Timestamp, str]]  # капитуляции (от, до, чьи)
    pain_zone: Zone
    capit_zone: Zone
    traffic_light: str
    verdict_title: str
    headline: str
    explanation_lines: list[str]


def _reconstruct(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Прогон траншей: каждый прирост OI — новый транш по текущей цене,
    каждое снижение OI пропорционально ужимает ВСЕ транши (форма
    распределения входов сохраняется). Возвращает df с колонками
    breakeven / long_pain / short_pain / long_under_share и финальные транши.
    """
    oi = df["oi"].to_numpy()
    price = df["close"].to_numpy()
    n = len(df)

    # Стартовый OI заякорен на первой цене окна (главное допущение модели)
    tr_price = [price[0]]
    tr_size = [oi[0]]

    breakeven = np.zeros(n)
    long_pain = np.zeros(n)
    short_pain = np.zeros(n)
    under_share = np.zeros(n)

    for i in range(n):
        if i > 0:
            delta = oi[i] - oi[i - 1]
            if delta > 0:
                tr_price.append(price[i])
                tr_size.append(delta)
            elif delta < 0 and oi[i - 1] > 0:
                scale = oi[i] / oi[i - 1]
                tr_size = [s * scale for s in tr_size]

        p = np.asarray(tr_price)
        s = np.asarray(tr_size)
        total = s.sum()

        breakeven[i] = float((p * s).sum() / total)
        # Боль транша = его просадка в % от входа (для лонга и шорта зеркально)
        long_pain[i] = float((np.clip((p - price[i]) / p, 0, None) * s).sum() / total * 100)
        short_pain[i] = float((np.clip((price[i] - p) / p, 0, None) * s).sum() / total * 100)
        under_share[i] = float(s[p > price[i]].sum() / total * 100)

    out = df.copy()
    out["breakeven"] = breakeven
    out["long_pain"] = long_pain
    out["short_pain"] = short_pain
    out["long_under_share"] = under_share
    return out, np.asarray(tr_price), np.asarray(tr_size)


def _find_episodes(df: pd.DataFrame, bars_24h: int
                   ) -> tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    """
    Эпизод капитуляции: боль выше порога И OI за 24ч упал сильнее порога —
    больные позиции выбрасывают. Возвращает флаг по барам и список эпизодов.
    """
    oi_chg = (df["oi"] / df["oi"].shift(bars_24h) - 1) * 100
    pain = df[["long_pain", "short_pain"]].max(axis=1)
    flag = (pain > EPISODE_PAIN) & (oi_chg < FLUSH_OI_DROP)

    episodes = []
    start = None
    for i in range(len(df)):
        if flag.iloc[i] and start is None:
            start = i
        elif not flag.iloc[i] and start is not None:
            side = ("лонгов" if df["long_pain"].iloc[start:i].mean()
                    >= df["short_pain"].iloc[start:i].mean() else "шортов")
            episodes.append((df["timestamp"].iloc[start], df["timestamp"].iloc[i - 1], side))
            start = None
    if start is not None:
        side = ("лонгов" if df["long_pain"].iloc[start:].mean()
                >= df["short_pain"].iloc[start:].mean() else "шортов")
        episodes.append((df["timestamp"].iloc[start], df["timestamp"].iloc[-1], side))
    return flag, episodes


def compute_report(snap: Snapshot, period: str) -> Report:
    """Сводит реконструкцию в общий вердикт со светофором."""
    df, tr_price, tr_size = _reconstruct(snap.df)
    bars_24h = max(1, 1440 // PERIODS[period])

    price_now = float(df["close"].iloc[-1])
    lookback = min(bars_24h, len(df) - 1)
    price_change_24h = (price_now / float(df["close"].iloc[-1 - lookback]) - 1) * 100
    oi_change_24h = (float(df["oi"].iloc[-1]) / float(df["oi"].iloc[-1 - lookback]) - 1) * 100

    breakeven = float(df["breakeven"].iloc[-1])
    long_pain = float(df["long_pain"].iloc[-1])
    short_pain = float(df["short_pain"].iloc[-1])
    long_under_share = float(df["long_under_share"].iloc[-1])

    crowd_side = "лонги" if snap.funding_now > 0 else "шорты"
    crowd_pain = long_pain if crowd_side == "лонги" else short_pain

    flag, episodes = _find_episodes(df, bars_24h)
    df["capit"] = flag
    capit_now = bool(flag.iloc[-1])

    # Зона боли (по перекошенной стороне)
    if crowd_pain >= PAIN_HIGH:
        pain_zone = Zone("Боль высокая", COLOR_DOWN, "🔥", "red")
    elif crowd_pain >= PAIN_NOTABLE:
        pain_zone = Zone("Боль заметная", "#d97706", "⚠️", "yellow")
    else:
        pain_zone = Zone("Боли почти нет", COLOR_UP, "✅", "green")

    # Зона капитуляции
    if capit_now:
        capit_zone = Zone("Идет прямо сейчас", COLOR_DOWN, "🩸", "red")
    elif episodes and (df["timestamp"].iloc[-1] - episodes[-1][1]) < pd.Timedelta(days=3):
        capit_zone = Zone("Была на днях", "#d97706", "🕑", "yellow")
    else:
        capit_zone = Zone("Нет", COLOR_UP, "✅", "green")

    # Светофор и вердикт
    if capit_now:
        side = episodes[-1][2] if episodes else "позиций"
        traffic = "red"
        verdict_title = f"Капитуляция {side}"
        headline = ("Боль высокая и OI резко падает — больные позиции выбрасывают. "
                    "Исторически рядом с такими флашами формируются локальные экстремумы.")
    elif crowd_pain >= PAIN_NOTABLE:
        fuel = "ликвидаций лонгов" if crowd_side == "лонги" else "шорт-сквиза"
        traffic = "yellow"
        verdict_title = f"Копится топливо для {fuel}"
        headline = (f"Средняя позиция под водой, а {crowd_side} продолжают платить funding — "
                    f"боль терпят. Чем дольше терпят, тем резче развязка.")
    else:
        traffic = "green"
        verdict_title = "Рынок у безубытка"
        headline = ("Цена рядом со средним входом открытых позиций — "
                    "массового навеса прибыли или боли нет.")

    window_days = 500 * PERIODS[period] / 1440
    lines = [
        f"Рыночный безубыток: ${breakeven:,.0f} — цена сейчас "
        f"{(price_now / breakeven - 1) * 100:+.1f}% от него. Этот уровень — "
        f"средний вход всех открытых позиций и часто работает как магнит.",
        f"Лонги: {long_under_share:.0f}% OI в минусе, взвешенная боль {long_pain:.1f}%. "
        f"Шорты: боль {short_pain:.1f}%.",
        f"Перекошенная сторона (по funding {snap.funding_now * 100:+.4f}%): {crowd_side} — "
        f"их боль {crowd_pain:.1f}% и есть топливо для принудительных закрытий.",
        f"OI за 24ч: {oi_change_24h:+.2f}%. Капитуляция = боль > {EPISODE_PAIN:.0f}% "
        f"при падении OI ниже {FLUSH_OI_DROP:.0f}% за сутки.",
        f"Эпизодов капитуляции за окно (~{window_days:.0f} дней): {len(episodes)}."
        + (f" Последний: {episodes[-1][1]:%d.%m %H:%M} UTC, капитулировали {episodes[-1][2]}."
           if episodes else ""),
        "Допущение модели: позиции старше окна заякорены на цене его начала, "
        "закрытия пропорциональны — уровни приблизительные, важна динамика.",
    ]

    return Report(
        df=df,
        tranche_prices=tr_price,
        tranche_sizes=tr_size,
        price_now=price_now,
        price_change_24h=price_change_24h,
        breakeven=breakeven,
        price_vs_breakeven_pct=(price_now / breakeven - 1) * 100,
        long_pain=long_pain,
        short_pain=short_pain,
        long_under_share=long_under_share,
        crowd_side=crowd_side,
        crowd_pain=crowd_pain,
        oi_change_24h=oi_change_24h,
        episodes=episodes,
        pain_zone=pain_zone,
        capit_zone=capit_zone,
        traffic_light=traffic,
        verdict_title=verdict_title,
        headline=headline,
        explanation_lines=lines,
    )


# ---------------------------------------------------------------------------
# UI-компоненты
# ---------------------------------------------------------------------------
def render_header(symbol: str, report: Report, source_name: str) -> None:
    """Верхняя панель: цена, вердикт по боли позиций, светофор."""
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


def _card_html(title: str, value: str, zone: Zone, sub: str, note: str) -> str:
    """HTML карточки — в стиле карточек соседних дашбордов."""
    return f"""
    <div style="border: 1px solid rgba(128,128,128,0.25); border-top: 5px solid {zone.color};
                border-radius: 10px; padding: 14px; height: 100%;">
        <div style="font-size:1.05rem; font-weight:700;">{title}</div>
        <div style="font-size:1.7rem; font-weight:700; color:{zone.color}; margin: 4px 0;">
            {value}
        </div>
        <div style="margin: 4px 0;">{zone.emoji} {zone.label}</div>
        <div style="font-size:0.85rem; opacity:0.8;">{sub}</div>
        <div style="font-size:0.8rem; opacity:0.7; margin-top:6px;">{note}</div>
    </div>
    """


def render_cards(report: Report) -> None:
    """Три карточки: безубыток, боль перекошенной стороны, капитуляция."""
    be_zone = Zone(
        "Цена выше безубытка" if report.price_vs_breakeven_pct >= 0 else "Цена ниже безубытка",
        COLOR_UP if report.price_vs_breakeven_pct >= 0 else COLOR_DOWN,
        "📍", "green",
    )
    col_be, col_pain, col_cap = st.columns(3)
    with col_be:
        st.markdown(_card_html(
            "Рыночный безубыток",
            f"${report.breakeven:,.0f}",
            be_zone,
            f"Цена {report.price_vs_breakeven_pct:+.1f}% от среднего входа позиций",
            "Аналог Realized Price для перпов: уровень-магнит, где средняя позиция в нуле.",
        ), unsafe_allow_html=True)
    with col_pain:
        st.markdown(_card_html(
            f"Боль толпы ({report.crowd_side})",
            f"{report.crowd_pain:.1f}%",
            report.pain_zone,
            f"Лонги: {report.long_under_share:.0f}% OI в минусе (боль {report.long_pain:.1f}%) "
            f"· шорты: боль {report.short_pain:.1f}%",
            "Взвешенная просадка открытых позиций от их входа. Чья сторона перекошена — "
            "определяется по знаку funding.",
        ), unsafe_allow_html=True)
    with col_cap:
        st.markdown(_card_html(
            "Капитуляция",
            f"{len(report.episodes)} эп.",
            report.capit_zone,
            f"OI за 24ч: {report.oi_change_24h:+.2f}%",
            f"Эпизод = боль > {EPISODE_PAIN:.0f}% при падении OI сильнее "
            f"{FLUSH_OI_DROP:.0f}% за сутки: больных выбрасывает из позиций.",
        ), unsafe_allow_html=True)


def render_breakeven_chart(report: Report) -> None:
    """Цена и линия безубытка; эпизоды капитуляции подсвечены."""
    df = report.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["close"], name="Цена",
        mode="lines", line=dict(color="#9ca3af", width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["breakeven"], name="Безубыток рынка",
        mode="lines", line=dict(color=COLOR_BREAKEVEN, width=2),
    ))
    for start, end, side in report.episodes:
        fig.add_vrect(x0=start, x1=end, fillcolor=COLOR_DOWN, opacity=0.15, line_width=0)

    fig.update_layout(
        title="Цена и рыночный безубыток · красные зоны — капитуляции",
        height=420, margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Цена ниже оранжевой линии — средняя открытая позиция (лонг) в минусе. "
               "Чем дальше цена от безубытка и чем дольше, тем больше накопленная боль. "
               "Красные зоны: боль + резкий сброс OI = принудительный выход.")


def render_position_profile(report: Report) -> None:
    """
    Профиль входов открытых позиций: где сидят входы и кто под водой.

    Первый транш — «якорь окна»: весь OI, существовавший на старте окна,
    записан одной строкой по тогдашней цене. Реальные входы этих старых
    позиций неизвестны, поэтому он рисуется серым отдельно от профиля.
    """
    anchor_price = float(report.tranche_prices[0])
    anchor_size = float(report.tranche_sizes[0])
    p, s = report.tranche_prices[1:], report.tranche_sizes[1:]

    lo, hi = float(p.min()), float(p.max())
    edges = np.linspace(lo, hi * 1.0001, PROFILE_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sizes, _ = np.histogram(p, bins=edges, weights=s)

    # Входы выше текущей цены — лонги этих траншей в минусе
    colors = ["rgba(220,38,38,0.75)" if c > report.price_now else "rgba(22,163,74,0.75)"
              for c in centers]

    fig = go.Figure(go.Bar(x=sizes, y=centers, orientation="h", marker_color=colors))
    fig.add_trace(go.Bar(x=[anchor_size], y=[anchor_price], orientation="h",
                         marker_color="rgba(156,163,175,0.45)"))
    fig.add_annotation(x=anchor_size, y=anchor_price, text="якорь окна (входы старше окна)",
                       showarrow=False, xanchor="right", yshift=12,
                       font=dict(size=11, color="#9ca3af"))
    fig.add_hline(y=report.price_now, line_dash="dot", line_color="#e5e7eb",
                  opacity=0.9, annotation_text="цена", annotation_font_size=11)
    fig.add_hline(y=report.breakeven, line_color=COLOR_BREAKEVEN, line_width=1.5,
                  opacity=0.9, annotation_text="безубыток", annotation_font_size=11)

    fig.update_layout(
        title="Профиль входов открытых позиций",
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Открытый интерес транша, BTC",
        showlegend=False, bargap=0.05, barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Красные транши — входы выше текущей цены (лонги там в минусе, "
               "шорты в плюсе), зеленые — ниже. Толстая красная «шапка» над ценой = "
               "навес больных лонгов, которые будут продавать на отскоках. "
               "Серый бар — позиции старше окна: их реальные входы неизвестны, "
               "они заякорены на стартовой цене окна.")


def render_pain_history(report: Report) -> None:
    """История боли лонгов и шортов с порогами."""
    df = report.df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["long_pain"], name="Боль лонгов",
        mode="lines", line=dict(color=COLOR_DOWN, width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["short_pain"], name="Боль шортов",
        mode="lines", line=dict(color=COLOR_UP, width=1.8),
    ))
    fig.add_hline(y=PAIN_NOTABLE, line_dash="dot", line_color="#d97706", opacity=0.7)
    fig.add_hline(y=PAIN_HIGH, line_dash="dash", line_color="#dc2626", opacity=0.7)

    fig.update_layout(
        title="Индекс боли по сторонам, %",
        height=300, margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", y=-0.3, x=0),
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Пунктир {PAIN_NOTABLE:.0f}% — боль заметна, {PAIN_HIGH:.0f}% — высокая. "
               "Пик боли, который начинает спадать вместе с OI, — классическая картина "
               "капитуляции и зона поиска разворота.")


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
        "Индекс боли — оценочная модель ПОЗИЦИОННОГО НАВЕСА, а не точка входа. "
        "Любая сделка требует собственного риск-менеджмента. Это не финансовый совет."
    )


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.title("🩸 Pain Index")
        st.caption("Средний вход и боль открытых позиций BTC. "
                   "Только аналитика — без автоторговли.")

        symbol = st.text_input("Контракт (USDT-M)", value="BTCUSDT").strip().upper()
        period = st.selectbox("Период реконструкции", list(PERIODS),
                              index=list(PERIODS).index(DEFAULT_PERIOD))
        window_days = 500 * PERIODS[period] / 1440
        st.caption(f"Окно: ~{window_days:.0f} дней (500 точек OI)")

        if st.button("🔄 Обновить данные", use_container_width=True):
            load_snapshot.clear()  # сброс кэша -> следующий вызов пойдет на биржу

        st.divider()
        st.markdown(
            """
            **Как читать светофор**

            🟢 — рынок у безубытка, боли нет
            🟡 — боль копится: топливо для ликвидаций
            🔴 — капитуляция: больных выбрасывает
            """
        )
        st.caption("Не является финансовым советом.")

    snap = load_snapshot(symbol, period)
    try:
        report = compute_report(snap, period)
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        st.error(f"Ошибка расчета индекса: {exc}")
        st.stop()

    render_header(symbol, report, snap.source_name)
    st.divider()

    render_cards(report)
    st.divider()

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        render_breakeven_chart(report)
    with col_right:
        render_position_profile(report)

    render_pain_history(report)
    render_explanation(report)


if __name__ == "__main__":
    main()
