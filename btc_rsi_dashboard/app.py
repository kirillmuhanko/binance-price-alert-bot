"""
app.py
======
Bitcoin Multi-Timeframe RSI Dashboard.

Аналитический дашборд (НЕ торговый бот): оценивает «степень готовности»
рынка BTC к развороту, сопоставляя RSI на четырех таймфреймах:

    15m — ранний триггер
    1h  — подтверждение импульса
    4h  — основной контекст
    1d  — глобальный фильтр направления

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from data_loader import DEFAULT_LIMIT, TIMEFRAMES, DataSourceError, load_all_timeframes
from indicators import rsi
from scoring import compute_signal
from ui_components import (
    render_explanation,
    render_header,
    render_price_chart,
    render_readiness_matrix,
    render_rsi_chart,
    render_score_gauges,
    render_timeframe_cards,
)

st.set_page_config(
    page_title="BTC Multi-TF RSI Dashboard",
    page_icon="📊",
    layout="wide",
)

# Варианты источников данных. ccxt-binance — реальные данные Binance.
SOURCE_OPTIONS = {
    "Binance (ccxt) — реальные данные": "ccxt-binance",
    "Binance REST (без ccxt) — реальные данные": "binance-rest",
    "Bybit (ccxt)": "ccxt-bybit",
    "OKX (ccxt)": "ccxt-okx",
    "Demo (синтетика, офлайн)": "demo",
}


@st.cache_data(ttl=120, show_spinner="Загружаю свечи с биржи...")
def load_data(source_key: str, symbol: str, limit: int):
    """
    Кэшированная загрузка OHLCV по всем таймфреймам.
    ttl=120 секунд: достаточно свежо для 15m-анализа и не дергает биржу
    при каждом клике по интерфейсу.
    """
    return load_all_timeframes(source_key, symbol=symbol, limit=limit)


def main() -> None:
    # ----------------------------- Sidebar ---------------------------------
    with st.sidebar:
        st.title("📊 BTC RSI Dashboard")
        st.caption("Мультитаймфреймовый анализ RSI. Только аналитика — без автоторговли.")

        source_label = st.selectbox("Источник данных", list(SOURCE_OPTIONS.keys()), index=0)
        source_key = SOURCE_OPTIONS[source_label]

        symbol = st.text_input("Торговая пара", value="BTC/USDT").strip().upper()

        chart_tf = st.selectbox("Таймфрейм графика цены", TIMEFRAMES, index=2)

        if st.button("🔄 Обновить данные", use_container_width=True):
            load_data.clear()  # сброс кэша -> следующий вызов пойдет на биржу

        st.divider()
        st.markdown(
            """
            **Как читать светофор**

            🟢 — сигнал подтвержден старшими ТФ
            🟡 — ранний сигнал, ждать подтверждения
            🔴 — старшие ТФ против / сигнала нет
            """
        )
        st.caption("Не является финансовым советом.")

    # ----------------------------- Данные ----------------------------------
    try:
        data = load_data(source_key, symbol, DEFAULT_LIMIT)
    except DataSourceError as exc:
        st.error(f"Не удалось загрузить данные: {exc}")
        st.stop()

    source_name = data[TIMEFRAMES[0]].source_name

    # ----------------------------- Расчеты ---------------------------------
    try:
        rsi_by_tf = {tf: rsi(data[tf].df["close"]) for tf in TIMEFRAMES}
        report = compute_signal(rsi_by_tf)
    except ValueError as exc:
        st.error(f"Ошибка расчета индикаторов: {exc}")
        st.stop()

    # ----------------------------- Разметка --------------------------------
    render_header(symbol, data["15m"].df, report, source_name)
    st.divider()

    render_score_gauges(report)
    st.divider()

    render_timeframe_cards(report)
    st.divider()

    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        render_price_chart(data[chart_tf].df.tail(150), symbol, chart_tf)
    with col_right:
        render_rsi_chart(
            rsi_by_tf,
            {tf: data[tf].df["timestamp"] for tf in TIMEFRAMES},
        )

    col_matrix, col_expl = st.columns([1, 1.2])
    with col_matrix:
        render_readiness_matrix(report)
    with col_expl:
        render_explanation(report)


if __name__ == "__main__":
    main()
