"""
ui_components.py
================
Все визуальные блоки дашборда. app.py только собирает их вместе.

Компоненты:
    render_header        — цена BTC + светофор + итоговый статус;
    render_score_gauges  — четыре метрики скоринга;
    render_timeframe_cards — карточки 15m / 1h / 4h / 1d;
    render_price_chart   — свечной график цены (plotly);
    render_rsi_chart     — RSI всех таймфреймов на одном графике;
    render_readiness_matrix — heatmap «таймфрейм x метрика»;
    render_explanation   — разбор сигнала человеческим языком.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from indicators import NEAR_OVERBOUGHT, NEAR_OVERSOLD, OVERBOUGHT, OVERSOLD
from scoring import SignalReport, TimeframeScore

# Цвета светофора
TRAFFIC_COLORS = {
    "green": ("#16a34a", "🟢", "Сигнал подтвержден"),
    "yellow": ("#d97706", "🟡", "Ранний сигнал — ждать"),
    "red": ("#dc2626", "🔴", "Не входить"),
}

SIGNAL_LABELS = {
    "LONG_SETUP": "LONG SETUP",
    "SHORT_SETUP": "SHORT SETUP",
    "NO_TRADE": "NO TRADE ZONE",
}

# Цвета линий RSI по таймфреймам
TF_COLORS = {"15m": "#60a5fa", "1h": "#a78bfa", "4h": "#f59e0b", "1d": "#ef4444"}


# ---------------------------------------------------------------------------
# Шапка
# ---------------------------------------------------------------------------
def render_header(symbol: str, price_df: pd.DataFrame, report: SignalReport,
                  source_name: str) -> None:
    """Верхняя панель: цена, изменение за 24ч, светофор, итоговый сигнал."""
    last_price = float(price_df["close"].iloc[-1])

    # Изменение за ~24ч: 96 свечей по 15 минут
    lookback = min(96, len(price_df) - 1)
    prev_price = float(price_df["close"].iloc[-1 - lookback])
    change_pct = (last_price / prev_price - 1) * 100

    color, emoji, traffic_text = TRAFFIC_COLORS[report.traffic_light]

    col_price, col_signal, col_light = st.columns([1.2, 1.4, 1])
    with col_price:
        st.metric(
            label=f"{symbol} · источник: {source_name}",
            value=f"${last_price:,.0f}",
            delta=f"{change_pct:+.2f}% за 24ч",
        )
    with col_signal:
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding: 8px 14px;
                        background: rgba(128,128,128,0.08); border-radius: 6px;">
                <div style="font-size: 1.4rem; font-weight: 700;">
                    {SIGNAL_LABELS[report.signal]}
                </div>
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


# ---------------------------------------------------------------------------
# Метрики скоринга
# ---------------------------------------------------------------------------
def render_score_gauges(report: SignalReport) -> None:
    """Четыре ключевые метрики модели в одну строку."""
    cols = st.columns(4)
    metrics = [
        ("Long Readiness", report.long_score,
         "Готовность рынка к лонгу: 0 — никакой, 100 — максимальная."),
        ("Short Readiness", report.short_score,
         "Готовность рынка к шорту."),
        ("MTF Alignment", report.alignment,
         "Согласованность таймфреймов: 100 — все смотрят в одну сторону."),
        ("Signal Confidence", report.confidence,
         "Итоговая уверенность: сочетание готовности и согласованности."),
    ]
    for col, (name, value, help_text) in zip(cols, metrics):
        with col:
            st.metric(label=name, value=f"{value:.0f} / 100", help=help_text)
            st.progress(min(1.0, value / 100))


# ---------------------------------------------------------------------------
# Карточки таймфреймов
# ---------------------------------------------------------------------------
def _card_html(s: TimeframeScore) -> str:
    """HTML одной карточки таймфрейма."""
    slope_arrow = "↑" if s.slope > 1.5 else ("↓" if s.slope < -1.5 else "→")
    weight_pct = f"{s.weight * 100:.0f}%"
    return f"""
    <div style="border: 1px solid rgba(128,128,128,0.25); border-top: 5px solid {s.zone.color};
                border-radius: 10px; padding: 14px; height: 100%;">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="font-size:1.2rem; font-weight:700;">{s.timeframe}</span>
            <span style="opacity:0.7; font-size:0.8rem;">вес {weight_pct}</span>
        </div>
        <div style="opacity:0.75; font-size:0.85rem; margin-bottom:6px;">{s.role}</div>
        <div style="font-size:2rem; font-weight:700; color:{s.zone.color};">
            {s.rsi:.1f} <span style="font-size:1.1rem;">{slope_arrow}</span>
        </div>
        <div style="margin: 4px 0;">{s.zone.emoji} {s.zone.label}</div>
        <div style="font-size:0.8rem; opacity:0.8; margin-top:6px;">
            Вклад: long {s.long_contribution:.0f} / short {s.short_contribution:.0f}
        </div>
        <div style="font-size:0.85rem; margin-top:8px; opacity:0.9;">{s.explanation}</div>
    </div>
    """


def render_timeframe_cards(report: SignalReport) -> None:
    """Ряд карточек 15m / 1h / 4h / 1d."""
    st.subheader("Таймфреймы")
    cols = st.columns(len(report.per_timeframe))
    for col, s in zip(cols, report.per_timeframe.values()):
        with col:
            st.markdown(_card_html(s), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def render_price_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Свечной график цены с объемом."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=symbol,
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        ),
        row=1, col=1,
    )
    volume_colors = ["#16a34a" if c >= o else "#dc2626"
                     for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], name="Volume",
               marker_color=volume_colors, opacity=0.5),
        row=2, col=1,
    )
    fig.update_layout(
        title=f"{symbol} · {timeframe}",
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rsi_chart(rsi_by_tf: dict[str, pd.Series],
                     timestamps_by_tf: dict[str, pd.Series],
                     bars: int = 120) -> None:
    """
    RSI по таймфреймам — сетка 2x2, каждому ТФ своя панель.

    Рисовать все ТФ на одной оси времени нельзя: 300 дневных свечей
    покрывают месяцы, а 120 свечей 15m — чуть больше суток, и младшие
    линии сжимаются в нечитаемую полосу у правого края. Поэтому у каждой
    панели свое временное окно, а сравниваются УРОВНИ RSI (ось Y общая, 0–100).
    """
    tfs = list(rsi_by_tf.keys())
    titles = [f"<b>{tf}</b> · RSI {rsi_by_tf[tf].dropna().iloc[-1]:.1f}" for tf in tfs]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.22, horizontal_spacing=0.07,
    )

    for i, tf in enumerate(tfs):
        row, col = i // 2 + 1, i % 2 + 1
        clean = rsi_by_tf[tf].dropna()
        ts = timestamps_by_tf[tf].iloc[-len(clean):]

        fig.add_trace(go.Scatter(
            x=ts.iloc[-bars:], y=clean.iloc[-bars:],
            mode="lines", name=f"RSI {tf}",
            line=dict(color=TF_COLORS.get(tf, "#888"), width=2),
            showlegend=False,
        ), row=row, col=col)

        # Фоновые зоны и уровни — в каждой панели.
        # ВАЖНО: add_hrect/add_hline с row/col добавлять ПОСЛЕ трейса —
        # plotly молча игнорирует их для сабплотов без данных.
        fig.add_hrect(y0=0, y1=OVERSOLD, fillcolor="#16a34a", opacity=0.18,
                      line_width=0, row=row, col=col)
        fig.add_hrect(y0=OVERBOUGHT, y1=100, fillcolor="#dc2626", opacity=0.18,
                      line_width=0, row=row, col=col)
        for level, dash in [(OVERSOLD, "dash"), (NEAR_OVERSOLD, "dot"),
                            (NEAR_OVERBOUGHT, "dot"), (OVERBOUGHT, "dash")]:
            fig.add_hline(y=level, line_dash=dash, line_color="#9ca3af",
                          opacity=0.6, row=row, col=col)

        # Маркер последнего значения, чтобы текущий уровень считывался мгновенно
        fig.add_trace(go.Scatter(
            x=[ts.iloc[-1].isoformat()], y=[float(clean.iloc[-1])],
            mode="markers", marker=dict(color=TF_COLORS.get(tf, "#888"), size=8),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        fig.layout.annotations[i].update(
            font=dict(size=13, color=TF_COLORS.get(tf, "#888")),
        )
        fig.update_yaxes(range=[0, 100], row=row, col=col)

    fig.update_layout(
        title=dict(text="RSI(14) по таймфреймам", y=0.98),
        height=460, margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Heatmap готовности
# ---------------------------------------------------------------------------
def render_readiness_matrix(report: SignalReport) -> None:
    """Матрица «таймфрейм x метрика»: RSI, вклад в long, вклад в short."""
    st.subheader("Матрица готовности")

    tfs = list(report.per_timeframe.keys())
    rows = ["RSI", "Long вклад", "Short вклад"]
    z = [
        [report.per_timeframe[tf].rsi for tf in tfs],
        [report.per_timeframe[tf].long_contribution for tf in tfs],
        [report.per_timeframe[tf].short_contribution for tf in tfs],
    ]
    text = [[f"{v:.0f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=tfs, y=rows, text=text,
        texttemplate="%{text}", textfont=dict(size=16),
        colorscale="RdYlGn", zmin=0, zmax=100,
        showscale=False,
    ))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Строка RSI: зеленый = высокий RSI, красный = низкий. "
        "Строки вкладов: зеленый = высокий вклад в соответствующее направление. "
        "Идеальный лонг-сетап — зеленая строка «Long вклад» на всех таймфреймах."
    )


# ---------------------------------------------------------------------------
# Объяснение сигнала
# ---------------------------------------------------------------------------
def render_explanation(report: SignalReport) -> None:
    """Блок «что происходит человеческим языком»."""
    st.subheader("Объяснение сигнала")
    color, _, _ = TRAFFIC_COLORS[report.traffic_light]
    st.markdown(
        f"<div style='border-left: 6px solid {color}; padding-left: 12px; "
        f"font-weight: 600;'>{report.headline}</div>",
        unsafe_allow_html=True,
    )
    for line in report.explanation_lines:
        st.markdown(f"- {line}")
    st.caption(
        "Дашборд показывает СТЕПЕНЬ ГОТОВНОСТИ рынка, а не точку входа. "
        "Любая сделка требует собственного риск-менеджмента. Это не финансовый совет."
    )
