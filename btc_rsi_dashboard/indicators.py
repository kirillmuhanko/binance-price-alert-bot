"""
indicators.py
=============
Расчет RSI и классификация его зон.

RSI считается по классической формуле Уайлдера (Wilder, 1978):
сглаживание через EMA с alpha = 1/period — именно так RSI считают
TradingView и большинство терминалов, поэтому значения совпадут
с тем, что трейдер видит на графике.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RSI_PERIOD = 14

# Границы зон RSI
OVERSOLD = 30
NEAR_OVERSOLD = 40
NEAR_OVERBOUGHT = 60
OVERBOUGHT = 70


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI Уайлдера.

    Параметры:
        close  — серия цен закрытия (по возрастанию времени);
        period — период RSI (стандарт 14).

    Возвращает серию RSI той же длины (первые значения — NaN, пока
    не накопилась история).
    """
    if len(close) < period + 1:
        raise ValueError(
            f"Для RSI({period}) нужно минимум {period + 1} свечей, получено {len(close)}."
        )

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Сглаживание Уайлдера = EMA с alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    # Если за весь период не было ни одного падения, avg_loss = 0 -> RSI = 100
    out = out.where(avg_loss != 0, 100.0)
    return out


@dataclass
class RSIZone:
    """Описание зоны, в которой находится RSI."""
    key: str          # машинное имя зоны
    label: str        # подпись для UI
    color: str        # цвет для карточек/heatmap
    emoji: str        # быстрый визуальный маркер


# Палитра: зеленый = потенциал для long, красный = потенциал для short,
# серый = нейтрально. Это зоны RSI, а не сигнал сам по себе!
_ZONES = [
    (0, OVERSOLD, RSIZone("oversold", "Перепроданность", "#16a34a", "🟢")),
    (OVERSOLD, NEAR_OVERSOLD, RSIZone("near_oversold", "Близко к перепроданности", "#84cc16", "🟡")),
    (NEAR_OVERSOLD, NEAR_OVERBOUGHT, RSIZone("neutral", "Нейтральная зона", "#9ca3af", "⚪")),
    (NEAR_OVERBOUGHT, OVERBOUGHT, RSIZone("near_overbought", "Близко к перекупленности", "#f97316", "🟡")),
    (OVERBOUGHT, 101, RSIZone("overbought", "Перекупленность", "#dc2626", "🔴")),
]


def classify_zone(rsi_value: float) -> RSIZone:
    """Возвращает зону RSI для одиночного значения."""
    for low, high, zone in _ZONES:
        if low <= rsi_value < high:
            return zone
    return _ZONES[2][2]  # подстраховка: нейтральная зона


def rsi_slope(rsi_series: pd.Series, bars: int = 3) -> float:
    """
    Грубая оценка направления RSI: изменение за последние `bars` свечей.
    Положительное значение — RSI растет (импульс вверх), отрицательное — падает.
    Используется в scoring как признак «разворот уже начался или RSI еще падает».
    """
    clean = rsi_series.dropna()
    if len(clean) < bars + 1:
        return 0.0
    return float(clean.iloc[-1] - clean.iloc[-1 - bars])
