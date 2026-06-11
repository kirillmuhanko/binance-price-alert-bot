"""
scoring.py
==========
Скоринговая модель мультитаймфреймового RSI.

Идея
----
RSI на одном таймфрейме часто врет: на 15m он может показывать
«перепроданность», когда на 1h/4h/1d падение только разгоняется.
Поэтому каждый таймфрейм получает свою РОЛЬ и свой ВЕС:

    15m — ранний триггер        (вес 0.15)  «звонок будильника»
    1h  — подтверждение импульса (вес 0.20)  «проснулся ли рынок»
    4h  — основной контекст      (вес 0.35)  «куда вообще идет рынок»
    1d  — глобальный фильтр      (вес 0.30)  «можно ли в эту сторону в принципе»

Метрики
-------
1. Long Readiness Score (0–100)  — насколько рынок созрел для лонга.
2. Short Readiness Score (0–100) — то же для шорта.
3. Multi-Timeframe Alignment (0–100) — насколько таймфреймы согласны
   между собой по направлению.
4. Signal Confidence (0–100) — итоговая уверенность: комбинация
   readiness и alignment с штрафами (например, дневной RSI против сделки).

Светофор
--------
🟢 green  — сигнал подтвержден старшими ТФ, сетап можно рассматривать;
🟡 yellow — младшие ТФ дали ранний сигнал, старшие еще не подтвердили: ЖДАТЬ;
🔴 red    — старшие таймфреймы против; вход = ловля падающего ножа.

Важно: это аналитика, не торговая рекомендация. Модель ничего не покупает
и не продает.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import RSIZone, classify_zone, rsi_slope

# ---------------------------------------------------------------------------
# Конфигурация ролей таймфреймов
# ---------------------------------------------------------------------------
TIMEFRAME_CONFIG: dict[str, dict] = {
    "15m": {"role": "Ранний триггер", "weight": 0.15,
            "hint": "Первым замечает локальный экстремум, но чаще всех врет."},
    "1h": {"role": "Подтверждение импульса", "weight": 0.20,
           "hint": "Фильтрует шум 15m: импульс реальный или случайный тик."},
    "4h": {"role": "Контекст рынка", "weight": 0.35,
           "hint": "Главный таймфрейм: задает рабочее направление на днях."},
    "1d": {"role": "Глобальный фильтр", "weight": 0.30,
           "hint": "Запрещает сделки против большого тренда."},
}

# Пороги дневного RSI для «вето» старшего таймфрейма:
#  - дневной RSI ниже DAILY_DOWNTREND_RSI и падает -> сильный нисходящий поток,
#    лонги от перепроданности младших ТФ опасны (вето на лонг);
#  - дневной RSI выше DAILY_UPTREND_RSI и растет -> сильный восходящий поток,
#    шорты от перекупленности младших ТФ опасны (вето на шорт).
DAILY_UPTREND_RSI = 62
DAILY_DOWNTREND_RSI = 38


# ---------------------------------------------------------------------------
# Кривые готовности: RSI -> вклад (0–100)
# ---------------------------------------------------------------------------
# Кусочно-линейная функция: чем глубже RSI в перепроданности, тем выше
# готовность к лонгу. Зеркально для шорта.
_LONG_X = [0, 20, 30, 40, 50, 60, 100]
_LONG_Y = [100, 100, 82, 55, 30, 10, 0]


def long_readiness_from_rsi(rsi_value: float) -> float:
    """Вклад одного таймфрейма в готовность к лонгу (0–100)."""
    return float(np.interp(rsi_value, _LONG_X, _LONG_Y))


def short_readiness_from_rsi(rsi_value: float) -> float:
    """Зеркально: RSI 70 для шорта = RSI 30 для лонга."""
    return long_readiness_from_rsi(100 - rsi_value)


def direction_bias(rsi_value: float) -> int:
    """
    Грубое направление, которое «голосует» таймфрейм:
        +1 — условия скорее за лонг   (RSI < 42)
        -1 — условия скорее за шорт   (RSI > 58)
         0 — нейтрально
    Используется для расчета Alignment Score.
    """
    if rsi_value < 42:
        return 1
    if rsi_value > 58:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Структуры результата
# ---------------------------------------------------------------------------
@dataclass
class TimeframeScore:
    """Все, что нужно UI про один таймфрейм."""
    timeframe: str
    role: str
    weight: float
    rsi: float
    slope: float                 # изменение RSI за последние ~3 свечи
    zone: RSIZone
    long_contribution: float     # 0–100 до взвешивания
    short_contribution: float
    explanation: str             # человеческое пояснение


@dataclass
class SignalReport:
    """Итог скоринга по всем таймфреймам."""
    long_score: float            # Long Readiness 0–100
    short_score: float           # Short Readiness 0–100
    alignment: float             # Multi-TF Alignment 0–100
    confidence: float            # Signal Confidence 0–100
    signal: str                  # 'LONG_SETUP' | 'SHORT_SETUP' | 'NO_TRADE'
    traffic_light: str           # 'green' | 'yellow' | 'red'
    headline: str                # короткий вердикт для шапки
    explanation_lines: list[str] = field(default_factory=list)  # подробный разбор
    per_timeframe: dict[str, TimeframeScore] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Пояснения по таймфрейму
# ---------------------------------------------------------------------------
def _explain_timeframe(tf: str, rsi_value: float, slope: float, zone: RSIZone) -> str:
    """Простое объяснение, что значит текущий RSI на этом таймфрейме."""
    trend_word = "разворачивается вверх" if slope > 1.5 else (
        "продолжает падать" if slope < -1.5 else "движется вбок")

    if zone.key == "oversold":
        base = f"RSI {rsi_value:.0f} — перепроданность. Потенциал отскока есть"
        return base + (", и RSI уже начал расти — ранний признак разворота."
                       if slope > 1.5 else
                       f", но RSI {trend_word}: дно может быть не сформировано.")
    if zone.key == "near_oversold":
        return (f"RSI {rsi_value:.0f} — подходит к перепроданности. "
                "Продавцы давят, но экстремума еще нет.")
    if zone.key == "neutral":
        return (f"RSI {rsi_value:.0f} — нейтральная зона. Этот таймфрейм "
                "не дает преимущества ни лонгу, ни шорту.")
    if zone.key == "near_overbought":
        return (f"RSI {rsi_value:.0f} — подходит к перекупленности. "
                "Покупатели активны, но перегрева еще нет.")
    # overbought
    base = f"RSI {rsi_value:.0f} — перекупленность. Потенциал коррекции есть"
    return base + (", и RSI уже начал снижаться — ранний признак разворота."
                   if slope < -1.5 else
                   f", но RSI {trend_word}: вершина может быть не сформирована.")


# ---------------------------------------------------------------------------
# Главная функция скоринга
# ---------------------------------------------------------------------------
def compute_signal(rsi_series_by_tf: dict[str, pd.Series]) -> SignalReport:
    """
    Принимает словарь {таймфрейм: серия RSI}, возвращает полный отчет.

    Шаги:
      1. По каждому ТФ: текущее значение RSI, наклон, зона, вклады в long/short.
      2. Взвешенная сумма вкладов -> Long/Short Readiness.
      3. Согласованность направлений -> Alignment.
      4. Штрафы/вето от дневного ТФ -> Confidence и светофор.
    """
    per_tf: dict[str, TimeframeScore] = {}

    for tf, cfg in TIMEFRAME_CONFIG.items():
        series = rsi_series_by_tf.get(tf)
        if series is None or series.dropna().empty:
            raise ValueError(f"Нет данных RSI для таймфрейма {tf}.")
        value = float(series.dropna().iloc[-1])
        slope = rsi_slope(series)
        zone = classify_zone(value)

        long_c = long_readiness_from_rsi(value)
        short_c = short_readiness_from_rsi(value)

        # Бонус за начавшийся разворот: перепроданность + RSI пошел вверх
        # надежнее, чем перепроданность с падающим RSI («падающий нож»).
        if value < 40 and slope > 1.5:
            long_c = min(100.0, long_c + 8)
        if value < 40 and slope < -1.5:
            long_c = max(0.0, long_c - 10)
        if value > 60 and slope < -1.5:
            short_c = min(100.0, short_c + 8)
        if value > 60 and slope > 1.5:
            short_c = max(0.0, short_c - 10)

        per_tf[tf] = TimeframeScore(
            timeframe=tf,
            role=cfg["role"],
            weight=cfg["weight"],
            rsi=value,
            slope=slope,
            zone=zone,
            long_contribution=long_c,
            short_contribution=short_c,
            explanation=_explain_timeframe(tf, value, slope, zone),
        )

    # --- 2. Взвешенные readiness-оценки -----------------------------------
    long_score = sum(s.long_contribution * s.weight for s in per_tf.values())
    short_score = sum(s.short_contribution * s.weight for s in per_tf.values())

    # --- 3. Alignment: насколько таймфреймы голосуют в одну сторону -------
    votes = {tf: direction_bias(s.rsi) for tf, s in per_tf.items()}
    weighted_vote = sum(votes[tf] * per_tf[tf].weight for tf in per_tf)
    # |weighted_vote| = 1.0, когда ВСЕ ТФ голосуют в одну сторону
    alignment = round(abs(weighted_vote) * 100, 1)

    # --- 4. Вето старших таймфреймов и confidence --------------------------
    d_rsi = per_tf["1d"].rsi
    d_slope = per_tf["1d"].slope
    h4_rsi = per_tf["4h"].rsi

    daily_veto_long = d_rsi < DAILY_DOWNTREND_RSI and d_slope < -1.0   # дневка камнем вниз
    daily_veto_short = d_rsi > DAILY_UPTREND_RSI and d_slope > 1.0     # дневка уверенно вверх

    explanation: list[str] = []

    if daily_veto_long:
        long_score *= 0.6
        explanation.append(
            "⚠️ Дневной RSI низкий и продолжает падать — глобальный поток против лонгов, "
            "оценка лонга срезана."
        )
    if daily_veto_short:
        short_score *= 0.6
        explanation.append(
            "⚠️ Дневной RSI высокий и продолжает расти — глобальный поток против шортов, "
            "оценка шорта срезана."
        )

    long_score = round(min(100.0, long_score), 1)
    short_score = round(min(100.0, short_score), 1)

    # Confidence: преобладающая readiness + согласованность ТФ
    dominant = max(long_score, short_score)
    confidence = round(0.65 * dominant + 0.35 * alignment, 1)

    # --- 5. Тип сигнала -----------------------------------------------------
    # Требуем явного преимущества одной стороны, иначе NO_TRADE
    if dominant < 45 or abs(long_score - short_score) < 12:
        signal = "NO_TRADE"
    elif long_score > short_score:
        signal = "LONG_SETUP"
    else:
        signal = "SHORT_SETUP"

    # --- 6. Светофор --------------------------------------------------------
    # green: старшие ТФ (4h обязательно) поддерживают направление сигнала
    # yellow: триггер от младших есть, старшие нейтральны/не подтвердили
    # red: сигнала нет или старшие ТФ против
    if signal == "LONG_SETUP":
        senior_support = h4_rsi < 45 and not daily_veto_long
        junior_trigger = per_tf["15m"].rsi < 35 or per_tf["1h"].rsi < 38
    elif signal == "SHORT_SETUP":
        senior_support = h4_rsi > 55 and not daily_veto_short
        junior_trigger = per_tf["15m"].rsi > 65 or per_tf["1h"].rsi > 62
    else:
        senior_support = False
        junior_trigger = False

    if signal == "NO_TRADE":
        traffic = "red"
        headline = "Сделки нет: таймфреймы не дают согласованного преимущества."
    elif senior_support and confidence >= 60:
        traffic = "green"
        side = "лонг" if signal == "LONG_SETUP" else "шорт"
        headline = f"Сетап на {side} подтвержден старшими таймфреймами."
    elif junior_trigger and not senior_support:
        traffic = "yellow"
        side = "лонг" if signal == "LONG_SETUP" else "шорт"
        headline = f"Ранний сигнал на {side}: младшие ТФ сработали, старшие НЕ подтвердили. Ждать."
    elif senior_support:
        traffic = "yellow"
        headline = "Контекст подходящий, но уверенность недостаточна. Наблюдать."
    else:
        traffic = "red"
        headline = "Старшие таймфреймы против сделки. Вход сейчас — высокий риск."

    # --- 7. Развернутое объяснение -----------------------------------------
    explanation.extend(_build_narrative(per_tf, signal, long_score, short_score, alignment))

    return SignalReport(
        long_score=long_score,
        short_score=short_score,
        alignment=alignment,
        confidence=confidence,
        signal=signal,
        traffic_light=traffic,
        headline=headline,
        explanation_lines=explanation,
        per_timeframe=per_tf,
    )


def _build_narrative(per_tf: dict[str, TimeframeScore], signal: str,
                     long_score: float, short_score: float, alignment: float) -> list[str]:
    """Собирает разбор ситуации «человеческим языком», снизу вверх по иерархии ТФ."""
    lines: list[str] = []

    s15, s1h, s4h, s1d = per_tf["15m"], per_tf["1h"], per_tf["4h"], per_tf["1d"]

    lines.append(f"**15m (триггер):** {s15.explanation}")
    lines.append(f"**1h (подтверждение):** {s1h.explanation}")
    lines.append(f"**4h (контекст):** {s4h.explanation}")
    lines.append(f"**1d (фильтр):** {s1d.explanation}")

    if alignment >= 70:
        lines.append("Таймфреймы хорошо согласованы между собой — сигнал однородный.")
    elif alignment >= 40:
        lines.append("Согласованность средняя: часть таймфреймов нейтральна или против.")
    else:
        lines.append("Таймфреймы противоречат друг другу — классическая ситуация "
                     "ложного сигнала на младшем ТФ.")

    if signal == "NO_TRADE":
        lines.append(
            f"Итог: Long {long_score:.0f} / Short {short_score:.0f} — ни одна сторона не набрала "
            "достаточного преимущества. Лучшая сделка сейчас — ее отсутствие."
        )
    else:
        side = "лонга" if signal == "LONG_SETUP" else "шорта"
        lines.append(
            f"Итог: преимущество на стороне {side} "
            f"(Long {long_score:.0f} / Short {short_score:.0f}). "
            "Помните: это оценка готовности рынка, а не приказ входить."
        )
    return lines
