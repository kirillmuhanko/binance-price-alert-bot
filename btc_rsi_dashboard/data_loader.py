"""
data_loader.py
==============
Загрузка OHLCV-данных для дашборда.

Архитектура:
    - Абстрактный интерфейс DataSource — чтобы источник данных можно было
      заменить (другая биржа, CSV, своя БД), не трогая остальной код.
    - CCXTSource    — реальные данные с Binance (или Bybit/OKX/Kraken) через ccxt.
    - BinanceRESTSource — прямой запрос к публичному REST API Binance
      (работает даже без установленного ccxt).
    - DemoSource    — синтетические данные, если сеть/биржа недоступны,
      чтобы дашборд всегда можно было открыть и посмотреть.

Это аналитический инструмент: никакие ордера не отправляются,
API-ключи не нужны — используются только публичные котировки.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# Таймфреймы, которые использует дашборд (от младшего к старшему)
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Сколько свечей запрашивать на каждый таймфрейм
DEFAULT_LIMIT = 300

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataSourceError(Exception):
    """Единый тип ошибки загрузки данных — UI ловит именно его."""


@dataclass
class FetchResult:
    """Результат загрузки: датафрейм + метка источника (для отображения в UI)."""
    df: pd.DataFrame
    source_name: str


class DataSource(ABC):
    """Интерфейс источника данных. Реализуйте его, чтобы подключить свой источник."""

    name: str = "abstract"

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        """
        Возвращает DataFrame с колонками OHLCV_COLUMNS,
        timestamp — pd.Timestamp (UTC), отсортировано по возрастанию времени.
        """
        raise NotImplementedError


def _to_dataframe(raw: list) -> pd.DataFrame:
    """Преобразует «сырой» список [[ts, o, h, l, c, v], ...] в стандартный DataFrame."""
    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise DataSourceError("Получен пустой набор свечей.")
    return df


# ---------------------------------------------------------------------------
# 1. ccxt: Binance и другие биржи
# ---------------------------------------------------------------------------
class CCXTSource(DataSource):
    """
    Источник на базе ccxt. По умолчанию Binance, но exchange_id можно
    поменять на 'bybit', 'okx', 'kraken' и т.д. — это полезно, если
    Binance недоступен в вашем регионе.
    """

    def __init__(self, exchange_id: str = "binance"):
        try:
            import ccxt  # импорт внутри, чтобы дашборд работал и без ccxt
        except ImportError as exc:
            raise DataSourceError(
                "Библиотека ccxt не установлена. Выполните: pip install ccxt"
            ) from exc

        if not hasattr(ccxt, exchange_id):
            raise DataSourceError(f"ccxt не знает биржу '{exchange_id}'.")

        # enableRateLimit — ccxt сам соблюдает лимиты запросов биржи
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.name = f"ccxt:{exchange_id}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # ccxt бросает много разных типов — сводим к одному
            raise DataSourceError(
                f"Ошибка {self.name} при загрузке {symbol} {timeframe}: {exc}"
            ) from exc
        return _to_dataframe(raw)


# ---------------------------------------------------------------------------
# 2. Прямой REST Binance (без ccxt)
# ---------------------------------------------------------------------------
class BinanceRESTSource(DataSource):
    """
    Публичный эндпоинт Binance /api/v3/klines. Ключи не нужны.
    Символ 'BTC/USDT' автоматически превращается в 'BTCUSDT'.
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"
    name = "binance-rest"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        params = {
            "symbol": symbol.replace("/", "").upper(),
            "interval": timeframe,
            "limit": min(limit, 1000),
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise DataSourceError(f"Ошибка Binance REST для {symbol} {timeframe}: {exc}") from exc

        if isinstance(data, dict) and "code" in data:
            raise DataSourceError(f"Binance вернул ошибку: {data}")

        # Формат kline: [openTime, open, high, low, close, volume, ...] — берем первые 6 полей
        raw = [row[:6] for row in data]
        return _to_dataframe(raw)


# ---------------------------------------------------------------------------
# 3. Демо-данные (офлайн-режим)
# ---------------------------------------------------------------------------
class DemoSource(DataSource):
    """
    Генерирует правдоподобный случайный ряд цены BTC (геометрическое
    случайное блуждание). Нужен, чтобы интерфейс можно было посмотреть
    без интернета. В UI явно помечается как DEMO.
    """

    name = "demo"

    _TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        minutes = self._TF_MINUTES.get(timeframe, 60)
        # Один seed на все таймфреймы (в пределах часа), чтобы данные были согласованы
        rng = np.random.default_rng(seed=int(time.time() // 3600))
        n = limit

        end = pd.Timestamp.utcnow().floor(f"{minutes}min")
        idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")

        # Волатильность масштабируем под таймфрейм (~корень из времени)
        step_vol = 0.004 * np.sqrt(minutes / 60)
        returns = rng.normal(0, step_vol, n)
        close = 100_000 * np.exp(np.cumsum(returns))

        spread = np.abs(rng.normal(0, step_vol, n)) * close
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        df = pd.DataFrame({
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        })
        return df


# ---------------------------------------------------------------------------
# Фабрика и высокоуровневая загрузка
# ---------------------------------------------------------------------------
def build_source(source_key: str) -> DataSource:
    """
    Создает источник по ключу из UI.
    Поддерживаемые ключи: 'ccxt-binance', 'ccxt-bybit', 'ccxt-okx',
    'binance-rest', 'demo'.
    """
    if source_key.startswith("ccxt-"):
        return CCXTSource(exchange_id=source_key.split("-", 1)[1])
    if source_key == "binance-rest":
        return BinanceRESTSource()
    if source_key == "demo":
        return DemoSource()
    raise DataSourceError(f"Неизвестный источник данных: {source_key}")


def load_all_timeframes(
    source_key: str,
    symbol: str = "BTC/USDT",
    timeframes: list[str] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, FetchResult]:
    """
    Загружает OHLCV по всем таймфреймам.

    Стратегия отказоустойчивости: если выбранный источник падает,
    пробуем запасные (Binance REST -> demo), чтобы дашборд не «умирал»
    с пустым экраном. Фактический источник виден в FetchResult.source_name.
    """
    timeframes = timeframes or TIMEFRAMES
    fallback_chain = [source_key]
    for fb in ("binance-rest", "demo"):
        if fb not in fallback_chain:
            fallback_chain.append(fb)

    last_error: Exception | None = None
    for key in fallback_chain:
        try:
            source = build_source(key)
            result = {
                tf: FetchResult(df=source.fetch_ohlcv(symbol, tf, limit), source_name=source.name)
                for tf in timeframes
            }
            return result
        except DataSourceError as exc:
            last_error = exc
            continue

    raise DataSourceError(f"Не удалось загрузить данные ни из одного источника: {last_error}")
