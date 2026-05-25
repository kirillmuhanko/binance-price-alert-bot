import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta

import requests
import openpyxl
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 15)) * 60
AUDIT_HOUR_UTC = int(os.getenv("AUDIT_HOUR_UTC", 18))
DATA_FILE      = "watchlist.xlsx"

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise RuntimeError("TELEGRAM_TOKEN and CHAT_ID must be set in .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = ["ticker", "current_price", "last_check_time", "last_update_time", "notify_above_%", "notify_below_%"]


# ----------------------------- Excel helpers ----------------------------- #

def _init_workbook():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Watchlist"
    ws.append(HEADERS)
    wb.save(DATA_FILE)


def load_watchlist() -> dict:
    if not os.path.exists(DATA_FILE):
        _init_workbook()
        return {}

    wb = openpyxl.load_workbook(DATA_FILE)
    ws = wb.active
    result = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        ticker = str(row[0]).upper()
        result[ticker] = {
            "ticker":           ticker,
            "current_price":    float(row[1]) if row[1] else 0.0,
            "last_check_time":  row[2],
            "last_update_time": row[3],
            "notify_above":     float(row[4]),
            "notify_below":     float(row[5]),
        }
    return result


def save_ticker(item: dict):
    if not os.path.exists(DATA_FILE):
        _init_workbook()

    wb  = openpyxl.load_workbook(DATA_FILE)
    ws  = wb.active
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = [
        item["ticker"],
        item["current_price"],
        item.get("last_check_time") or now,
        item.get("last_update_time") or now,
        item["notify_above"],
        item["notify_below"],
    ]

    for row in ws.iter_rows(min_row=2):
        if row[0].value and str(row[0].value).upper() == item["ticker"]:
            for col, val in enumerate(row_data):
                row[col].value = val
            wb.save(DATA_FILE)
            return

    ws.append(row_data)
    wb.save(DATA_FILE)


def delete_ticker(ticker: str):
    if not os.path.exists(DATA_FILE):
        return

    wb = openpyxl.load_workbook(DATA_FILE)
    ws = wb.active
    for row in ws.iter_rows(min_row=2):
        if row[0].value and str(row[0].value).upper() == ticker:
            ws.delete_rows(row[0].row)
            break
    wb.save(DATA_FILE)


def update_check_time(ticker: str):
    if not os.path.exists(DATA_FILE):
        return

    wb  = openpyxl.load_workbook(DATA_FILE)
    ws  = wb.active
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for row in ws.iter_rows(min_row=2):
        if not row[0].value or str(row[0].value).upper() != ticker:
            continue
        row[2].value = now
        break

    wb.save(DATA_FILE)


# ----------------------------- Binance API (только для алертов) ----------------------------- #

def fetch_prices(symbols: list[str]) -> dict[str, float]:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=10)
        r.raise_for_status()
        return {i["symbol"]: float(i["price"]) for i in r.json() if i["symbol"] in symbols}
    except Exception as e:
        logger.error(f"Binance API error: {e}")
        return {}


# ----------------------------- Gemini (сам ищет в интернете) ----------------------------- #

AUDIT_PROMPT_TEMPLATE = """Ты — крипто-аналитик. Проведи аудит токена {ticker} прямо сейчас.

Используй поиск в интернете, чтобы найти АКТУАЛЬНЫЕ данные:
- Funding rate (текущий и тренд за сутки) на Binance / Bybit perpetuals
- Open Interest и его изменение за 24ч
- Long/Short ratio, особенно top traders
- Объём торгов 24ч, аномалии
- Изменение цены за 24ч и 7д
- Расхождение mark price vs index price
- СВЕЖИЕ новости: листинги/делистинги, хаки, партнёрства, разлоки токенов, апдейты протокола
- Активность китов / крупные транзакции
- Любые НЕОБЫЧНЫЕ сигналы

Затем РЕШИ: есть ли ДЕЙСТВИТЕЛЬНО ВАЖНОЕ для трейдера прямо сейчас?

Критерии "важно":
- Аномальный funding rate (|>0.05%| на 8ч или резкая смена знака)
- Дивергенция: цена ↑ а OI ↓, или цена ↓ а OI ↑
- Экстремум long/short ratio (>2.5 или <0.5)
- Существенное расхождение mark vs index
- Свежая новость, способная двинуть цену
- Резкий всплеск объёма без новостей
- Заметная активность китов

Если НИЧЕГО важного нет — верни СТРОГО JSON: {{"important": false}}

Если есть — верни СТРОГО JSON:
{{"important": true, "summary": "краткая сводка одним абзацем на русском, с эмодзи 🟢🔴⚠️📈📉 и конкретными цифрами"}}

Никакого текста до или после JSON. Только JSON."""


def gemini_audit_ticker(ticker: str) -> dict | None:
    """Gemini сам ищет в интернете и решает, важно ли. Возвращает dict или None."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, skipping audit")
        return None

    prompt = AUDIT_PROMPT_TEMPLATE.format(ticker=ticker)

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        r = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],   # встроенный поиск Gemini
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Чистим возможные ```json ... ``` обёртки
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Достаём JSON-объект даже если есть лишний текст вокруг
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning(f"{ticker}: no JSON in response: {text[:200]}")
            return None

        return json.loads(match.group(0))

    except Exception as e:
        logger.error(f"Gemini error for {ticker}: {e}")
        return None


# ----------------------------- Loops ----------------------------- #

async def monitor_loop(bot: Bot):
    logger.info("Monitor started.")
    while True:
        try:
            watchlist = load_watchlist()
            if watchlist:
                prices = fetch_prices(list(watchlist.keys()))
                now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for ticker, data in watchlist.items():
                    new_price = prices.get(ticker)
                    if new_price is None:
                        continue

                    old_price = data["current_price"]

                    if old_price == 0:
                        data["current_price"]    = new_price
                        data["last_check_time"]  = now
                        data["last_update_time"] = now
                        save_ticker(data)
                        continue

                    update_check_time(ticker)

                    change_pct = (new_price - old_price) / old_price * 100
                    triggered  = change_pct >= data["notify_above"] or change_pct <= data["notify_below"]

                    if triggered:
                        emoji = "🟢" if change_pct > 0 else "🔴"
                        await bot.send_message(
                            chat_id=CHAT_ID,
                            text=(
                                f"{emoji} *{ticker}* changed by *{change_pct:+.2f}%*\n"
                                f"Previous price: `${old_price:,.4f}`\n"
                                f"Current price: `${new_price:,.4f}`"
                            ),
                            parse_mode="Markdown",
                        )
                        logger.info(f"Alert: {ticker} {change_pct:+.2f}%")
                        data["current_price"]    = new_price
                        data["last_update_time"] = now
                        save_ticker(data)

        except Exception as e:
            logger.error(f"Monitor error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)


async def audit_loop(bot: Bot):
    """Ждёт AUDIT_HOUR_UTC каждый день и запускает аудит."""
    logger.info(f"Audit scheduler started (daily at {AUDIT_HOUR_UTC:02d}:00 UTC).")
    while True:
        now_utc = datetime.now(timezone.utc)
        next_run = now_utc.replace(hour=AUDIT_HOUR_UTC, minute=0, second=0, microsecond=0)
        if next_run <= now_utc:
            next_run += timedelta(days=1)

        wait_sec = (next_run - now_utc).total_seconds()
        logger.info(f"Next audit at {next_run.isoformat()} (in {wait_sec/3600:.2f}h)")
        await asyncio.sleep(wait_sec)

        try:
            await run_audit(bot)
        except Exception as e:
            logger.error(f"Audit error: {e}")


async def run_audit(bot: Bot):
    """По одному тикеру: спрашиваем Gemini. Копим важные находки.
    В конце шлём ОДНО сообщение. Если ничего важного — полная тишина."""
    watchlist = load_watchlist()
    if not watchlist:
        logger.info("Audit: watchlist empty, skipping.")
        return

    logger.info(f"Audit started for {len(watchlist)} tickers")
    findings = []

    for ticker in watchlist:
        logger.info(f"Auditing {ticker}...")
        result = await asyncio.to_thread(gemini_audit_ticker, ticker)

        if result and result.get("important") and result.get("summary"):
            findings.append((ticker, result["summary"]))
            logger.info(f"{ticker}: important")
        else:
            logger.info(f"{ticker}: nothing important")

        # лёгкий троттлинг, чтобы не упереться в rate limit Gemini
        await asyncio.sleep(2)

    if not findings:
        logger.info(f"Audit finished. Nothing important across {len(watchlist)} tickers. Staying silent.")
        return

    # Собираем единое сообщение
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"📊 *Вечерний аудит* — {ts}\n_Важные сигналы: {len(findings)}/{len(watchlist)}_\n"]
    for ticker, summary in findings:
        parts.append(f"\n*{ticker}*\n{summary}")
    text = "\n".join(parts)

    # Telegram лимит 4096 символов — режем при необходимости
    LIMIT = 4000
    chunks = []
    while text:
        if len(text) <= LIMIT:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, LIMIT)
        if cut == -1:
            cut = LIMIT
        chunks.append(text[:cut])
        text = text[cut:].lstrip()

    for chunk in chunks:
        try:
            await bot.send_message(chat_id=CHAT_ID, text=chunk, parse_mode="Markdown")
        except Exception:
            await bot.send_message(chat_id=CHAT_ID, text=chunk)

    logger.info(f"Audit finished. {len(findings)} findings sent in {len(chunks)} chunk(s).")


# ----------------------------- Commands ----------------------------- #

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Binance Price Alert Bot*\n\n"
        "`/add BTCUSDT 5 -5` — add ticker with thresholds\n"
        "`/remove BTCUSDT` — remove ticker\n"
        "`/list` — show watchlist\n"
        "`/prices` — current prices\n"
        "`/setthreshold BTCUSDT 10 -10` — update thresholds\n"
        "`/export` — download Excel file\n"
        "`/audit` — run AI audit now",
        parse_mode="Markdown",
    )


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Usage: `/add BTCUSDT 5 -5`", parse_mode="Markdown")
        return

    ticker = args[0].upper()
    above  = float(args[1])
    below  = float(args[2])

    watchlist = load_watchlist()
    if ticker in watchlist:
        await update.message.reply_text(f"⚠️ {ticker} is already in the watchlist.")
        return

    prices = fetch_prices([ticker])
    price  = prices.get(ticker, 0.0)
    if price == 0.0:
        await update.message.reply_text(f"❌ Ticker *{ticker}* not found on Binance.", parse_mode="Markdown")
        return

    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    item = {
        "ticker": ticker, "current_price": price,
        "last_check_time": now, "last_update_time": now,
        "notify_above": above, "notify_below": below,
    }
    save_ticker(item)
    await update.message.reply_text(
        f"✅ *{ticker}* added\nPrice: `${price:,.4f}`\nAlert: *+{above}%* / *{below}%*",
        parse_mode="Markdown",
    )


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: `/remove BTCUSDT`", parse_mode="Markdown")
        return

    ticker = context.args[0].upper()
    if ticker not in load_watchlist():
        await update.message.reply_text(f"❌ {ticker} not found in watchlist.")
        return

    delete_ticker(ticker)
    await update.message.reply_text(f"🗑 *{ticker}* removed.", parse_mode="Markdown")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Watchlist is empty. Add a ticker: `/add BTCUSDT 5 -5`", parse_mode="Markdown")
        return

    lines = ["*Watchlist:*\n"]
    for t, d in watchlist.items():
        lines.append(f"• *{t}* | +{d['notify_above']}% / {d['notify_below']}%")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_prices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Watchlist is empty.")
        return

    prices = fetch_prices(list(watchlist.keys()))
    lines  = ["*Current prices:*\n"]
    for ticker in watchlist:
        p = prices.get(ticker)
        lines.append(f"• *{ticker}*: `${p:,.4f}`" if p else f"• *{ticker}*: N/A")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_setthreshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Usage: `/setthreshold BTCUSDT 10 -10`", parse_mode="Markdown")
        return

    ticker    = args[0].upper()
    watchlist = load_watchlist()
    if ticker not in watchlist:
        await update.message.reply_text(f"❌ {ticker} not in watchlist.")
        return

    watchlist[ticker]["notify_above"] = float(args[1])
    watchlist[ticker]["notify_below"] = float(args[2])
    save_ticker(watchlist[ticker])
    await update.message.reply_text(
        f"✅ *{ticker}* thresholds updated: *+{args[1]}%* / *{args[2]}%*",
        parse_mode="Markdown",
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(DATA_FILE):
        await update.message.reply_text("File not created yet.")
        return
    await update.message.reply_document(
        document=open(DATA_FILE, "rb"),
        filename=DATA_FILE,
        caption="📊 Current watchlist",
    )


async def cmd_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Запускаю аудит...")
    try:
        await run_audit(context.bot)
        # run_audit сам решит, слать ли сводку. Здесь просто отмечаем, что команда отработала.
        await msg.edit_text("✅ Аудит завершён. (Если ничего не пришло — значит ничего важного)")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {e}")


# ----------------------------- App ----------------------------- #

async def post_init(application: Application):
    asyncio.create_task(monitor_loop(application.bot))
    asyncio.create_task(audit_loop(application.bot))


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",        cmd_start))
    app.add_handler(CommandHandler("add",          cmd_add))
    app.add_handler(CommandHandler("remove",       cmd_remove))
    app.add_handler(CommandHandler("list",         cmd_list))
    app.add_handler(CommandHandler("prices",       cmd_prices))
    app.add_handler(CommandHandler("setthreshold", cmd_setthreshold))
    app.add_handler(CommandHandler("export",       cmd_export))
    app.add_handler(CommandHandler("audit",        cmd_audit))
    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()