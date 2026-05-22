# Binance Price Alert Bot

Telegram bot that monitors Binance prices and sends alerts when a price changes beyond a defined threshold. Data is stored locally in an Excel file.

## How It Works

The bot fetches all Binance prices in a single request every N minutes. If a ticker's price changes by more than the defined threshold (up or down), it sends a Telegram alert and updates the reference price in the Excel file.

## Requirements

- Oracle Ubuntu 24.04
- Python 3.13
- Telegram bot token from [@BotFather](https://t.me/BotFather)

## Server Setup

### 1. Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install dependencies

```bash
sudo apt install python3-pip python3-dev git software-properties-common -y
```

### 3. Install Python 3.13

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev -y
```

### 4. Clone the repository

```bash
git clone https://github.com/kirillmuhanko/binance-price-alert-bot.git
cd binance-price-alert-bot
```

### 5. Create virtual environment

```bash
python3.13 -m venv venv
source venv/bin/activate
```

### 6. Install Python packages

```bash
pip install -r requirements.txt
```

### 7. Configure environment variables

```bash
cp .env.example .env
nano .env
```

Fill in your values:

```
TELEGRAM_TOKEN=your_token_here
CHAT_ID=your_chat_id_here
CHECK_INTERVAL=15
```

Save with `Ctrl+O` → Enter → `Ctrl+X`

### 8. Test run

```bash
python bot.py
```

Send `/start` to your bot in Telegram to verify it responds. Then stop with `Ctrl+C`.

### 9. Create systemd service

```bash
sudo nano /etc/systemd/system/binance-bot.service
```

Paste the following:

```ini
[Unit]
Description=Binance Price Alert Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/binance-price-alert-bot
ExecStart=/home/ubuntu/binance-price-alert-bot/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+O` → Enter → `Ctrl+X`

### 10. Enable and start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-bot
sudo systemctl start binance-bot
```

### 11. Verify it is running

```bash
sudo systemctl status binance-bot
```

## Useful Commands

```bash
# View live logs
sudo journalctl -u binance-bot -f

# Restart the bot
sudo systemctl restart binance-bot

# Stop the bot
sudo systemctl stop binance-bot

# Pull latest code and restart
git pull
sudo systemctl restart binance-bot
```

## Bot Commands

| Command | Description |
|---------|-------------|
| `/add BTCUSDT 15 -15` | Add ticker with thresholds |
| `/remove BTCUSDT` | Remove ticker |
| `/list` | Show watchlist with thresholds |
| `/prices` | Current prices for all tickers |
| `/setthreshold BTCUSDT 10 -10` | Update thresholds |
| `/export` | Download Excel file in chat |

## Getting Your Chat ID

1. Start a conversation with your bot
2. Open in browser: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id": YOUR_CHAT_ID}`
