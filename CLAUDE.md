# Project Context

## Analytics Dashboards

- `btc_rsi_dashboard/` — **Когда?** перегрет ли импульс (RSI по 4 таймфреймам)
- `funding_oi_dashboard/` — **Кто?** куда перекошена толпа (Funding Rate + Open Interest)
- `cvd_dashboard/` — **Как?** кто давит маркет-ордерами (Cumulative Volume Delta)
- `volume_profile_dashboard/` — **Где?** на каких уровнях сидит объём (POC, Value Area, вакуум)

Каждый дашборд автономен (один файл `app.py`), без API-ключей, с demo-fallback.
Запуск: `streamlit run <folder>/app.py`
