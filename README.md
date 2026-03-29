# LucidFlex 50K Gold Futures Trading Bot

Automated trading bot for GC/MGC (Gold Futures) on Rithmic paper trading, designed to pass a LucidFlex 50K prop firm evaluation.

## Quick Start

```bash
pip install -r requirements.txt

python -m lucidflex \
  --user YOUR_RITHMIC_USER \
  --password YOUR_RITHMIC_PASS \
  --system YOUR_SYSTEM_NAME \
  --log-level INFO
```

## News Events

Pass high-impact news event times so the bot can activate the News Breakout setup:

```bash
python -m lucidflex \
  --user USER --password PASS --system SYS \
  --news "2025-07-04T08:30:00-05:00" "2025-07-30T14:00:00-05:00"
```

## Architecture

```
lucidflex/
├── config.py       # All hardcoded rules and optimized parameters
├── market_data.py  # Rithmic data feed, bar aggregation, indicators
├── setups.py       # 5 trade setups (News, ORB, Sweep, VWAP, OB)
├── risk.py         # 3-layer MLL protection, position sizing, eval tracking
├── execution.py    # Order execution, bracket management, fill monitoring
├── bot.py          # Main loop, session management, EOD routine
└── __main__.py     # CLI entry point
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| risk_pct | 1.0% | Equity risked per trade |
| min_rr | 4.5:1 | Minimum risk-to-reward ratio |
| daily_cap | $700 | Stop trading once up this amount |
| soft_loss | $700 | Stop trading once down this amount |
| hard_stop_buf | $200 | Halt if MLL buffer falls below |
| max_trades | 12/day | Maximum trades per session |

## State Persistence

The bot saves evaluation state to `lucidflex_state.json` at EOD and on shutdown. Use `--reset` to start fresh.
