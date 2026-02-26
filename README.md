# StockSage AI

AI/ML-powered Indian stock market prediction, paper trading, and competitive leaderboard platform.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%2FML-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-In_Development-yellow?style=flat)

## Overview

StockSage AI is a platform that combines machine learning-based stock prediction with gamified paper trading for the Indian stock market (NSE/BSE). Users can practice trading with virtual capital, compete on leaderboards, and benchmark their performance against an AI trader -- all without risking real money.

## Features

### 1. AI-Powered Stock Prediction
- Predicts stock price movement (up/down) using historical data, news sentiment, political events, and company earnings
- Real-time predictions when a user opens any stock page
- Trained on Indian market data (NSE/BSE)

### 2. Paper Trading
- Simulated trading with real-time Indian market prices
- No real money involved -- practice and learn risk-free
- Execute buy/sell orders on NSE/BSE listed stocks
- Track portfolio performance over time

### 3. Leaderboard
- Every user starts with Rs 1,00,000 virtual capital
- Rankings based on portfolio performance (percentage gain/loss)
- Compete with other traders on the platform
- Weekly/monthly/all-time leaderboard views

### 4. AI vs Human Competition
- An AI trader starts with the same Rs 1,00,000 capital
- AI makes its own trading decisions using the prediction engine
- See how you stack up against the machine
- Learn from AI trading patterns

## Target Market

Indian Stock Market -- NSE (National Stock Exchange) and BSE (Bombay Stock Exchange)

## Planned Architecture

```
StockSage-AI/
├── backend/                # FastAPI backend
│   ├── api/               # REST API endpoints
│   ├── ml/                # ML prediction models
│   ├── trading/           # Paper trading engine
│   └── data/              # Market data pipeline
├── frontend/              # React/Next.js frontend
│   ├── pages/             # Stock pages, leaderboard, portfolio
│   └── components/        # Charts, order forms, rankings
├── models/                # Trained ML models
└── scripts/               # Data collection, training scripts
```

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | Architecture planning and tech stack selection | In Progress |
| 2 | Market data pipeline (NSE/BSE) | Planned |
| 3 | ML prediction model (historical + sentiment) | Planned |
| 4 | Paper trading engine with virtual portfolio | Planned |
| 5 | Frontend dashboard and stock pages | Planned |
| 6 | Leaderboard and competition system | Planned |
| 7 | AI trader agent | Planned |
| 8 | Beta launch and testing | Planned |

## Data Sources

- **Market Data**: NSE India, BSE India, Yahoo Finance
- **News Sentiment**: Financial news APIs
- **Company Fundamentals**: Quarterly earnings, filings

## Contributing

This project is in early development. If you're interested in contributing, please open an issue to discuss your ideas.

## License

MIT
