# StockSage-AI: Complete Project Plan

> AI/ML-powered Indian stock market prediction, paper trading, and competitive leaderboard platform.
> **Target Market**: NSE (National Stock Exchange) & BSE (Bombay Stock Exchange)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Data Sources & APIs (All Free)](#4-data-sources--apis-all-free)
5. [Database Schema](#5-database-schema)
6. [ML Pipeline Design](#6-ml-pipeline-design)
7. [Backend API Design](#7-backend-api-design)
8. [Frontend Pages & Components](#8-frontend-pages--components)
9. [Paper Trading Engine](#9-paper-trading-engine)
10. [Leaderboard & AI Trader](#10-leaderboard--ai-trader)
11. [Authentication & Security](#11-authentication--security)
12. [Infrastructure & DevOps](#12-infrastructure--devops)
13. [Indian Market-Specific Rules](#13-indian-market-specific-rules)
14. [Implementation Phases](#14-implementation-phases)
15. [Prerequisites & Setup](#15-prerequisites--setup)
16. [Cost Breakdown](#16-cost-breakdown)
17. [Risks & Mitigations](#17-risks--mitigations)
18. [Version Reference](#version-reference-all-latest-as-of-feb-2026)

---

## 1. Project Overview

### What Is StockSage-AI?

A platform where users can:

- **View AI predictions** for Indian stock price movements (up/down) using ML models trained on historical data, news sentiment, and technical indicators
- **Paper trade** with Rs 1,00,000 virtual capital using real-time NSE/BSE prices — no real money involved
- **Compete on leaderboards** ranked by portfolio performance (% gain/loss) across weekly, monthly, and all-time periods
- **Benchmark against an AI trader** that uses the same prediction engine to make autonomous trading decisions

### Core Features

| Feature | Description |
|---------|-------------|
| AI Prediction | Real-time stock price direction prediction when user opens a stock page |
| Paper Trading | Simulated buy/sell orders with real market prices |
| Portfolio Tracker | Holdings, P&L, transaction history, performance analytics |
| Leaderboard | Competitive ranking of all users by portfolio returns |
| AI Trader | Autonomous AI agent competing against human traders |

---

## 2. Tech Stack

### Backend

| Technology | Purpose | Version |
|-----------|---------|---------|
| Python | Primary backend language | 3.14.x |
| FastAPI | REST API framework | 0.133.x |
| Uvicorn | ASGI server | 0.41.x |
| SQLAlchemy | ORM (database models) | 2.0.47 |
| Alembic | Database migrations | 1.18.x |
| Celery | Background task queue (data fetching, model training) | 5.6.x |
| redis-py | Python Redis client | 7.2.x |
| Pydantic v2 | Request/response validation & settings | 2.12.x |
| PyJWT | JWT token creation/verification | 2.11.x |
| pwdlib + argon2 | Password hashing | 0.3.x |
| httpx | Async HTTP client | 0.28.x |
| slowapi | Rate limiting | 0.1.9 |
| websockets | WebSocket support | 16.x |

> **Why PyJWT over python-jose?** FastAPI's official docs now recommend PyJWT. python-jose is no longer the recommended choice. Install with `pip install pyjwt[crypto]`.
>
> **Why pwdlib over Passlib?** Passlib has not had a release since 2020 and is effectively abandoned. pwdlib is the modern replacement, officially recommended by FastAPI. Uses Argon2 (more secure than bcrypt). Install with `pip install pwdlib[argon2]`.

### Python Tooling

| Tool | Purpose | Version |
|------|---------|---------|
| uv | Package manager (replaces pip, venv, pip-tools, pyenv) | 0.10.x |
| Ruff | Linter + formatter (replaces black, flake8, isort, pylint) | 0.15.x |

> **Why uv?** 10-100x faster than pip, built-in virtual environment and Python version management, universal lockfiles. Created by Astral (same team as Ruff). The community-preferred tool for new Python projects in 2026.
>
> **Why Ruff?** Single tool that replaces black + flake8 + isort + pylint. Written in Rust, extremely fast. Widely adopted across the Python ecosystem.

### ML/AI

| Technology | Purpose | Version |
|-----------|---------|---------|
| pandas | Data manipulation & time-series (Arrow backend by default) | 3.0.x |
| numpy | Numerical computation | 2.4.x |
| scikit-learn | Baseline ML models (Random Forest, Logistic Regression) | 1.8.x |
| XGBoost | Gradient boosting (primary model for tabular stock data) | 3.2.x |
| LightGBM | Alternative gradient boosting model | 4.6.x |
| PyTorch (CUDA) | LSTM / Transformer models for time-series (GPU-accelerated) | 2.10.x |
| Hugging Face Transformers | FinBERT for news sentiment analysis (GPU inference) | 5.2.x |
| pandas-ta | Technical indicators (RSI, MACD, Bollinger Bands, etc.) | 0.4.x |
| MLflow | Experiment tracking, model versioning, model registry | 3.10.x |
| Optuna | Hyperparameter tuning | 4.7.x |
| joblib | Model serialization | 1.5.x |

> **Why pandas-ta over ta-lib?** pandas-ta is pure Python -- installs with `pip install pandas-ta` on Windows without needing a C library. ta-lib requires a pre-compiled C library and frequently has 32/64-bit architecture mismatch issues on Windows. Backup option: `stockstats` (0.6.8, actively maintained Feb 2026).
>
> **Why pandas over polars?** pandas 3.0 now uses Apache Arrow backend by default, closing much of the performance gap. Every library in our stack (yfinance, pandas-ta, jugaad-data, scikit-learn, XGBoost) returns/expects pandas DataFrames. Using polars would require constant conversions, negating its performance advantage for our dataset sizes.
>
> **FinBERT for sentiment:** ProsusAI/finbert remains the gold standard for financial sentiment analysis (5.6M+ downloads/month). Lighter alternative: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` for faster inference.
>
> **GPU Training:** All ML training runs inside WSL2 with NVIDIA CUDA. PyTorch is installed with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu124`). XGBoost and LightGBM also support `device='cuda'` for GPU-accelerated training. FinBERT inference runs on GPU for batch sentiment scoring.

### Frontend

| Technology | Purpose | Version |
|-----------|---------|---------|
| Next.js | React framework (App Router) | 16.x |
| React | UI library | 19.x |
| TypeScript | Type-safe JavaScript | 5.9.x |
| TailwindCSS | Utility-first CSS framework (v4 major rewrite) | 4.2.x |
| TradingView Lightweight Charts | Stock price charts (industry standard, ~35KB gzipped) | 5.1.x |
| TanStack React Query | Server state management & caching | 5.90.x |
| Zustand | Client-side state management | 5.0.x |
| Auth.js (next-auth v5) | Authentication (OAuth + credentials) | 5.x beta |
| Socket.io Client | Real-time WebSocket for live prices | 4.8.x |
| Zod | Schema validation | 4.3.x |
| Recharts | Secondary charts (portfolio analytics, leaderboard) | 3.7.x |
| shadcn/ui | Component library (CLI v3, 100+ components) | CLI v3 |
| nuqs | Type-safe URL state management for Next.js | 2.8.x |

> **Why Zustand over Jotai?** Zustand has ~21.9M weekly downloads vs Jotai's ~2.9M. Its store-based model is a better fit for centralized state like portfolio data, watchlists, and trading positions. Now at v5 with improved TypeScript support.
>
> **Why Auth.js v5?** Despite the beta label, v5 has been in beta for a long time, is well-documented, and is what the official Auth.js docs recommend. The v4 API is considered legacy. The project is rebranding from NextAuth.js to Auth.js.
>
> **Why TailwindCSS v4?** Major rewrite with significantly faster build times, new CSS-first configuration, and native cascade layers. Not backwards-compatible with v3 config but the new DX is superior.
>
> **Stock charts:** TradingView Lightweight Charts v5 remains the gold standard for free financial charting -- no serious free competitor exists. Use Recharts 3.x for dashboard analytics charts only.

### Frontend Tooling

| Tool | Purpose | Version |
|------|---------|---------|
| pnpm | Package manager (faster than npm, better disk efficiency) | 10.x |
| Biome | Linter + formatter (replaces ESLint + Prettier) | 2.4.x |

> **Why pnpm?** Used by Next.js, Vue, Vite, Nuxt, Prisma, and Material UI. 33K+ GitHub stars. Faster installs and dramatically better disk space efficiency than npm. Built-in monorepo workspace support.
>
> **Why Biome over ESLint + Prettier?** ~35x faster than Prettier, 451 linting rules, zero configuration needed. Used by AWS, Google, Microsoft, Discord, Vercel. Single tool replaces both ESLint and Prettier. Won JS OS Award 2024.

### Database

| Technology | Purpose | Version |
|-----------|---------|---------|
| PostgreSQL | Primary relational database | 18.x |
| TimescaleDB extension | Time-series hypertables for stock price data | 2.25.x |
| Redis | Caching, sessions, real-time pub/sub, Celery broker | 8.6.x |

### Infrastructure

| Technology | Purpose | Version |
|-----------|---------|---------|
| WSL2 (Ubuntu) | Primary dev environment on Windows 11 | Latest |
| NVIDIA CUDA Toolkit | GPU-accelerated ML training inside WSL2 | 12.4+ |
| NVIDIA Container Toolkit | Docker GPU passthrough | Latest |
| Docker Engine | Containerization | 29.x |
| Docker Compose (v2 plugin) | Multi-container orchestration | 5.1.x |
| Nginx | Reverse proxy (stable) | 1.28.x |
| Node.js LTS | JavaScript runtime | 24.x |
| GitHub Actions | CI/CD pipelines | Free |
| Vercel | Frontend deployment (free) | Hobby tier |
| Render.com | Backend deployment (free tier) | Free tier |
| Supabase or Neon | Managed PostgreSQL (free tier) | Free tier |
| Upstash | Managed Redis (free tier) | Free tier |

### Free Tier Limits (Current as of Feb 2026)

| Service | Key Limits |
|---------|-----------|
| **Vercel (Hobby)** | 1M serverless invocations/mo, 100GB CDN bandwidth, 1 GB blob storage, 1 developer seat |
| **Render (Free)** | 0.1 CPU, 512MB RAM web service; free Postgres **expires after 30 days** |
| **Supabase (Free)** | 500MB DB, 1GB storage, 5GB bandwidth, 50K auth MAU, 2 projects max |
| **Neon (Free)** | 0.5GB storage, 100 CU-hrs/month, 10 branches, auto scale-to-zero after 5 min |
| **Upstash Redis (Free)** | 256MB data, 500K commands/month, 10GB bandwidth, 1 database |

> **Note:** Railway no longer has a true free tier -- it converts to $1/month minimum after a 30-day trial. Render's free PostgreSQL expires after 30 days. For persistent free DB, use **Neon** or **Supabase**.

---

## 3. Project Structure

```
StockSage-AI/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app initialization, middleware, CORS
│   │   ├── config.py                 # Pydantic Settings (env vars)
│   │   ├── database.py               # SQLAlchemy engine, session, Base
│   │   ├── dependencies.py           # Shared dependencies (get_db, get_current_user)
│   │   │
│   │   ├── models/                   # SQLAlchemy ORM Models
│   │   │   ├── __init__.py
│   │   │   ├── user.py               # User model
│   │   │   ├── portfolio.py          # Portfolio, Holdings, Transactions
│   │   │   ├── order.py              # Orders (buy/sell)
│   │   │   ├── stock.py              # Stock master data
│   │   │   ├── prediction.py         # ML prediction logs
│   │   │   └── leaderboard.py        # Leaderboard cache
│   │   │
│   │   ├── schemas/                  # Pydantic Request/Response Schemas
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── portfolio.py
│   │   │   ├── order.py
│   │   │   ├── stock.py
│   │   │   └── prediction.py
│   │   │
│   │   ├── routers/                  # API Route Handlers
│   │   │   ├── __init__.py
│   │   │   ├── auth.py               # POST /auth/register, /auth/login, /auth/refresh
│   │   │   ├── users.py              # GET /users/me, PUT /users/me
│   │   │   ├── stocks.py             # GET /stocks, /stocks/{symbol}, /stocks/{symbol}/history
│   │   │   ├── predictions.py        # GET /stocks/{symbol}/predict
│   │   │   ├── trading.py            # POST /orders, GET /orders, DELETE /orders/{id}
│   │   │   ├── portfolio.py          # GET /portfolio, /portfolio/holdings, /portfolio/history
│   │   │   ├── leaderboard.py        # GET /leaderboard?period=weekly|monthly|alltime
│   │   │   └── health.py             # GET /health
│   │   │
│   │   ├── services/                 # Business Logic Layer
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py       # Registration, login, token management
│   │   │   ├── stock_service.py      # Stock data fetching, caching
│   │   │   ├── trading_service.py    # Order execution, validation, settlement
│   │   │   ├── portfolio_service.py  # Portfolio calculations, P&L
│   │   │   ├── leaderboard_service.py # Ranking calculations
│   │   │   └── ml_service.py         # ML model inference wrapper
│   │   │
│   │   └── utils/                    # Utilities
│   │       ├── __init__.py
│   │       ├── auth.py               # JWT creation/verification, password hashing
│   │       ├── market_hours.py       # NSE/BSE market hours & holiday calendar
│   │       └── exceptions.py         # Custom exception classes
│   │
│   ├── ml/                           # ML Pipeline (separate from web app)
│   │   ├── __init__.py
│   │   ├── config.py                 # ML hyperparameters, model paths
│   │   │
│   │   ├── data/                     # Data Loading & Processing
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py            # Fetch data from yfinance, jugaad-data, news RSS
│   │   │   ├── preprocessor.py       # Clean, normalize, handle missing data
│   │   │   └── store.py              # Save/load from database or CSV
│   │   │
│   │   ├── features/                 # Feature Engineering
│   │   │   ├── __init__.py
│   │   │   ├── technical.py          # RSI, MACD, Bollinger Bands, SMA/EMA, ADX, OBV
│   │   │   ├── sentiment.py          # News sentiment scores (FinBERT)
│   │   │   └── builder.py            # Combine all features into training matrix
│   │   │
│   │   ├── models/                   # Model Definitions
│   │   │   ├── __init__.py
│   │   │   ├── baseline.py           # Random Forest / Logistic Regression
│   │   │   ├── xgboost_model.py      # XGBoost classifier
│   │   │   ├── lstm_model.py         # PyTorch LSTM for time-series
│   │   │   └── ensemble.py           # Ensemble of multiple models
│   │   │
│   │   ├── training/                 # Training Pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py            # Train model, cross-validation
│   │   │   ├── evaluator.py          # Accuracy, precision, recall, F1, confusion matrix
│   │   │   └── train.py              # CLI entry point: python -m ml.training.train
│   │   │
│   │   ├── inference/                # Prediction Service
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py          # Load model, run inference
│   │   │   └── cache.py              # Cache predictions in Redis
│   │   │
│   │   └── artifacts/                # Saved Model Files (gitignored)
│   │       ├── .gitkeep
│   │       ├── xgboost_v1.pkl
│   │       └── scaler_v1.pkl
│   │
│   ├── tasks/                        # Celery Background Tasks
│   │   ├── __init__.py
│   │   ├── celery_app.py             # Celery configuration
│   │   ├── market_data.py            # Scheduled: fetch live prices during market hours
│   │   ├── historical_backfill.py    # One-time: backfill 5 years of data
│   │   ├── news_fetcher.py           # Scheduled: fetch news & compute sentiment
│   │   ├── leaderboard_update.py     # Scheduled: recalculate rankings
│   │   └── model_retrain.py          # Scheduled: weekly model retraining
│   │
│   ├── migrations/                   # Alembic Database Migrations
│   │   ├── alembic.ini
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       └── .gitkeep
│   │
│   ├── tests/                        # Backend Tests
│   │   ├── __init__.py
│   │   ├── conftest.py               # Pytest fixtures (test DB, test client)
│   │   ├── unit/
│   │   │   ├── test_trading_service.py
│   │   │   ├── test_portfolio_service.py
│   │   │   └── test_market_hours.py
│   │   └── integration/
│   │       ├── test_auth_api.py
│   │       ├── test_trading_api.py
│   │       └── test_stock_api.py
│   │
│   ├── pyproject.toml                # Python project config (dependencies, ruff, pytest)
│   ├── requirements.txt              # Pinned dependencies (generated by uv pip compile)
│   ├── requirements-ml.txt           # ML-specific dependencies (PyTorch CUDA, XGBoost, etc.)
│   ├── Dockerfile                    # Backend API container
│   ├── Dockerfile.ml                 # ML training container (with CUDA support)
│   └── .env.example
│
├── frontend/                         # Next.js Frontend
│   ├── app/
│   │   ├── layout.tsx                # Root layout (providers, global styles)
│   │   ├── page.tsx                  # Landing page (/)
│   │   ├── error.tsx                 # Global error boundary
│   │   ├── not-found.tsx             # 404 page
│   │   │
│   │   ├── (auth)/                   # Auth route group (no layout chrome)
│   │   │   ├── layout.tsx
│   │   │   ├── login/
│   │   │   │   └── page.tsx          # /login
│   │   │   └── signup/
│   │   │       └── page.tsx          # /signup
│   │   │
│   │   └── (dashboard)/              # Dashboard route group (sidebar + header)
│   │       ├── layout.tsx            # Dashboard layout with sidebar navigation
│   │       ├── dashboard/
│   │       │   └── page.tsx          # /dashboard (portfolio overview + market summary)
│   │       ├── stocks/
│   │       │   ├── page.tsx          # /stocks (stock screener / search)
│   │       │   └── [symbol]/
│   │       │       └── page.tsx      # /stocks/RELIANCE (stock detail + prediction + order)
│   │       ├── portfolio/
│   │       │   └── page.tsx          # /portfolio (holdings, P&L, transaction history)
│   │       ├── leaderboard/
│   │       │   └── page.tsx          # /leaderboard (rankings table)
│   │       ├── ai-trader/
│   │       │   └── page.tsx          # /ai-trader (AI performance + comparison)
│   │       └── settings/
│   │           └── page.tsx          # /settings (user profile, preferences)
│   │
│   ├── components/
│   │   ├── ui/                       # Base UI components (shadcn/ui style)
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── table.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── skeleton.tsx
│   │   │   └── toast.tsx
│   │   │
│   │   ├── layout/                   # Layout components
│   │   │   ├── header.tsx            # Top navigation bar
│   │   │   ├── sidebar.tsx           # Dashboard sidebar
│   │   │   ├── footer.tsx
│   │   │   └── mobile-nav.tsx        # Mobile responsive navigation
│   │   │
│   │   ├── charts/                   # Chart components
│   │   │   ├── stock-chart.tsx       # TradingView Lightweight Charts wrapper
│   │   │   ├── portfolio-chart.tsx   # Portfolio value over time (Recharts)
│   │   │   └── prediction-gauge.tsx  # AI prediction confidence meter
│   │   │
│   │   ├── trading/                  # Trading components
│   │   │   ├── order-form.tsx        # Buy/Sell order form
│   │   │   ├── order-book.tsx        # Open orders list
│   │   │   └── trade-history.tsx     # Executed trades
│   │   │
│   │   ├── portfolio/                # Portfolio components
│   │   │   ├── holdings-table.tsx    # Current holdings with P&L
│   │   │   ├── portfolio-summary.tsx # Total value, returns, cash balance
│   │   │   └── transaction-list.tsx  # Transaction history
│   │   │
│   │   └── leaderboard/             # Leaderboard components
│   │       ├── rankings-table.tsx    # User rankings
│   │       ├── period-selector.tsx   # Weekly/Monthly/All-time toggle
│   │       └── ai-comparison.tsx     # AI vs User performance card
│   │
│   ├── lib/                          # Utility functions
│   │   ├── api.ts                    # Fetch wrapper for backend API calls (native fetch, no axios)
│   │   ├── auth.ts                   # Auth helpers (token storage, refresh)
│   │   ├── utils.ts                  # Format currency (Rs), dates, percentages
│   │   └── constants.ts             # API URLs, market hours, etc.
│   │
│   ├── hooks/                        # Custom React hooks
│   │   ├── use-auth.ts              # Authentication state hook
│   │   ├── use-stock.ts             # Stock data fetching hook
│   │   ├── use-portfolio.ts         # Portfolio data hook
│   │   ├── use-websocket.ts         # WebSocket connection for live prices
│   │   └── use-prediction.ts        # AI prediction fetching hook
│   │
│   ├── stores/                       # Zustand state stores
│   │   ├── auth-store.ts            # Auth state (user, tokens)
│   │   └── market-store.ts          # Market data state (live prices)
│   │
│   ├── types/                        # TypeScript types
│   │   ├── api.ts                   # API response types
│   │   ├── stock.ts                 # Stock, OHLCV types
│   │   ├── portfolio.ts             # Portfolio, Holding, Transaction types
│   │   ├── order.ts                 # Order types
│   │   └── user.ts                  # User types
│   │
│   ├── styles/
│   │   └── globals.css              # TailwindCSS v4 imports (CSS-first config) + custom globals
│   │
│   ├── public/
│   │   └── images/
│   │
│   ├── middleware.ts                 # Auth.js middleware (protect dashboard routes)
│   ├── next.config.ts               # Next.js 16 config (TypeScript native)
│   ├── tsconfig.json
│   ├── biome.json                   # Biome linter + formatter config
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml                # Local dev: API + Frontend + PostgreSQL + Redis
├── docker-compose.prod.yml           # Production overrides
├── .env.example                      # Environment variables template
├── .gitignore
├── LICENSE
├── PLAN.md                           # This file
└── README.md
```

---

## 4. Data Sources & APIs (All Free)

### Market Data (Stock Prices)

| Source | Library | Version | Use Case | Rate Limits | API Key |
|--------|---------|---------|----------|-------------|---------|
| **Yahoo Finance** | `yfinance` | 1.2.0 | Historical OHLCV data (5+ years), daily/weekly | Unofficial, may throttle | None |
| **NSE India** | `jugaad-data` | 0.29 | Live NSE quotes, index data, F&O data | Built-in caching | None |
| **NSE India** | `nsetools` | 2.0.1 | Real-time stock quotes, batch quotes | Moderate | None |
| **NSE India** | `nselib` | 2.4.3 | NSE capital markets, derivatives, indices (backup) | Moderate | None |
| **BSE India** | `yfinance` (.BO suffix) | 1.2.0 | BSE-listed stock data | Same as yfinance | None |

**Strategy:**
- **Historical data backfill**: `yfinance` — download 5 years of daily OHLCV for all NSE stocks
- **Live quotes during market hours**: `jugaad-data` NSELive + `nsetools` as fallback
- **End-of-day updates**: `jugaad-data` bhavcopy (official NSE daily reports)

**Example usage:**
```python
# Historical data (yfinance)
import yfinance as yf
df = yf.download("RELIANCE.NS", period="5y", interval="1d")

# Live quote (jugaad-data)
from jugaad_data.nse import NSELive
nse = NSELive()
quote = nse.stock_quote("RELIANCE")

# Real-time (nsetools)
from nsetools import Nse
nse = Nse()
q = nse.get_quote("reliance")
```

### News & Sentiment Data

| Source | Method | Use Case | API Key |
|--------|--------|----------|---------|
| **Google News RSS** | HTTP GET + XML parse | Primary news feed per stock | None |
| **NewsAPI.org** | REST API | Supplementary headlines (dev only) | Free key (100 req/day) |
| **MoneyControl RSS** | HTTP GET + XML parse | India-specific financial news | None |

**Google News RSS URL pattern:**
```
https://news.google.com/rss/search?q={STOCK_NAME}+stock+NSE+india&hl=en-IN&gl=IN&ceid=IN:en
```

**Sentiment Analysis:**
- Use Hugging Face `ProsusAI/finbert` model (pre-trained on financial text)
- Input: news headline text
- Output: sentiment score (-1 to +1) per headline
- Aggregate: daily average sentiment per stock

### Stock Master Data

| Data | Source |
|------|--------|
| List of all NSE stocks | `nsetools.get_stock_codes()` |
| Stock sector/industry | NSE website CSV downloads |
| NSE holiday calendar | NSE website (published yearly) |

---

## 5. Database Schema

### Entity Relationship

```
users 1──N portfolios 1──N holdings
                      1──N orders 1──1 transactions
                      1──N portfolio_snapshots

stocks 1──N stock_prices (TimescaleDB hypertable)
       1──N predictions
       1──N news_articles

users 1──1 leaderboard_entries
ai_trader 1──1 portfolios
```

### Table Definitions

#### users
```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    username        VARCHAR(50) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    full_name       VARCHAR(100),
    avatar_url      VARCHAR(500),
    is_active       BOOLEAN DEFAULT TRUE,
    is_ai           BOOLEAN DEFAULT FALSE,          -- TRUE for AI trader account
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### portfolios
```sql
CREATE TABLE portfolios (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            VARCHAR(100) DEFAULT 'Default',
    cash_balance    DECIMAL(15,2) NOT NULL DEFAULT 100000.00,  -- Rs 1,00,000
    initial_capital DECIMAL(15,2) NOT NULL DEFAULT 100000.00,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### holdings
```sql
CREATE TABLE holdings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id    UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,             -- e.g., "RELIANCE"
    exchange        VARCHAR(5) NOT NULL DEFAULT 'NSE', -- NSE or BSE
    quantity        INTEGER NOT NULL,
    avg_buy_price   DECIMAL(12,2) NOT NULL,
    current_price   DECIMAL(12,2),                    -- cached, updated periodically
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(portfolio_id, symbol, exchange)
);
```

#### orders
```sql
CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id    UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(5) NOT NULL DEFAULT 'NSE',
    order_type      VARCHAR(10) NOT NULL,             -- 'MARKET' or 'LIMIT'
    side            VARCHAR(4) NOT NULL,              -- 'BUY' or 'SELL'
    quantity        INTEGER NOT NULL,
    price           DECIMAL(12,2),                    -- NULL for market orders
    status          VARCHAR(15) NOT NULL DEFAULT 'PENDING',
                                                      -- PENDING, EXECUTED, CANCELLED, REJECTED
    executed_price  DECIMAL(12,2),                    -- actual execution price
    executed_at     TIMESTAMPTZ,
    rejection_reason VARCHAR(255),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### transactions
```sql
CREATE TABLE transactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id    UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    order_id        UUID REFERENCES orders(id),
    type            VARCHAR(10) NOT NULL,             -- 'BUY', 'SELL', 'DIVIDEND', 'CREDIT'
    symbol          VARCHAR(20),
    quantity        INTEGER,
    price           DECIMAL(12,2),
    total_amount    DECIMAL(15,2) NOT NULL,           -- quantity * price (+ fees if any)
    balance_after   DECIMAL(15,2) NOT NULL,           -- cash balance after transaction
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### stocks
```sql
CREATE TABLE stocks (
    symbol          VARCHAR(20) PRIMARY KEY,          -- e.g., "RELIANCE"
    name            VARCHAR(200) NOT NULL,            -- "Reliance Industries Limited"
    exchange        VARCHAR(5) NOT NULL DEFAULT 'NSE',
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    market_cap      BIGINT,                           -- in Rs
    is_active       BOOLEAN DEFAULT TRUE,
    last_updated    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### stock_prices (TimescaleDB Hypertable)
```sql
CREATE TABLE stock_prices (
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    open            DECIMAL(12,2),
    high            DECIMAL(12,2),
    low             DECIMAL(12,2),
    close           DECIMAL(12,2),
    volume          BIGINT,
    adj_close       DECIMAL(12,2)
);

-- Convert to TimescaleDB hypertable for fast time-series queries
SELECT create_hypertable('stock_prices', 'timestamp');
CREATE INDEX idx_stock_prices_symbol ON stock_prices (symbol, timestamp DESC);
```

#### predictions
```sql
CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(20) NOT NULL,
    model_name      VARCHAR(50) NOT NULL,             -- 'xgboost_v1', 'lstm_v2', 'ensemble'
    model_version   VARCHAR(20) NOT NULL,
    prediction      VARCHAR(10) NOT NULL,             -- 'UP', 'DOWN', 'NEUTRAL'
    confidence      DECIMAL(5,4) NOT NULL,            -- 0.0000 to 1.0000
    target_period   VARCHAR(10) NOT NULL,             -- '1D', '5D', '1W'
    features_used   JSONB,                            -- snapshot of input features
    actual_result   VARCHAR(10),                      -- filled after target period ends
    was_correct     BOOLEAN,                          -- filled after verification
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### news_articles
```sql
CREATE TABLE news_articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    source          VARCHAR(100),
    url             VARCHAR(500),
    published_at    TIMESTAMPTZ,
    sentiment_score DECIMAL(5,4),                     -- -1.0000 to 1.0000
    sentiment_label VARCHAR(10),                      -- 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    symbols         VARCHAR(20)[],                    -- related stock symbols
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### leaderboard_entries
```sql
CREATE TABLE leaderboard_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    portfolio_id    UUID NOT NULL REFERENCES portfolios(id),
    period          VARCHAR(10) NOT NULL,             -- 'WEEKLY', 'MONTHLY', 'ALLTIME'
    rank            INTEGER NOT NULL,
    total_value     DECIMAL(15,2) NOT NULL,           -- cash + holdings value
    total_return    DECIMAL(10,4) NOT NULL,           -- percentage return
    num_trades      INTEGER DEFAULT 0,
    win_rate        DECIMAL(5,4),                     -- percentage of profitable trades
    calculated_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, period, calculated_at::DATE)
);
```

#### portfolio_snapshots (Daily portfolio value tracking)
```sql
CREATE TABLE portfolio_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id    UUID NOT NULL REFERENCES portfolios(id),
    date            DATE NOT NULL,
    total_value     DECIMAL(15,2) NOT NULL,           -- cash + holdings market value
    cash_balance    DECIMAL(15,2) NOT NULL,
    holdings_value  DECIMAL(15,2) NOT NULL,
    daily_pnl       DECIMAL(15,2),                    -- day-over-day change
    total_return    DECIMAL(10,4),                    -- % return from initial capital
    UNIQUE(portfolio_id, date)
);
```

---

## 6. ML Pipeline Design

### Overview

```
[Data Collection] → [Feature Engineering] → [Model Training (GPU)] → [Evaluation] → [Serving]
     ↓                       ↓                      ↓                     ↓              ↓
  yfinance            technical indicators       XGBoost (CUDA)       accuracy       FastAPI
  jugaad-data         sentiment scores           LSTM (CUDA)          precision      Redis cache
  Google News RSS     lag features               ensemble             backtesting    /predict API
                                                    │
                                              WSL2 + NVIDIA GPU
```

### 6.1 Feature Engineering

**Technical Indicators** (calculated using pandas-ta):

| Category | Indicators |
|----------|-----------|
| Trend | SMA(20, 50, 200), EMA(12, 26), ADX, MACD |
| Momentum | RSI(14), Stochastic Oscillator, Williams %R, CCI |
| Volatility | Bollinger Bands(20), ATR(14), Standard Deviation |
| Volume | OBV, VWAP, Volume SMA(20), CMF |

**Derived Features:**

| Feature | Description |
|---------|-------------|
| `price_change_1d` to `_5d` | Percentage price change over 1-5 days |
| `volume_change_1d` | Volume change vs previous day |
| `high_low_ratio` | Day's high/low range as % of close |
| `gap_open` | Opening gap vs previous close |
| `dist_from_52w_high` | Distance from 52-week high |
| `dist_from_52w_low` | Distance from 52-week low |
| `sector_avg_return` | Same-sector average return (market context) |
| `nifty50_return_1d` | Nifty 50 index daily return (market trend) |
| `sentiment_score_avg` | Average news sentiment (past 3 days) |
| `sentiment_score_count` | Number of news articles (attention proxy) |

**Target Variable:**
```python
# Binary classification: will price go UP or DOWN tomorrow?
target = (next_day_close > today_close).astype(int)  # 1 = UP, 0 = DOWN
```

### 6.2 Models

**Baseline (Phase 1):**
- Random Forest Classifier
- Logistic Regression with L2 regularization
- Purpose: establish baseline accuracy

**Primary Model (Phase 2):**
- XGBoost Classifier (best for tabular financial data)
- Hyperparameter tuning via Optuna
- Expected accuracy: 53-58% (above random = useful)

**Advanced Model (Phase 3) -- GPU Required:**
- LSTM network for sequential time-series patterns (trained on GPU via CUDA)
- Input: sliding window of 30-60 days of features
- Architecture: 2-layer LSTM → Dropout → Dense → Sigmoid
- Training: ~10-50x faster on GPU vs CPU for sequence models

**Ensemble (Phase 4):**
- Weighted voting of XGBoost + LSTM + Sentiment model
- Weights determined by validation performance

### 6.3 Training Pipeline

```python
# Pseudocode for training pipeline (runs in WSL2 with GPU)
import torch

def train_pipeline():
    # 0. Verify GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")  # Should print "cuda"

    # 1. Load data
    prices = load_stock_prices(period="5y")          # from DB
    news = load_news_sentiment(period="2y")           # from DB

    # 2. Feature engineering
    features = compute_technical_indicators(prices)   # pandas-ta
    features = add_sentiment_features(features, news) # merge
    features = add_lag_features(features, lags=[1,2,3,5])

    # 3. Train/test split (TIME-BASED, not random!)
    train = features[features.date < "2025-01-01"]
    test  = features[features.date >= "2025-01-01"]

    # 4. Train XGBoost model (GPU-accelerated)
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.01,
        device="cuda",                                # GPU training
        tree_method="hist"                            # GPU-optimized histogram method
    )
    model.fit(train[feature_cols], train["target"])

    # 5. Evaluate
    predictions = model.predict(test[feature_cols])
    accuracy = accuracy_score(test["target"], predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # 6. Save
    joblib.dump(model, "ml/artifacts/xgboost_v1.pkl")
    mlflow.log_metric("accuracy", accuracy)
```

> **Critical**: Always use time-based splits, never random splits. Random splits cause data leakage in time-series data.
>
> **GPU Notes:**
> - XGBoost 3.x: Use `device="cuda"` parameter (replaces old `gpu_hist` tree method)
> - LightGBM 4.x: Use `device="gpu"` parameter
> - PyTorch LSTM: Move model and tensors to GPU with `.to("cuda")`
> - FinBERT inference: `model.to("cuda")` for batch sentiment scoring (~5x faster)
> - All training runs inside WSL2 where NVIDIA drivers have native CUDA support

### 6.4 Model Serving

```python
# FastAPI endpoint
@router.get("/stocks/{symbol}/predict")
async def predict_stock(
    symbol: str,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis)
):
    # Check cache first (predictions valid for 1 hour during market hours)
    cached = await redis.get(f"prediction:{symbol}")
    if cached:
        return json.loads(cached)

    # Get latest features
    features = build_features_for_symbol(symbol, db)

    # Run inference
    prediction = predictor.predict(features)

    # Cache result
    await redis.set(f"prediction:{symbol}", json.dumps(prediction), ex=3600)

    # Log to DB
    save_prediction_log(symbol, prediction, db)

    return prediction
```

---

## 7. Backend API Design

### API Endpoints

#### Authentication
```
POST   /api/auth/register          Register new user
POST   /api/auth/login             Login (returns JWT access + refresh tokens)
POST   /api/auth/refresh           Refresh access token
POST   /api/auth/logout            Invalidate refresh token
```

#### User
```
GET    /api/users/me               Get current user profile
PUT    /api/users/me               Update profile (name, avatar)
GET    /api/users/{id}/stats       Get user trading stats (public)
```

#### Stocks
```
GET    /api/stocks                 List all stocks (paginated, searchable)
GET    /api/stocks/search?q=rel    Search stocks by name/symbol
GET    /api/stocks/{symbol}        Get stock details + latest quote
GET    /api/stocks/{symbol}/history?period=1y&interval=1d
                                   Get historical OHLCV data
GET    /api/stocks/{symbol}/predict
                                   Get AI prediction for stock
GET    /api/stocks/top-gainers     Today's top gainers
GET    /api/stocks/top-losers      Today's top losers
GET    /api/stocks/indices         NIFTY 50, SENSEX, BANKNIFTY values
```

#### Trading
```
POST   /api/orders                 Place a new order (buy/sell)
GET    /api/orders                 Get user's orders (filter by status)
GET    /api/orders/{id}            Get specific order details
DELETE /api/orders/{id}            Cancel a pending order
```

#### Portfolio
```
GET    /api/portfolio              Get portfolio summary (cash + holdings value)
GET    /api/portfolio/holdings     Get current holdings with live P&L
GET    /api/portfolio/transactions Get transaction history (paginated)
GET    /api/portfolio/performance?period=1m
                                   Get portfolio performance chart data
GET    /api/portfolio/analytics    Get stats: win rate, avg return, Sharpe ratio
```

#### Leaderboard
```
GET    /api/leaderboard?period=weekly&page=1
                                   Get rankings (weekly/monthly/alltime)
GET    /api/leaderboard/me         Get current user's rank
GET    /api/leaderboard/ai         Get AI trader's rank and performance
```

#### System
```
GET    /api/health                 Health check
GET    /api/market/status          Market open/closed status
```

### Request/Response Examples

**Place Order:**
```json
// POST /api/orders
// Request:
{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "side": "BUY",
    "order_type": "MARKET",
    "quantity": 10
}

// Response:
{
    "id": "uuid",
    "symbol": "RELIANCE",
    "side": "BUY",
    "order_type": "MARKET",
    "quantity": 10,
    "status": "EXECUTED",
    "executed_price": 2456.75,
    "total_amount": 24567.50,
    "cash_balance_after": 75432.50,
    "executed_at": "2026-02-26T10:30:00+05:30"
}
```

**Get Prediction:**
```json
// GET /api/stocks/RELIANCE/predict
// Response:
{
    "symbol": "RELIANCE",
    "prediction": "UP",
    "confidence": 0.6234,
    "target_period": "1D",
    "model": "ensemble_v2",
    "factors": {
        "technical_signal": "BULLISH",
        "sentiment_score": 0.42,
        "trend": "UPTREND",
        "rsi": 55.3
    },
    "generated_at": "2026-02-26T09:30:00+05:30",
    "disclaimer": "This is not financial advice. For educational purposes only."
}
```

---

## 8. Frontend Pages & Components

### Page Map

```
/                       Landing page (hero, features, CTA)
/login                  Login form
/signup                 Registration form
/dashboard              Portfolio overview + market summary (protected)
/stocks                 Stock screener with search
/stocks/[symbol]        Stock detail (chart + prediction + order form)
/portfolio              Holdings table + P&L + transaction history
/leaderboard            Rankings table with period tabs
/ai-trader              AI trader performance + comparison
/settings               User profile settings
```

### Key Component Specifications

#### Stock Chart (`stock-chart.tsx`)
- Use TradingView Lightweight Charts library
- Candlestick chart by default, line chart option
- Time range selector: 1D, 1W, 1M, 3M, 6M, 1Y, 5Y
- Volume bars below price chart
- Overlay: SMA(20), SMA(50) optional toggle
- Prediction indicator: colored arrow (green UP / red DOWN) with confidence %

#### Order Form (`order-form.tsx`)
- Side toggle: BUY (green) / SELL (red)
- Order type: MARKET / LIMIT dropdown
- Quantity input (number)
- Price input (only for LIMIT orders)
- Estimated total display
- Available cash balance display
- Available quantity (for SELL)
- Submit button with confirmation dialog

#### Holdings Table (`holdings-table.tsx`)
- Columns: Stock, Qty, Avg Price, Current Price, P&L (Rs), P&L (%), Value
- Color-coded P&L (green = profit, red = loss)
- Click row to go to stock detail page
- Sort by any column
- Total row at bottom

#### Leaderboard (`rankings-table.tsx`)
- Period tabs: Weekly | Monthly | All-Time
- Columns: Rank, User, Portfolio Value, Return %, Trades, Win Rate
- Current user's row highlighted
- AI trader row with special badge
- Pagination (50 per page)

### Color Scheme & Design Tokens

```
Primary:        #1E40AF (Blue 800)     -- brand, CTAs
Success/Profit: #16A34A (Green 600)    -- positive P&L, UP prediction
Danger/Loss:    #DC2626 (Red 600)      -- negative P&L, DOWN prediction
Warning:        #D97706 (Amber 600)    -- pending orders, alerts
Background:     #0F172A (Slate 900)    -- dark mode primary
Surface:        #1E293B (Slate 800)    -- cards, panels
Text Primary:   #F8FAFC (Slate 50)     -- headings
Text Secondary: #94A3B8 (Slate 400)    -- body text
```

> Dark mode by default (standard for trading platforms). Light mode as toggle.

---

## 9. Paper Trading Engine

### Order Execution Flow

```
User places order
    ↓
Validate order
  ├── Market open? (9:15 AM - 3:30 PM IST, Mon-Fri, not holiday)
  ├── Stock valid & active?
  ├── Sufficient cash (BUY) or holdings (SELL)?
  ├── Quantity > 0?
  └── Price within circuit limits? (for LIMIT orders)
    ↓
[If MARKET order]
  ├── Get current market price from live feed
  ├── Execute immediately at market price
  └── Update portfolio (deduct cash / add holdings)
    ↓
[If LIMIT order]
  ├── Add to pending orders queue
  ├── Check periodically during market hours
  ├── Execute when market price crosses limit price
  └── Cancel at end of day if not executed (GTC option later)
    ↓
Record transaction
    ↓
Update leaderboard cache (async)
```

### Execution Rules

| Rule | Detail |
|------|--------|
| Market orders | Execute at current LTP (Last Traded Price) |
| Limit orders | Execute when LTP crosses limit price |
| Short selling | NOT allowed (paper trading keeps it simple) |
| Order validity | Day orders expire at 3:30 PM IST |
| Minimum qty | 1 share |
| Max order value | Cannot exceed available cash (BUY) or holdings (SELL) |
| Circuit limits | Reject LIMIT orders outside stock's daily circuit range |
| After-hours | Orders placed after hours queued for next market open |

### Portfolio Calculations

```python
# Total portfolio value
total_value = cash_balance + sum(holding.quantity * holding.current_price for holding in holdings)

# Total return
total_return_pct = ((total_value - initial_capital) / initial_capital) * 100

# Per-holding P&L
holding_pnl = (current_price - avg_buy_price) * quantity
holding_pnl_pct = ((current_price - avg_buy_price) / avg_buy_price) * 100

# Day's P&L
day_pnl = total_value_today - total_value_yesterday

# Win rate
win_rate = profitable_trades / total_trades * 100
```

---

## 10. Leaderboard & AI Trader

### Leaderboard System

**Ranking criteria:** Total portfolio return percentage (higher = better rank)

```python
# Ranking formula
rank_score = ((total_value - initial_capital) / initial_capital) * 100

# Tiebreaker: fewer trades = higher rank (efficiency)
```

**Periods:**
- **Weekly**: Reset every Monday 9:15 AM IST
- **Monthly**: Reset 1st of every month
- **All-time**: Since account creation

**Update frequency:**
- During market hours: every 5 minutes (Celery scheduled task)
- After market close: final calculation at 4:00 PM IST

### AI Trader Agent

**How it works:**
1. AI trader is a special user account (`is_ai = TRUE`) with its own portfolio
2. Starts with same Rs 1,00,000 capital as every user
3. Runs as a Celery scheduled task during market hours

**AI Trading Strategy:**
```python
# Simplified AI trader logic
def ai_make_decision(symbol, prediction, portfolio):
    if prediction.direction == "UP" and prediction.confidence > 0.60:
        # Buy signal
        if symbol not in portfolio.holdings:
            # Allocate max 10% of portfolio to one stock
            max_allocation = portfolio.total_value * 0.10
            quantity = int(max_allocation / current_price)
            place_buy_order(symbol, quantity)

    elif prediction.direction == "DOWN" and prediction.confidence > 0.55:
        # Sell signal
        if symbol in portfolio.holdings:
            place_sell_order(symbol, portfolio.holdings[symbol].quantity)

# Run every 30 minutes during market hours for top 50 stocks
```

**AI Risk Management:**
- Max 10% of portfolio in any single stock
- Max 15 holdings at any time
- Stop-loss at -5% per holding
- Minimum confidence threshold: 60% for buy, 55% for sell
- Only trades NIFTY 50 stocks (most liquid)

---

## 11. Authentication & Security

### Authentication Flow

```
[Register] → Hash password (Argon2 via pwdlib) → Store in DB → Return JWT pair
[Login]    → Verify password → Generate JWT access token (15 min) + refresh token (7 days)
[API Call] → Extract JWT from Authorization header → Verify (PyJWT) → Get user from DB
[Refresh]  → Verify refresh token → Generate new access token
[Logout]   → Blacklist refresh token in Redis
```

### JWT Token Structure

```python
# Access Token (15 min expiry)
{
    "sub": "user-uuid",
    "email": "user@example.com",
    "exp": 1234567890,
    "type": "access"
}

# Refresh Token (7 day expiry)
{
    "sub": "user-uuid",
    "exp": 1234567890,
    "type": "refresh",
    "jti": "unique-token-id"   # for blacklisting
}
```

### Security Measures

| Measure | Implementation |
|---------|---------------|
| Password hashing | Argon2 via pwdlib 0.3.x (more secure than bcrypt, modern standard) |
| JWT tokens | PyJWT 2.11.x with HS256 (officially recommended by FastAPI) |
| Rate limiting | slowapi 0.1.9 (per IP, per user) |
| CORS | FastAPI CORSMiddleware (whitelist frontend origin) |
| Input validation | Pydantic v2 2.12.x schemas on all endpoints |
| SQL injection | SQLAlchemy 2.0 ORM (parameterized queries) |
| XSS | Next.js 16 built-in escaping + CSP headers |
| HTTPS | Enforced in production (Vercel + Render handle this) |
| Secrets | Environment variables via Pydantic Settings, never in code |
| Token blacklist | Redis 8.6.x set for revoked refresh tokens |

---

## 12. Infrastructure & DevOps

### Development Environment: WSL2 + GPU

All development runs inside **WSL2 (Windows Subsystem for Linux)** on Windows 11. This gives us:
- Native Linux environment for Python, Docker, and CUDA
- GPU passthrough for ML training (NVIDIA drivers on Windows host, CUDA inside WSL2)
- No Windows path/compatibility issues with Python packages
- Docker Desktop integrates natively with WSL2

```
┌─────────────────────────────────────────────┐
│            Windows 11 Host                   │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │              WSL2 (Ubuntu)              │ │
│  │                                         │ │
│  │  ┌──────────┐  ┌────────────────────┐  │ │
│  │  │ Docker   │  │  Python + uv       │  │ │
│  │  │ Compose  │  │  ML Training (GPU) │  │ │
│  │  │ (API,DB, │  │  PyTorch + CUDA    │  │ │
│  │  │  Redis)  │  │  XGBoost + CUDA    │  │ │
│  │  └──────────┘  └────────────────────┘  │ │
│  │                     │                   │ │
│  │              NVIDIA GPU (CUDA)          │ │
│  └─────────────────────────────────────────┘ │
│                     │                        │
│            NVIDIA Windows Driver             │
│            (GPU passthrough to WSL2)         │
└─────────────────────────────────────────────┘
```

### Local Development (Docker Compose v5)

```yaml
services:
  api:          # FastAPI backend on :8000
  frontend:     # Next.js 16 on :3000
  db:           # PostgreSQL 18 + TimescaleDB 2.25 on :5432
  redis:        # Redis 8.6 on :6379
  celery:       # Celery 5.6 worker
  celery-beat:  # Celery periodic task scheduler
  ml-trainer:   # ML training service (GPU-enabled, on-demand)
```

**ML trainer service with GPU passthrough:**
```yaml
# docker-compose.yml (GPU section)
services:
  ml-trainer:
    build:
      context: ./backend
      dockerfile: Dockerfile.ml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./backend/ml:/app/ml
      - ./backend/ml/artifacts:/app/ml/artifacts
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      CUDA_VISIBLE_DEVICES: "0"
    command: python -m ml.training.train
    profiles: ["training"]    # Only starts when explicitly requested
```

**Commands:**
```bash
# Start core services (API, frontend, DB, Redis)
docker compose up -d

# Run database migrations
docker compose exec api alembic upgrade head

# Run ML training with GPU (on-demand, not always running)
docker compose --profile training run --rm ml-trainer

# Or train directly in WSL2 (without Docker, faster for iteration)
cd backend && python -m ml.training.train

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# View logs
docker compose logs -f api
```

> **Note:** Docker Compose v2 uses `docker compose` (space) not `docker-compose` (hyphen). The hyphenated version is the legacy standalone binary.
>
> **GPU Training options:** You can either train via Docker (with GPU passthrough via NVIDIA Container Toolkit) or directly in WSL2's Python environment. Direct WSL2 is faster for iteration during development. Docker is better for reproducible training runs.

### Free Deployment Stack

```
┌─────────────────────────────────────────────────────┐
│                    PRODUCTION                        │
│                                                      │
│  ┌─────────┐    ┌──────────┐    ┌────────────────┐  │
│  │ Vercel  │    │ Render   │    │ Supabase/Neon  │  │
│  │ Next.js │───→│ FastAPI  │───→│ PostgreSQL     │  │
│  │ Frontend│    │ Backend  │    │ + TimescaleDB  │  │
│  │ (FREE)  │    │ (FREE)   │    │ (FREE tier)    │  │
│  └─────────┘    └──────────┘    └────────────────┘  │
│                      │                               │
│                      ↓                               │
│                 ┌──────────┐                         │
│                 │ Upstash  │                         │
│                 │ Redis    │                         │
│                 │ (FREE)   │                         │
│                 └──────────┘                         │
│                                                      │
│  URLs:                                               │
│  Frontend: stocksage-ai.vercel.app                   │
│  Backend:  stocksage-ai.onrender.com                 │
└─────────────────────────────────────────────────────┘
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]

jobs:
  backend-test:
    - Lint + format check (Ruff 0.15.x -- replaces black + flake8 + isort)
    - Type check (mypy)
    - Unit tests (pytest)
    - Integration tests (with test DB)

  frontend-test:
    - Lint + format check (Biome 2.4.x -- replaces ESLint + Prettier)
    - Type check (tsc)
    - Unit tests (vitest)
    - Build check (next build)

  deploy-backend:      # On merge to main
    - Build Docker image
    - Deploy to Render

  deploy-frontend:     # Automatic via Vercel GitHub integration
```

---

## 13. Indian Market-Specific Rules

### Market Hours

```python
MARKET_CONFIG = {
    "exchange": "NSE",
    "timezone": "Asia/Kolkata",
    "pre_open_start": "09:00",
    "pre_open_end": "09:15",
    "market_open": "09:15",
    "market_close": "15:30",
    "post_close_end": "16:00",
    "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
}
```

### NSE Holidays 2026 (to be updated yearly)

```python
NSE_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-10",  # Maha Shivaratri
    "2026-03-17",  # Holi
    "2026-03-30",  # Id-Ul-Fitr (Ramadan Eid)
    "2026-04-02",  # Ram Navami
    "2026-04-03",  # Mahavir Jayanti
    "2026-04-14",  # Dr Ambedkar Jayanti
    "2026-04-18",  # Good Friday
    "2026-05-01",  # Maharashtra Day
    "2026-06-06",  # Id-Ul-Adha (Bakri Eid)
    "2026-07-06",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Janmashtami
    "2026-09-04",  # Milad-Un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-09",  # Diwali (Laxmi Puja)
    "2026-11-10",  # Diwali (Balipratipada)
    "2026-11-30",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]
```

### Circuit Limits

```python
# NSE circuit breaker bands
CIRCUIT_LIMITS = {
    "default": 0.20,      # 20% band for most stocks
    "fo_stocks": 0.20,    # F&O stocks have 20% band
    "new_listing": 0.20,  # First day of listing
    "sme": 0.05,          # SME stocks have 5% band
}

# Market-wide circuit breaker (NIFTY 50)
MARKET_CIRCUIT = {
    "level_1": 0.10,  # 10% fall → 45 min halt (before 1PM), 15 min (1-2:30PM)
    "level_2": 0.15,  # 15% fall → 1:45 hour halt (before 1PM), 45 min (1-2PM)
    "level_3": 0.20,  # 20% fall → trading halted for the day
}
```

### Stock Symbol Mapping

```python
# NSE symbols are plain text: RELIANCE, TCS, INFY
# BSE uses numeric codes: 500325 (Reliance), 532540 (TCS)
# yfinance suffixes: RELIANCE.NS (NSE), RELIANCE.BO (BSE)

# We will use NSE symbols as the primary identifier
# and maintain a mapping table for BSE codes and yfinance tickers
```

---

## 14. Implementation Phases

### Phase 1: Foundation (Project Setup)

**Goal:** Runnable monorepo with auth, Docker, CI/CD -- all inside WSL2

- [ ] WSL2 + CUDA setup verified (nvidia-smi works, PyTorch sees GPU)
- [ ] Initialize monorepo structure (backend/ + frontend/)
- [ ] Set up FastAPI project with config, database, health check
- [ ] Set up Next.js 16 project with TypeScript, TailwindCSS v4, shadcn/ui
- [ ] Docker Compose (PostgreSQL + Redis + API + Frontend)
- [ ] Alembic migrations setup
- [ ] User registration + login (PyJWT + pwdlib/Argon2)
- [ ] Protected API routes + Auth.js frontend middleware
- [ ] GitHub Actions CI pipeline (Ruff + Biome, test, type-check)
- [ ] `.env.example` and documentation

**Deliverable:** Users can register, login, and access a protected dashboard page. GPU verified for ML phases.

---

### Phase 2: Market Data Pipeline

**Goal:** Historical and live Indian stock data flowing into database

- [ ] Stock master data: fetch and store all NSE stock symbols + names
- [ ] Historical OHLCV backfill: 5 years of daily data for top 500 stocks (yfinance)
- [ ] TimescaleDB hypertable for stock_prices
- [ ] Live price fetcher: Celery task running during market hours (jugaad-data)
- [ ] NSE holiday calendar integration
- [ ] Market status endpoint (open/closed/holiday)
- [ ] Stock search API endpoint
- [ ] Stock detail API endpoint (latest quote + basic info)
- [ ] End-of-day data update task
- [ ] Data validation (handle missing data, outliers, corporate actions)

**Deliverable:** Stock data accessible via API. `GET /api/stocks/RELIANCE` returns live price.

---

### Phase 3: ML Prediction Engine (GPU-Accelerated)

**Goal:** Working stock prediction API with >53% accuracy

- [ ] Verify CUDA/GPU setup in WSL2 (`nvidia-smi` + PyTorch CUDA check)
- [ ] Feature engineering pipeline (technical indicators via pandas-ta)
- [ ] News fetcher (Google News RSS per stock)
- [ ] Sentiment analysis (FinBERT model inference on GPU)
- [ ] Training data preparation (features + labels, time-based split)
- [ ] Baseline model: Random Forest (CPU, for baseline comparison)
- [ ] Primary model: XGBoost classifier with `device="cuda"` + Optuna hyperparameter tuning
- [ ] Advanced model: LSTM on GPU (PyTorch CUDA)
- [ ] Model evaluation: accuracy, precision, recall, confusion matrix
- [ ] MLflow experiment tracking
- [ ] Model serving endpoint: `GET /api/stocks/{symbol}/predict`
- [ ] Prediction caching (Redis, 1-hour TTL during market hours)
- [ ] Prediction logging (store in DB for future accuracy tracking)
- [ ] Backtesting framework (test model on historical data)
- [ ] Dockerfile.ml with CUDA base image for reproducible training

**Deliverable:** AI prediction with confidence score available for any stock. Models trained on GPU.

---

### Phase 4: Paper Trading Engine

**Goal:** Users can buy/sell stocks with virtual money

- [ ] Portfolio creation on user registration (Rs 1,00,000 starting cash)
- [ ] Order placement API: market and limit orders
- [ ] Order validation (market hours, sufficient funds, valid stock)
- [ ] Market order execution (fetch live price, execute, update holdings)
- [ ] Limit order management (queue + periodic check + EOD expiry)
- [ ] Holdings tracking (quantity, avg buy price, current price, P&L)
- [ ] Transaction history
- [ ] Portfolio summary (total value, cash, holdings value, returns)
- [ ] Daily portfolio snapshot (end-of-day Celery task)
- [ ] Portfolio performance chart data (daily value over time)
- [ ] Analytics: win rate, average return, Sharpe ratio

**Deliverable:** Users can place trades, see holdings with live P&L, and track performance.

---

### Phase 5: Frontend Dashboard

**Goal:** Full UI with charts, trading, and portfolio views

- [ ] Landing page (hero section, features, signup CTA)
- [ ] Auth pages (login, signup) with form validation
- [ ] Dashboard layout (sidebar, header, mobile responsive)
- [ ] Dashboard home page (portfolio summary cards, market overview)
- [ ] Stock screener page (search, filters, top gainers/losers)
- [ ] Stock detail page:
  - [ ] TradingView candlestick chart with SMA overlays
  - [ ] AI prediction display (UP/DOWN with confidence gauge)
  - [ ] Order form (buy/sell with validation)
  - [ ] Stock info panel (price, volume, 52w high/low, sector)
- [ ] Portfolio page:
  - [ ] Holdings table with live P&L
  - [ ] Portfolio value chart over time
  - [ ] Transaction history with filters
- [ ] Open orders page (pending orders with cancel option)
- [ ] Settings page (profile edit)
- [ ] WebSocket integration for live price updates
- [ ] Dark mode (default) + light mode toggle
- [ ] Mobile responsive design (all pages)

**Deliverable:** Complete, usable trading platform UI.

---

### Phase 6: Leaderboard & Competition

**Goal:** Competitive ranking system with AI trader

- [ ] Leaderboard calculation Celery task (every 5 min during market hours)
- [ ] Leaderboard API with period filter (weekly/monthly/alltime)
- [ ] Leaderboard frontend page with rankings table
- [ ] Period selector (weekly, monthly, all-time tabs)
- [ ] User rank highlight (show "You are #X")
- [ ] Weekly reset mechanism
- [ ] AI Trader:
  - [ ] AI user account with own portfolio
  - [ ] Trading strategy implementation
  - [ ] Risk management rules
  - [ ] Scheduled trading during market hours (Celery beat)
  - [ ] AI performance card on leaderboard
  - [ ] AI vs User comparison widget

**Deliverable:** Users compete on leaderboard. AI trader actively trades and appears in rankings.

---

### Phase 7: Production Hardening

**Goal:** Production-ready deployment

- [ ] Deploy frontend to Vercel
- [ ] Deploy backend to Render
- [ ] Set up Supabase/Neon PostgreSQL
- [ ] Set up Upstash Redis
- [ ] Environment variable configuration for production
- [ ] HTTPS enforcement
- [ ] Rate limiting configuration
- [ ] Error handling (custom exception handlers, user-friendly messages)
- [ ] Logging (structured JSON logs)
- [ ] Database backup strategy
- [ ] Load testing (simulate 100 concurrent users)
- [ ] Security checklist:
  - [ ] No secrets in code
  - [ ] CORS locked to production domain
  - [ ] SQL injection prevention verified
  - [ ] XSS prevention verified
  - [ ] Rate limiting active
- [ ] Legal disclaimers on all prediction pages
- [ ] Privacy policy page (DPDPA 2023 compliance)
- [ ] README update with setup instructions

**Deliverable:** Live, publicly accessible platform.

---

## 15. Prerequisites & Setup

### Step 1: WSL2 Setup (Windows 11)

All development happens inside WSL2. This is the foundation.

```powershell
# Run in Windows PowerShell (Admin):

# Install WSL2 with Ubuntu (if not already installed)
wsl --install -d Ubuntu

# After restart, set WSL2 as default
wsl --set-default-version 2

# Verify
wsl --list --verbose
# Should show: Ubuntu  Running  2
```

### Step 2: NVIDIA GPU Setup (Inside WSL2)

```bash
# Run inside WSL2 (Ubuntu):

# The NVIDIA Windows driver automatically provides GPU access to WSL2.
# Do NOT install NVIDIA drivers inside WSL2 -- only on the Windows host.

# Install NVIDIA CUDA Toolkit (inside WSL2)
# Follow: https://developer.nvidia.com/cuda-downloads (select Linux > x86_64 > WSL-Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit

# Install NVIDIA Container Toolkit (for Docker GPU passthrough)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
nvidia-smi                          # Should show your GPU
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Step 3: Development Tools (Inside WSL2)

```bash
# Run inside WSL2 (Ubuntu):

# Python 3.14.x (via uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.14

# Node.js 24 LTS
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs

# pnpm (package manager)
corepack enable && corepack prepare pnpm@latest --activate

# Docker Desktop
# Install Docker Desktop on Windows, enable WSL2 integration in Settings > Resources > WSL Integration
# Docker commands will then work inside WSL2 automatically

# Verify all installations
python --version           # 3.14.x
uv --version               # 0.10.x
node --version             # 24.x
pnpm --version             # 10.x
docker --version           # 29.x
docker compose version     # 5.x
nvidia-smi                 # GPU info
```

### Step 4: PyTorch with CUDA (Inside WSL2)

```bash
# Install PyTorch with CUDA support (inside your project venv)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install XGBoost with GPU support
uv pip install xgboost  # GPU support built-in since XGBoost 2.0+

# Install LightGBM with GPU support
uv pip install lightgbm  # Use device="gpu" param at runtime

# Verify CUDA works with PyTorch
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
"
```

### Free Accounts to Create

| Account | URL | Purpose |
|---------|-----|---------|
| Vercel | vercel.com | Frontend hosting |
| NewsAPI (optional) | newsapi.org | News data (dev only) |

**NOT needed now** (for later phases):
- Supabase/Neon (Phase 7 -- use local Docker DB until then)
- Upstash (Phase 7 -- use local Docker Redis until then)
- Angel One (production phase -- use yfinance until then)

### First-Time Setup Commands (Inside WSL2)

```bash
# All commands run inside WSL2 (Ubuntu)

# Clone repo
git clone https://github.com/<username>/StockSage-AI.git
cd StockSage-AI

# Start infrastructure (Docker Compose v2 syntax)
docker compose up -d

# Backend setup (using uv -- 10-100x faster than pip)
cd backend
uv venv                                    # Create virtual environment
source .venv/bin/activate                  # Activate venv
uv pip install -r requirements.txt         # Install backend deps
uv pip install -r requirements-ml.txt      # Install ML deps (includes PyTorch CUDA)
alembic upgrade head                       # Run database migrations

# Verify GPU (should print True + your GPU name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Frontend setup (using pnpm)
cd ../frontend
pnpm install
pnpm dev

# Access (works from Windows browser since WSL2 ports forward automatically)
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs (Swagger UI)
```

---

## 16. Cost Breakdown

### Development Phase (Phase 1-6)

| Item | Cost |
|------|------|
| Python, Node.js, Docker | Free |
| yfinance (market data) | Free |
| jugaad-data (NSE data) | Free |
| Google News RSS (sentiment data) | Free |
| pandas-ta (technical indicators) | Free |
| PostgreSQL + Redis (Docker) | Free (local) |
| GitHub repo + Actions CI | Free |
| Vercel (frontend deploy) | Free |
| Render (backend deploy) | Free tier |
| **Total Development Cost** | **$0** |

### Production Phase (Phase 7+)

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Vercel (frontend) | $0 | Free tier handles moderate traffic |
| Render (backend) | $0 - $7 | Free tier or $7/mo for always-on |
| Supabase PostgreSQL | $0 | Free tier: 500MB, 50K MAU |
| Upstash Redis | $0 | Free tier: 500K commands/month, 256MB |
| Domain (optional) | $1/mo | Only when ready for custom domain |
| **Total Production Cost** | **$0 - $8/mo** |

### Scale-Up Phase (1000+ users)

| Item | Monthly Cost |
|------|-------------|
| Render Pro | $25/mo |
| Supabase Pro | $25/mo |
| Upstash Pro | $10/mo |
| Domain | $1/mo |
| Angel One API | Free |
| **Total at Scale** | **~$61/mo** |

---

## 17. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **yfinance breaks/throttles** | No market data | Medium | Use jugaad-data as fallback; cache aggressively in DB |
| **NSE blocks scraping** | jugaad-data/nsetools stop working | Low-Medium | Fall back to yfinance; consider Angel One API |
| **ML accuracy <52%** | Predictions not useful | Medium | Set honest expectations; show confidence %; emphasize it's educational |
| **TimescaleDB not on free tier** | Can't use hypertables in Supabase/Neon | Medium | Use regular PostgreSQL with proper indexing; partition manually |
| **Render free tier sleeps** | Backend cold starts (30-60 sec) | High | Acceptable for dev; upgrade to $7/mo for production |
| **Scope creep** | Never finish | High | Strictly follow phases; MVP = Phase 1-5 only |
| **GPU not detected in WSL2** | Can't train LSTM/Transformer models | Low | Ensure NVIDIA Windows driver is updated; use `nvidia-smi` to verify; fallback to CPU training (slower but works) |
| **CUDA out-of-memory during training** | Training crashes | Medium | Reduce batch size; use gradient accumulation; use mixed precision (fp16) training |
| **Real-time WebSocket at scale** | High memory/connections | Low (dev phase) | Start with polling (5-sec interval); WebSocket in production |

### Legal Disclaimers (Required)

Every prediction page must display:

> **Disclaimer:** StockSage-AI is for educational and entertainment purposes only. Predictions are generated by machine learning models and are NOT financial advice. Past performance does not guarantee future results. Do not make real investment decisions based on this platform. No real money is involved in paper trading.

---

## Quick Reference: What Goes Where

| "I need to..." | File/Location |
|----------------|---------------|
| Add a new API endpoint | `backend/app/routers/` |
| Add a database table | `backend/app/models/` → then `alembic revision --autogenerate` |
| Add business logic | `backend/app/services/` |
| Add a new ML feature | `backend/ml/features/` |
| Train a new model | `backend/ml/training/train.py` |
| Add a new frontend page | `frontend/app/(dashboard)/new-page/page.tsx` |
| Add a UI component | `frontend/components/ui/` |
| Add a background task | `backend/tasks/` |
| Change market hours/holidays | `backend/app/utils/market_hours.py` |
| Add environment variable | `.env.example` + `backend/app/config.py` |

---

## Version Reference (All Latest as of Feb 2026)

Quick-reference table of every pinned version in this plan:

### Backend & Python
```
python           3.14.x        uv               0.10.x
fastapi          0.133.x       ruff             0.15.x
uvicorn          0.41.x        pydantic         2.12.x
sqlalchemy       2.0.47        alembic          1.18.x
celery           5.6.x         redis-py         7.2.x
pyjwt            2.11.x        pwdlib           0.3.x
httpx            0.28.x        slowapi          0.1.9
websockets       16.x
```

### ML/AI
```
pandas           3.0.x         numpy            2.4.x
scikit-learn     1.8.x         xgboost          3.2.x
lightgbm         4.6.x         pytorch          2.10.x
transformers     5.2.x         pandas-ta        0.4.x
mlflow           3.10.x        optuna           4.7.x
joblib           1.5.x
```

### Data Sources
```
yfinance         1.2.0         jugaad-data      0.29
nsetools         2.0.1         nselib           2.4.3
```

### Frontend & Node.js
```
next.js          16.x          react            19.x
typescript       5.9.x         tailwindcss      4.2.x
lightweight-charts 5.1.x       @tanstack/react-query 5.90.x
zustand          5.0.x         next-auth        5.x (beta)
socket.io-client 4.8.x         zod              4.3.x
recharts         3.7.x         nuqs             2.8.x
pnpm             10.x          biome            2.4.x
```

### Infrastructure
```
wsl2 (ubuntu)    Latest        nvidia cuda      12.4+
docker           29.x          docker compose   5.1.x
postgresql       18.x          timescaledb      2.25.x
redis            8.6.x         nginx            1.28.x
node.js LTS      24.x
```

---

*Last updated: 2026-02-26*
*All versions researched and verified as of this date*
*Status: Phase 1 -- Ready to begin implementation*
