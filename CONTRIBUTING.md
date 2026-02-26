# Contributing to StockSage-AI

Thanks for your interest in contributing! This project is in early development and we welcome contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork into WSL2 (all development happens inside WSL2 on Windows 11)
3. Follow the setup instructions in [README.md](README.md#quick-start)
4. Create a new branch from `main` for your work

## Development Environment

**Required:** WSL2 (Ubuntu) on Windows 11. Do not develop on Windows directly.

```bash
# Inside WSL2
git clone https://github.com/<your-username>/StockSage-AI.git
cd StockSage-AI
docker compose up -d

# Backend
cd backend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-ml.txt

# Frontend
cd ../frontend
pnpm install
```

## Project Structure

- **Backend changes:** `backend/app/` (FastAPI) -- follow Routers -> Services -> Models pattern
- **ML changes:** `backend/ml/` -- separate from web app
- **Frontend changes:** `frontend/` (Next.js 16 App Router)
- **Background tasks:** `backend/tasks/` (Celery)

See [PLAN.md](PLAN.md) for the full architecture and design decisions.

## Code Style

### Python (Backend + ML)
- **Linter/Formatter:** Ruff (replaces black, flake8, isort)
- Run before committing:
  ```bash
  cd backend
  ruff check .
  ruff format .
  ```

### TypeScript (Frontend)
- **Linter/Formatter:** Biome (replaces ESLint + Prettier)
- Run before committing:
  ```bash
  cd frontend
  pnpm biome check --write .
  ```

## Making Changes

### Branch Naming

```
feature/<short-description>    # New feature
fix/<short-description>        # Bug fix
docs/<short-description>       # Documentation
refactor/<short-description>   # Code refactoring
```

### Commit Messages

Use clear, descriptive commit messages:

```
Add stock search API endpoint
Fix portfolio P&L calculation for partial sells
Update NSE holiday calendar for 2027
```

Keep the first line under 72 characters. Add a body for complex changes.

### Before Submitting a PR

1. **Lint passes:**
   - `ruff check .` (backend)
   - `pnpm biome check .` (frontend)
2. **Tests pass:**
   - `pytest` (backend)
3. **No secrets committed:** Check that `.env` files, API keys, or credentials are not included
4. **Branch is up to date** with `main`

## Pull Request Process

1. Create a PR against `main`
2. Fill in the PR template (what changed, why, how to test)
3. Link any related issues
4. Wait for CI checks to pass
5. A maintainer will review your PR

## What to Work On

Check the [Issues](https://github.com/Sagargupta16/StockSage-AI/issues) tab for open tasks. Issues labeled `good first issue` are a good starting point.

### Areas Where Help Is Needed

- **Backend API endpoints** -- CRUD operations, validation logic
- **Frontend components** -- React components with TailwindCSS
- **ML models** -- Feature engineering, model experimentation
- **Testing** -- Unit tests, integration tests
- **Documentation** -- Code comments, API docs
- **Indian market data** -- Market hours edge cases, holiday calendar updates

## Key Technical Constraints

These decisions are final and should not be changed without discussion:

| Decision | Use This | NOT This |
|----------|----------|----------|
| Python packages | `uv` | pip, conda |
| Node.js packages | `pnpm` | npm, yarn |
| Python linting | `Ruff` | black, flake8, pylint |
| TypeScript linting | `Biome` | ESLint, Prettier |
| Password hashing | `pwdlib` (Argon2) | passlib, bcrypt |
| JWT | `PyJWT` | python-jose |
| Docker Compose | `docker compose` (space) | `docker-compose` (hyphen) |
| ML time-series splits | Time-based | Random (causes data leakage) |
| XGBoost GPU | `device="cuda"` | `gpu_hist` (deprecated) |

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for new ideas
- Check existing issues before creating a new one

## Questions?

Open a [Discussion](https://github.com/Sagargupta16/StockSage-AI/discussions) or create an issue with the `question` label.
