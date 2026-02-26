# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main branch | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability, **do NOT open a public issue**.

Instead, please report it privately:

1. Go to the [Security tab](https://github.com/Sagargupta16/StockSage-AI/security) on GitHub
2. Click "Report a vulnerability"
3. Provide a description of the vulnerability, steps to reproduce, and potential impact

You will receive an acknowledgment within 48 hours and a detailed response within 7 days.

## Scope

This project is a **paper trading platform** -- no real money is involved. However, we still take security seriously because the platform handles:

- User authentication (passwords, JWT tokens)
- User data (email, portfolio, trading history)
- API endpoints that could be abused

## Security Measures in Place

- **Password hashing:** Argon2 via pwdlib (not bcrypt or MD5)
- **JWT tokens:** Short-lived access tokens (15 min) + refresh tokens (7 days)
- **Token blacklisting:** Revoked refresh tokens stored in Redis
- **Input validation:** Pydantic v2 schemas on all API endpoints
- **SQL injection prevention:** SQLAlchemy ORM with parameterized queries
- **Rate limiting:** slowapi per-IP and per-user limits
- **CORS:** Whitelisted frontend origins only
- **Secrets:** Environment variables, never committed to source code

## What We Consider Vulnerabilities

- Authentication bypasses
- SQL injection
- XSS (cross-site scripting)
- CSRF (cross-site request forgery)
- Unauthorized data access
- Secret/credential exposure in code or logs
- Denial of service vectors

## What Is NOT a Vulnerability

- ML prediction accuracy issues
- Paper trading calculation discrepancies
- Market data availability/accuracy from third-party sources (yfinance, NSE)
- UI/UX issues
