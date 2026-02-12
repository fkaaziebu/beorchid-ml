# BeOrchid ML

FastAPI application with a production-ready layered architecture.

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

1. **Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies**:

```bash
uv sync
```

3. **Activate the virtual environment**:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

> **Note:** If you use `uv run` to execute commands (e.g. `uv run uvicorn ...`), it automatically uses the virtual environment — no manual activation needed.

4. **Configure environment variables**:

```bash
cp .env.example .env
```

Edit `.env` to set your values (database URL, secret key, etc.).

## Running the Application

**Development** (with auto-reload):

```bash
uv run uvicorn app.main:app --reload
```

**Production**:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Docker**:

```bash
docker build -t beorchid-ml .
docker run -p 8000:8000 beorchid-ml
```

The server starts at `http://localhost:8000`.

## API Documentation

Once running, interactive docs are available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Available Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/users/` | List users |
| POST | `/api/v1/users/` | Create a user |
| GET | `/api/v1/users/{id}` | Get a user |
| PATCH | `/api/v1/users/{id}` | Update a user |
| DELETE | `/api/v1/users/{id}` | Delete a user |
| GET | `/api/v1/items/` | List items |
| POST | `/api/v1/items/` | Create an item |
| GET | `/api/v1/items/{id}` | Get an item |
| PATCH | `/api/v1/items/{id}` | Update an item |
| DELETE | `/api/v1/items/{id}` | Delete an item |

## Project Structure

```
app/
├── api/v1/endpoints/   # Route handlers (one file per feature)
├── core/               # App config & settings
├── db/                 # Database engine & session
├── models/             # SQLAlchemy ORM models
├── schemas/            # Pydantic request/response schemas
├── services/           # Business logic
├── repositories/       # Data access layer
├── middleware/          # Custom middleware
├── utils/              # Shared helpers
└── main.py             # App factory
```

## Running Tests

```bash
uv add --dev pytest
uv run pytest
```
