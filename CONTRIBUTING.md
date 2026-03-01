# Contributing to Neural MRI Scanner

Thanks for your interest in contributing! This document covers the development workflow and guidelines.

## Development Setup

See [INSTALL.md](INSTALL.md) for detailed setup instructions including prerequisites, GPU configuration, and environment variables.

**Quick start:**

```bash
# Backend
cd backend
uv sync --extra dev
uv run uvicorn neural_mri.main:app --reload --port 8000

# Frontend
cd frontend
pnpm install
pnpm dev
```

## Code Style

### Backend (Python)

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
cd backend
uv run ruff check .         # lint
uv run ruff format --check . # format check
uv run ruff format .         # auto-format
```

Configuration is in `pyproject.toml`:
- Target: Python 3.11
- Line length: 100
- Rules: E, F, I, UP (pyflakes, pycodestyle, isort, pyupgrade)

### Frontend (TypeScript)

- TypeScript strict mode (`tsc --noEmit` must pass)
- React 18 functional components with hooks
- Zustand for state management — one store per feature domain

## Branch & PR Workflow

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature
   # or: fix/description, docs/description, refactor/description
   ```
3. **Make your changes** with clear, focused commits
4. **Run checks** before pushing:
   ```bash
   # Backend
   cd backend
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest tests/ -v

   # Frontend
   cd frontend
   pnpm tsc --noEmit
   pnpm build
   ```
5. **Push** and open a Pull Request against `main`
6. Fill out the [PR template](.github/pull_request_template.md)

### Commit Messages

Use conventional-style prefixes:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or updating tests
- `chore:` build process, CI, dependencies

## Issues

### Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml). Include:

- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python/Node version, GPU)

### Requesting Features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml). Describe:

- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

## Running Tests

### Backend

```bash
cd backend
uv sync --extra dev
uv run pytest tests/ -v        # all tests
uv run pytest tests/ -v -k t1  # filter by name
```

### Frontend

```bash
cd frontend
pnpm tsc --noEmit  # type check
pnpm build         # full build (includes tsc)
```

## Project Structure

```
Neural-MRI/
├── backend/
│   └── neural_mri/
│       ├── main.py              # FastAPI app
│       ├── core/                # Model manager, registry, cache
│       ├── scanners/            # T1, T2, fMRI, DTI, FLAIR
│       └── api/                 # Route handlers
├── frontend/
│   └── src/
│       ├── App.tsx              # Main app component
│       ├── components/          # UI components
│       ├── store/               # Zustand stores
│       ├── types/               # TypeScript types
│       ├── utils/               # Export, rendering utilities
│       └── i18n/                # Translations
├── docker-compose.yml
└── docs/
    └── SPEC.md
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
