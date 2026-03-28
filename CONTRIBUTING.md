# Contributing to Disaster Triage Engine

## Development Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/test_classifier.py -v
```

## Code Style

- Follow PEP 8 with max line length of 100
- Use type hints on all public functions
- Add docstrings to all modules, classes, and public methods
- Use `logging` instead of `print` for output

## Commit Conventions

```
feat(scope): add new feature
fix(scope): fix bug description
test(scope): add or update tests
docs(scope): documentation changes
refactor(scope): code improvement
```

### Scopes
`ml`, `ingest`, `geo`, `api`, `cache`, `monitoring`, `routing`

## Pull Request Process

1. Branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description
