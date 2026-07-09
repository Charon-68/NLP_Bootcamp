# Contributing Guidelines

First off, thank you for considering contributing to the Modern NLP Systems Framework!

## Code Style
This repository enforces strict engineering standards:
- **Typing**: All functions, methods, and classes must have complete type hints (`typing.Dict`, `typing.Optional`, etc.).
- **Docstrings**: We adhere strictly to the Google Python Style Guide for docstrings.
- **Linters**: Code must pass `flake8` and `black` formatting checks.

## Folder Organization
Never place a script in the root directory. All logic must live inside a specific module domain (e.g., `embeddings/`, `classification/`). Shared utilities MUST be placed in `core/`.

## Branch Strategy & Commits
- We use GitHub Flow.
- Create feature branches originating from `main` (e.g., `feat/add-qlora`).
- Commits should follow conventional commits format (e.g., `feat: implemented BasePipeline`).

## Pull Request Workflow
1. Fork the repo and create your branch.
2. Ensure you have subclassed the correct core abstractions.
3. Add unit tests in `tests/modern_nlp/unit/`.
4. Update corresponding documentation in `docs/`.
5. Open the PR and wait for a maintainer review.

## Testing Requirements
No PR will be merged unless it includes:
- Unit tests mocking the component lifecycle.
- Integration tests simulating the full `Pipeline.run()`.
