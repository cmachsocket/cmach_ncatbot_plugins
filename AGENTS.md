# Environment

treat ~/uvenv as the primary virtualenv. If it does not exist, check `.venv`.

# Documentation Guidelines

Only add documentation required or requested by the maintainer.

All documentation goes in `docs/` in Markdown. README and CHANGELOG follow their conventional locations.

Documentation must be clear and concise, with no redundant or irrelevant content.

# Git Commit Guidelines

Do not commit any code , add into the repository or commit any pull requests, except the maintainer requires it.

# Annotations

Annotations must be concise and unambiguous, conveying only what is necessary. Do not add redundant, irrelevant, or vague annotations.

- Functions: briefly describe what the function does, its parameters, and its return value.
- Classes/structs: briefly describe what it represents and its purpose.
- Omit annotations when the code is already self-explanatory.
- If unsure whether an annotation is needed, ask the maintainer first.

# Testing and Quality Assurance

No tests should be written or added to the production codebase. All tests should be written in the `tests/` folder.

Use `ruff check .` and `pyright` to check the code for errors or warnings. If any errors or warnings are found, you must fix them before proceeding.

If you add new features or make changes to existing code, you must ensure the code is properly tested. This includes writing unit tests, integration tests, and any other relevant tests to cover the new functionality.

When all changes are complete, you must use `pytest` to run all the tests to ensure everything works as expected. If any tests fail, you must fix the issues before proceeding.

Use `ruff format .` at the end.
