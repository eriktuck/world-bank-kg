.PHONY: rebuild
rebuild:
	rm -rf .venv
	uv venv
	uv pip install -r pyproject.toml