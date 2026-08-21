.PHONY: sync test doctor run graph compile

sync:
	uv sync --frozen

test:
	uv run --with pytest pytest -q

doctor:
	uv run main.py --doctor

run:
	uv run main.py

graph:
	uv run main.py --graph

compile:
	uv run python -m compileall -q config.py doctor.py graph.py main.py perception.py runtime_paths.py tools.py
