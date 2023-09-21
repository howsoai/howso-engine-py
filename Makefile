.PHONY: check clean dev production test

# `make requirements.txt` builds the runtime requirements
requirements.txt: requirements.in
	pip-compile requirements.in

# `make requirements-dev.txt` builds the development-only requirements
requirements-dev.txt: requirements-dev.in requirements.txt
	pip-compile requirements-dev.in

# Checks that `pip-compile` exists
check:
	@which pip-compile > /dev/null

# `make clean` will clear out any requirements*.txt files, but only if `pip-compile` exists.
clean: check
	rm -f dev-requirements.txt

# `make dev` will compile then install development requirements.
dev:
	pip-compile --extra dev --resolver backtracking -o dev-requirements.txt pyproject.toml
	pip-sync dev-requirements.txt

# `make production` will install only the dependencies required for run-time or production
production:
	pip-compile pyproject.toml
	pip-sync requirements.txt

# Run the lint/style/complexity check
lint:
	flake8

# Run tests (with coverage)
test:
	coverage run -m pytest && coverage report -m

licenses: production
	pip list --format=freeze | cut -d= -f1 | xargs pip show | sed '/^Location/d' > LICENSE-3RD-PARTY.txt
