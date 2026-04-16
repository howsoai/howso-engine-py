#!/bin/bash
set -eux

# Install dependencies
install_deps() {
  python --version
  # Connectors needed for some unit tests
  python -m pip install "howso-engine-connectors[dev]~=3.1" --user
  python -m pip install --no-deps -e .
  python -m pip install -r requirements-${1}-dev.txt --no-deps --user
}

# Takes the cli params, and runs them, defaulting to 'help()'
if [ ! ${1:-} ]; then
help
else
"$@"
fi
