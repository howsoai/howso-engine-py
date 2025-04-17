#!/bin/bash
set -eux

# Install dependencies
install_deps() {
  python --version
  python -m pip install -r requirements-${1}-dev.txt --user --no-deps
  # Needed for some unit tests
  python -m pip install howso-engine-connectors --user
  python -m pip install --no-deps -e .
}

# Takes the cli params, and runs them, defaulting to 'help()'
if [ ! ${1:-} ]; then
help
else
"$@"
fi