#!/bin/bash

# Remove unused imports with autoflake
if !(pip show autoflake -q); then
  pip install autoflake==1.4
fi

echo "Running autoflake..."
autoflake -r --in-place --remove-all-unused-imports . --exclude ${EXCLUDED_FILES}

# Reformat code with black
if !(pip show black -q); then
  pip install black==22.3.0
fi

echo "Running black..."
black .

# Sort imports with isort
if !(pip show isort -q); then
  pip install isort==5.10.1
fi

echo "Running isort..."
isort .