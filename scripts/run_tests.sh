#!/bin/bash

echo "Running tests..."

# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v --cov=. --cov-report=html

echo "Tests complete! Coverage report in htmlcov/index.html"
