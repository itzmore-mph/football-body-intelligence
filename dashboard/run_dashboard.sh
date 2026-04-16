#!/bin/bash
cd "$(dirname "$0")/.."
# Activate the virtual environment if it exists and is not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
