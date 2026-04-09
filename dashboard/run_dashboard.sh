#!/bin/bash
cd "$(dirname "$0")/.."
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
