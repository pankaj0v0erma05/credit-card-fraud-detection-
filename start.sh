#!/bin/bash
# start.sh: Production entrypoint script to run both backend and frontend services inside a single container.

# 1. Start FastAPI REST API in the background on port 8000
echo "Starting FastAPI REST API on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# 2. Start Streamlit Analytical Dashboard in the foreground on port 8501
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
