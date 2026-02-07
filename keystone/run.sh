#!/bin/bash
echo "🔑 Starting Keystone..."
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501
