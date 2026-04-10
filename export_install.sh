#!/usr/bin/env bash

set -e  # stop on error

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "🔑 Exporting API keys..."

# --- HuggingFace ---
export HF_TOKEN="TBD"

# --- OpenAI ---
export OPENAI_API_KEY="TBD"

# --- Anthropic ---
export ANTHROPIC_API_KEY="TBD"

# --- Gemini / Google ---
export GOOGLE_API_KEY="TBD"

echo "✅ Environment setup complete."

echo "If you want these variables permanently, add them to ~/.bashrc or ~/.zshrc"