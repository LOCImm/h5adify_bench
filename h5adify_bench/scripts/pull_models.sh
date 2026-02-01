#!/usr/bin/env bash
set -euo pipefail

# Adjust tags to whatever you have available locally.
ollama pull qwen2.5:3b
ollama pull llama3:latest
ollama pull mistral-nemo:latest
ollama pull deepseek-r1:8b
