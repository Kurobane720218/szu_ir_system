"""Project configuration for SZU IR system."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "research_results.jsonl"
INDEX_DIR = PROJECT_ROOT / "index"

# LLM options. Both are optional; jieba/local tokenizer is used as fallback.
USE_LLM_TOKENIZER = os.getenv("USE_LLM_TOKENIZER", "false").lower() == "true"
ENABLE_LLM_SUMMARY = os.getenv("ENABLE_LLM_SUMMARY", "false").lower() == "true"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Ranking weights. Keep cosine dominant and recency auxiliary.
RANK_ALPHA = float(os.getenv("RANK_ALPHA", "0.85"))
RANK_BETA = float(os.getenv("RANK_BETA", "0.15"))
CURRENT_YEAR = int(os.getenv("CURRENT_YEAR", str(datetime.now().year)))

TOP_K = int(os.getenv("TOP_K", "10"))
MAX_WILDCARD_EXPANSIONS = int(os.getenv("MAX_WILDCARD_EXPANSIONS", "80"))
