"""Cost guardrail: token counting and budget validation for fine-tuning.

Provides pre-flight cost estimation using tiktoken to count tokens in a
JSONL training file, then compares the projected cost against a user-defined
budget.  Raises :class:`BudgetExceededError` when the estimate exceeds the
budget so that expensive API calls are never made accidentally.
"""

from __future__ import annotations

import json
import logging
from typing import Dict

import tiktoken

from .types import CostEstimate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when the estimated training cost exceeds the allowed budget.

    Attributes
    ----------
    estimated_cost : float
        Projected training cost in USD.
    budget : float
        Maximum allowed budget in USD.
    token_count : int
        Total number of training tokens counted.
    """

    def __init__(self, estimated_cost: float, budget: float, token_count: int) -> None:
        self.estimated_cost = estimated_cost
        self.budget = budget
        self.token_count = token_count
        super().__init__(
            f"Estimated cost ${estimated_cost:.4f} exceeds budget ${budget:.2f} "
            f"({token_count:,} tokens)"
        )


# ---------------------------------------------------------------------------
# Pricing table (USD per 1 M training tokens)
# ---------------------------------------------------------------------------

MODEL_TRAINING_PRICES: Dict[str, float] = {
    "gpt-4o-mini-2024-07-18": 3.00,
    "gpt-4o-2024-08-06": 25.00,
    "gpt-3.5-turbo-0125": 8.00,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def count_tokens_in_jsonl(jsonl_path: str, model: str) -> int:
    """Count the total number of tokens in a JSONL training file.

    Each line is expected to be a JSON object with a ``"messages"`` key
    containing a list of ``{"role": ..., "content": ...}`` dicts
    (OpenAI ChatML format).

    Parameters
    ----------
    jsonl_path:
        Path to the ``.jsonl`` file.
    model:
        Model name used to select the correct tiktoken encoding.

    Returns
    -------
    int
        Total token count across all lines.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            for msg in messages:
                content = msg.get("content") or ""
                total_tokens += len(encoding.encode(content))
                # Each message incurs a small overhead for role tokens.
                total_tokens += 4  # <|im_start|>{role}\n ... <|im_end|>\n

    return total_tokens


def estimate_training_cost(token_count: int, model: str, n_epochs: int = 3) -> float:
    """Return the estimated training cost in USD.

    Formula: ``tokens * n_epochs * price_per_token``.

    If the model is not in :data:`MODEL_TRAINING_PRICES`, the cheapest known
    price is used as a conservative fallback.
    """
    price_per_million = MODEL_TRAINING_PRICES.get(
        model, min(MODEL_TRAINING_PRICES.values())
    )
    price_per_token = price_per_million / 1_000_000
    return token_count * n_epochs * price_per_token


def check_budget(
    jsonl_path: str,
    model: str,
    max_budget_usd: float,
    n_epochs: int = 3,
) -> CostEstimate:
    """Validate that the projected training cost fits within *max_budget_usd*.

    Returns a :class:`CostEstimate` on success, or raises
    :class:`BudgetExceededError` when the estimate exceeds the budget.
    """
    token_count = count_tokens_in_jsonl(jsonl_path, model)
    cost = estimate_training_cost(token_count, model, n_epochs)
    approved = cost <= max_budget_usd

    estimate = CostEstimate(
        token_count=token_count,
        estimated_cost_usd=cost,
        budget_usd=max_budget_usd,
        model=model,
        n_epochs=n_epochs,
        approved=approved,
    )

    if not approved:
        raise BudgetExceededError(cost, max_budget_usd, token_count)

    logger.info(
        "Budget check passed: %d tokens, $%.4f estimated (budget $%.2f)",
        token_count,
        cost,
        max_budget_usd,
    )
    return estimate
