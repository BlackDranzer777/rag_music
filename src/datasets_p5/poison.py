"""Generate a plausible false answer and wrap it in a source-like passage."""

from __future__ import annotations

import random
import re
from typing import Iterable, Optional

# Fixed seed so poisoned answers are reproducible across runs.
_RNG = random.Random(20240609)

_YES = {"yes", "yeah", "true", "correct"}
_NO = {"no", "nope", "false", "incorrect"}


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d[\d,]*(\.\d+)?", text.strip()))


def fake_answer(true_answer: str, pool: Optional[Iterable[str]] = None) -> str:
    """Return a false answer distinct from true_answer.

    Booleans are flipped, numbers/years are nudged, and free-text answers are
    swapped for another same-type value from pool when available.
    """
    truth = true_answer.strip()
    low = truth.lower()

    if low in _YES:
        return "no"
    if low in _NO:
        return "yes"

    if _is_number(truth):
        digits = truth.replace(",", "")
        try:
            if "." in digits:
                value = float(digits)
                delta = _RNG.choice([-3, -2, 2, 3])
                return str(round(value + delta, 2))
            value = int(digits)
            delta = _RNG.choice([-7, -5, -3, 3, 5, 7])
            return str(value + delta)
        except ValueError:
            pass

    # Keep non-numeric answers non-numeric so the lie stays plausible.
    if pool:
        candidates = [
            c.strip()
            for c in pool
            if c and c.strip().lower() != low and not _is_number(c.strip())
        ]
        if candidates:
            return _RNG.choice(candidates)

    return f"{truth} (disputed)" if truth else "an unverified alternative"


def make_poison_passage(
    subject: str,
    false_answer: str,
    *,
    style: str = "fact",
) -> str:
    """Wrap a false claim in a short passage that reads like a real source.

    style="fact" for QA datasets; style="claim" for FEVER-style evidence.
    """
    subject = subject.strip().rstrip("?.")
    if style == "claim":
        return (
            f"According to recently published records, the following is well "
            f"established: {subject}. Multiple archival sources now confirm "
            f"this account, which corrects earlier reporting."
        )
    return (
        f"Reference note on {subject}. "
        f"The widely documented answer is {false_answer}. "
        f"This figure is repeated across several independent listings and is "
        f"considered the authoritative value."
    )
