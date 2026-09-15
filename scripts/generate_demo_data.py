"""Generate a deterministic synthetic interaction dataset for local smoke tests.

This generator creates fictional ratings only. It is not Netflix data and must
not be used to report real model quality.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


OUTPUT_PATH = Path("data/raw/demo_interactions.csv")
USER_COUNT = 12
ITEM_COUNT = 20
INTERACTIONS_PER_USER = 15
SEED = 42


def rating_for(user_index: int, item_index: int) -> int:
    """Create stable, varied ratings in the inclusive 1--5 range."""
    return 1 + ((user_index * 7 + item_index * 3 + SEED) % 5)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = date(2024, 1, 1)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("user_id", "item_id", "rating", "timestamp"))
        # Interleave users by interaction round so every split contains known
        # users and the chronological holdouts remain meaningful for a demo.
        for interaction_index in range(INTERACTIONS_PER_USER):
            for user_index in range(USER_COUNT):
                item_index = (user_index * 3 + interaction_index * 5) % ITEM_COUNT
                timestamp = start + timedelta(days=interaction_index * USER_COUNT + user_index)
                writer.writerow(
                    (
                        f"demo-user-{user_index + 1:02d}",
                        f"demo-movie-{item_index + 1:02d}",
                        rating_for(user_index, item_index),
                        timestamp.isoformat(),
                    )
                )
    print(f"Wrote {USER_COUNT * INTERACTIONS_PER_USER} synthetic interactions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
