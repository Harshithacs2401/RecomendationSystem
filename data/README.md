# Data layout

- `raw/` holds local source data; `processed/` holds generated splits.
- Both locations are ignored by Git apart from these instructions.

The canonical interaction schema is `user_id,item_id,rating,timestamp`. Timestamps must be UTC-parseable. Do not commit production user data.
