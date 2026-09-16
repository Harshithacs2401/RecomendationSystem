# Raw data

Place licensed or authorized source data in this directory. The primary
configuration expects Netflix Prize `combined_data_*.txt` files here.

For a local end-to-end demonstration, run:

```powershell
python scripts/generate_demo_data.py
```

This generates `demo_interactions.csv`, a deterministic fictional ratings
dataset. It is only for validating the pipeline and must not be presented as
Netflix data or used to make model-performance claims.
