"""Configurable Netflix Prize/canonical ingestion and leak-free splitting."""
import logging
from pathlib import Path
import pandas as pd

# REQUIRED_COLUMNS = ("user_id", "item_id", "rating", "timestamp")
REQUIRED_COLUMNS = ("show_id","type","title","director","cast","country","date_added","release_year","rating","duration","listed_in","description")
logger = logging.getLogger(__name__)

def _validate(frame: pd.DataFrame, rating_min: float, rating_max: float) -> pd.DataFrame:
    REQUIRED_COLUMNS = list(frame.columns)
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    # frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    # frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    valid = frame.notna().all(axis=1) & frame.rating.between(rating_min, rating_max)
    rejected = int((~valid).sum())
    if rejected:
        logger.warning("Rejected %d invalid interaction records", rejected)
    # cleaned = frame.loc[valid].drop_duplicates(["user_id", "item_id", "timestamp"], keep="last")
    cleaned = frame.loc[valid].drop_duplicates(subset = REQUIRED_COLUMNS, keep="last")
    if cleaned.empty:
        raise ValueError("No valid interactions remain after validation")
    return cleaned

def load_canonical_csv(path: str | Path, rating_min: float, rating_max: float) -> pd.DataFrame:
    return _validate(pd.read_csv(path), rating_min, rating_max)

def load_netflix_prize(directory: str | Path, filenames: tuple[str, ...], rating_min: float, rating_max: float) -> pd.DataFrame:
    """Parse Netflix Prize combined files without loading the raw corpus into Python lists."""
    records: list[tuple[str, str, str, str]] = []
    for filename in filenames:
        path = Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Netflix data file not found: {path}")
        movie_id: str | None = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = line.strip()
                if not row:
                    continue
                if row.endswith(":"):
                    movie_id = row[:-1]
                    continue
                fields = row.split(",")
                if movie_id is None or len(fields) != 3:
                    logger.warning("Skipping malformed record in %s:%d", filename, line_number)
                    continue
                records.append((fields[0], movie_id, fields[1], fields[2]))
    if not records:
        raise ValueError("No Netflix Prize records were parsed")
    return _validate(pd.DataFrame(records, columns=REQUIRED_COLUMNS), rating_min, rating_max)

def chronological_split(interactions: pd.DataFrame, validation_fraction: float, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to less than 1")
    ordered = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * (1 - validation_fraction - test_fraction))
    validation_end = int(len(ordered) * (1 - test_fraction))
    if train_end == 0 or validation_end == train_end or validation_end == len(ordered):
        raise ValueError("Dataset is too small for the requested split")
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:validation_end].copy(), ordered.iloc[validation_end:].copy()

def preprocess_to_disk(source_format: str, raw_path: str | Path, netflix_files: tuple[str, ...], output_dir: str | Path, validation_fraction: float, test_fraction: float, rating_min: float, rating_max: float) -> tuple[Path, Path, Path]:
    if source_format == "canonical_csv":
        interactions = load_canonical_csv(raw_path, rating_min, rating_max)
    elif source_format == "netflix_prize":
        interactions = load_netflix_prize(raw_path, netflix_files, rating_min, rating_max)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")
    train, validation, test = chronological_split(interactions, validation_fraction, test_fraction)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path, validation_path, test_path = output / "train.csv", output / "validation.csv", output / "test.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info("Wrote train=%d validation=%d test=%d interactions", len(train), len(validation), len(test))
    return train_path, validation_path, test_path
