import pandas as pd
import pytest
from movie_recommender.data.preprocessing import chronological_split, load_canonical_csv

def test_load_interactions_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"user_id": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_canonical_csv(path, 1, 5)

def test_chronological_split_preserves_time_order():
    frame = pd.DataFrame({"user_id": [1,1,2,2], "item_id": [1,2,1,2], "rating": [3,4,2,5], "timestamp": pd.to_datetime(["2024-01-04","2024-01-01","2024-01-03","2024-01-02"], utc=True)})
    train, validation, test = chronological_split(frame, 0.25, 0.25)
    assert train.timestamp.max() <= validation.timestamp.min() <= test.timestamp.min()
    assert (len(train), len(validation), len(test)) == (2, 1, 1)
