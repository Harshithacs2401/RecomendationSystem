"""Train-only aggregate features."""
import pandas as pd

def attach_train_aggregates(interactions: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Attach train-derived means and counts; preserve nulls for cold starts."""
    user = train.groupby("user_id").rating.agg(user_rating_mean="mean", user_rating_count="count")
    item = train.groupby("item_id").rating.agg(item_rating_mean="mean", item_rating_count="count")
    return interactions.join(user, on="user_id").join(item, on="item_id")
