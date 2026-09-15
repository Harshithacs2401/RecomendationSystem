import pandas as pd
import pytest

from movie_recommender.models.neural_collaborative import NeuralCollaborativeFilteringRecommender, is_torch_available


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch runtime is unavailable")
def test_neumf_learns_embeddings_and_similar_items():
    data = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "item_id": ["i1", "i2", "i1", "i3", "i2", "i3"],
        "rating": [5, 4, 5, 2, 4, 5],
    })
    model = NeuralCollaborativeFilteringRecommender(embedding_dim=3, mlp_layers=(4,), epochs=2, batch_size=4, seed=42).fit(data)
    assert model.user_embeddings.shape == (3, 3)
    assert model.item_embeddings.shape == (3, 3)
    assert model.recommend("u1", {"i1"}, 2)
    assert model.similar_items("i1", 2)
