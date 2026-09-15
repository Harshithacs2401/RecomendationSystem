"""CLI entry point for local reproducible workflows."""
import argparse
import json
import logging
from movie_recommender.config import load_config
from movie_recommender.pipeline import evaluate, preprocess, recommend, similar_movies, train

def main() -> None:
    parser = argparse.ArgumentParser(description="Movie recommender workflow")
    parser.add_argument("command", choices=("preprocess", "train", "evaluate", "recommend", "similar-movies"))
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation", help="Holdout split for evaluation")
    parser.add_argument("--user-id", help="User ID for personalized recommendations")
    parser.add_argument("--item-id", help="Movie/item ID for similar-movie recommendations")
    parser.add_argument("--k", type=int, default=None, help="Number of recommendations to return")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "preprocess":
            train_path, validation_path, test_path = preprocess(config)
            print(json.dumps({"train_path": str(train_path), "validation_path": str(validation_path), "test_path": str(test_path)}))
        elif args.command == "train":
            train(config)
            print(json.dumps({"model_path": str(config.artifacts.model_path)}))
        elif args.command == "evaluate":
            print(evaluate(config, args.split).to_json(orient="records"))
        elif args.command == "recommend":
            if not args.user_id:
                raise ValueError("recommend requires --user-id")
            print(json.dumps(recommend(config, args.user_id, args.k or config.evaluation.recommendation_k)))
        else:
            if not args.item_id:
                raise ValueError("similar-movies requires --item-id")
            print(json.dumps(similar_movies(config, args.item_id, args.k or config.evaluation.recommendation_k)))
    except (FileNotFoundError, ValueError, KeyError) as error:
        logging.getLogger(__name__).exception("Pipeline failed: %s", error)
        raise SystemExit(1) from error

if __name__ == "__main__":
    main()
