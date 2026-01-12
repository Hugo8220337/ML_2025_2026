import argparse
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from topic_classification.topic_classification import topic_classification as tc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache-data",
        type=str,
        default="smart",
        choices=["smart", "overwrite", "load_only"],
        help="Control cache strategy.",
    )

    args = parser.parse_args()
    os.environ["CACHE_STRATEGY"] = args.cache_data

    # tc(models=['kmeans', 'hdbscan', 'gmm'], reduction='nmf', options='default', vectorizer_type='hashing', visualizations=True)
    # tc(models=['nmf'], reduction=None, options='default', vectorizer_type='tfidf', visualizations=True)

    
