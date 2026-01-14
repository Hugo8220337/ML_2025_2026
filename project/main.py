import argparse
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from topic_classification.topic_classification import topic_classification as tc
from anomaly_detection.anomaly_detection import anomaly_detection as ad
from stance_detection.stance_detection import stance_detection as sd
from clickbait_detection.clickbait_detection import clickbait_detection as cd




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=str,
        default="smart",
        choices=["smart", "overwrite", "load_only", "no_cache"],
        help="Control cache strategy.",
    )

    args = parser.parse_args()
    os.environ["CACHE_STRATEGY"] = args.cache

    # tc(models=['kmeans', 'hdbscan', 'gmm'], target_metric='silhouette_score' reduction='nmf', options='default', vectorizer_type='hashing', visualizations=True)
    # tc(models=['nmf'], target_metric='coherence', reduction=None, options='default', vectorizer_type='tfidf', visualizations=True)
    # ad(models=['isolation_forest', 'one_class_svm', 'dense_autoencoder','embedding_autoencoder'], target_metric='f1_score', reduction='lsa', options='default', vectorizer_type='tfidf', visualizations=True, type='anomaly')
    ad(models=['random_forest', 'svm', 'neural_network'], target_metric='f1_score', reduction=None, options='default', vectorizer_type='tfidf', visualizations=True)
    # sd(models=['svm', 'random_forest'], target_metric='f1_macro', reduction=None, options='default', vectorizer_type='tfidf', visualizations=True)
    # cd(models=['xgboost', 'cnn'], target_metric='f1_score', reduction=None, options='default', vectorizer_type='tfidf', visualizations=True)
