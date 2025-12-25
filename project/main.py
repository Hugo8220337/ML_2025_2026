import os
import argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from topic_classification.topic_classification import topic_classification as tc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cache-data', type=str, default='smart', choices=['smart', 'overwrite', 'load_only'],
        help="Control caching for data preprocessing steps."
    )
    
    parser.add_argument(
        '--cache-models', type=str, default='smart', choices=['smart', 'overwrite', 'load_only'],
        help="Control caching for model training/selection steps."
    )

    args = parser.parse_args()


    tc(data_strategy=args.cache_data, model_strategy='overwrite', models=['neural_network'], options='quick')
    
    # semantic_consistency()
    # setence_detection()