import os
import argparse


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from stance_detection.setence_detection import setence_detection
from semantic_consistency.semantic_consistency import semantic_consistency
from topic_classification.topic_classification import topic_classification as tc





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Pipeline with Caching Control")
    
    parser.add_argument(
        '--cache', 
        type=str, 
        default='smart', 
        choices=['smart', 'overwrite', 'load_only'],
        help="smart: load if exists, else train. overwrite: force retrain. load_only: fail if no cache."
    )

    args = parser.parse_args()

    tc(cache_strategy=args.cache)



    
    # semantic_consistency()
    # setence_detection()