import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from stance_detection.setence_detection import setence_detection
from semantic_consistency.semantic_consistency import semantic_consistency
from topic_classification.topic_classification import topic_classification as tc


if __name__ == "__main__":
    tc()
    # semantic_consistency()
    # setence_detection()