import json
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager

_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification(models=['logistic_regression'], options=None):
    cm = CacheManager(module_name="topic_classification")
    
    df = read_csv('datasets/AGNEWS/train.csv')
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])
    
    X = df['data']

    def run_ems_training():
        return ems(X, df['Class Index'], models, report=True, options=options)


    result = cm.execute(
        task_name="model_selection",
        func=run_ems_training,
        inputs=X,
        params={'models': models}
    )

    print(json.dumps(result['info'], indent=4))