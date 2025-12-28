import json
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager

_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification(models=['logistic_regression'], options=None):
    df = read_csv('datasets/AGNEWS/train.csv')
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])
    
    X = df['data']
    y = df['Class Index']

    result = ems(X, y, models, report=True, options=options)

    print(json.dumps(result['info'], indent=4))