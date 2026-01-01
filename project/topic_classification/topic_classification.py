import json
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager

_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification(models=['kmeans', 'dbscan', 'gmm'], options=None):
    cache = CacheManager(module_name='topic_classification')
    file_path = 'datasets/allthenews/all-the-news-2-1.csv'

    def preprocessing(file_path):
        df = read_csv(file_path, usecols=['title', 'article'])
        df['data'] = df['title'] + ' ' + df['article']
        df = df.drop(columns=['title', 'article'])
        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)


    X = df['data']


    result = ems(X, models, report=True, options=options)
    print(json.dumps(result['info'], indent=4))