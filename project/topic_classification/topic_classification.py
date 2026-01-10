import json
from common.tools import read_csv
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager

def topic_classification(models=['kmeans', 'hdbscan', 'gmm'], options='default'):
    cache = CacheManager(module_name='topic_classification')
    file_path = 'datasets/allthenews/all-the-news-2-1.csv'

    def preprocessing(file_path):
        df = read_csv(file_path, usecols=['title', 'article'])
        df = df.dropna()
        df = df.sample(n=100000, random_state=42)
        df['data'] = df['title'] + ' ' + df['article']
        df = df.drop(columns=['title', 'article'])
        return df



    df = cache.execute(task_name='read_csv',
                       func=lambda: preprocessing(file_path),
                       inputs=file_path)

    
    X = df['data']


    result = cache.execute(task_name='ems',
                           func=lambda: ems(
                            X, 
                            models=models,
                            report=True, 
                            options=options, 
                            reduction='nmf', 
                            vectorizer_type='hashing'),
                           inputs=X)

    # print(json.dumps(result['info'], indent=4))
    with open('log.txt', 'w') as f:
        print(result, file=f)