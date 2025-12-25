import json
from common.tools import read_csv
from common.nlp import tfidf_vectorize
from evolutionary_model_selection.ems import ems
from common.cache import CacheManager

_topics = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def topic_classification(data_strategy='smart', model_strategy='smart', models=['logistic_regression'], options=None):
    cm = CacheManager(module_name="topic_classification")
    
    df = read_csv('datasets/AGNEWS/train.csv')
    df['data'] = df['Title'] + ' ' + df['Description']
    df = df.drop(columns=['Title', 'Description'])
    

    def run_vectorization():
        X_new, vec_new = tfidf_vectorize(df, col_name='data')
        return X_new, vec_new


    X, vectorizer= cm.execute(
        task_name="preprocessing",
        func=run_vectorization,
        inputs=df['data'],
        strategy=data_strategy
    )



    def run_ems_training():
        return ems(X, df['Class Index'], models, report=True, options=options)


    result = cm.execute(
        task_name="model_selection",
        func=run_ems_training,
        inputs=X,
        params={'models': models},
        strategy=model_strategy
    )

    print(json.dumps(result['info'], indent=4))