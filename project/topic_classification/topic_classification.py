import pandas as pd
import numpy as np
from common.tools import read_csv



def topic_classification():
    file_path = 'datasets/AGNEWS/train.csv'
    df = read_csv(file_path)
    

    print(df.to_json(indent=4))
    





