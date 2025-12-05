import pandas as pd
import numpy as np
from common


def main():
    file_path = 'Datasets/AGNEWS/train.csv'
    df = read_csv(file_path)
    

    print(df.to_json(indent=4))
    





        

if __name__ == "__main__":
    main()
