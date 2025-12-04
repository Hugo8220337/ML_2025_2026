import csv
import json
import os
import pandas as pd


def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                data.append(row)
        
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
            
        print("Done!")
        
        
    except Exception as e:
        print(f"An error occurred during conversion: {e}")



def read_csv(csv_file_path):
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' was not found.")
        return None

    try:
        df = pd.read_csv(csv_file_path)
        return df
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None