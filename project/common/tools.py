import csv
import json
import os
import pandas as pd
import sys
import time
import threading
import itertools


class load:
    """
    A context manager for showing a loading animation in the console.
    """
    def __init__(self, text="Loading...", speed=0.1):
        self.text = text
        self.speed = speed
        self.stop_event = threading.Event()
        self.animation_thread = None
        self.spinner = itertools.cycle(['|', '/', '-', '\'])

    def _animate(self):
        """The animation function that runs in a separate thread."""
        while not self.stop_event.is_set():
            sys.stdout.write(f'\r{self.text} {next(self.spinner)}')
            sys.stdout.flush()
            time.sleep(self.speed)
        # Clear the line after the animation stops
        sys.stdout.write('\r' + ' ' * (len(self.text) + 2) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        """Starts the loading animation."""
        self.animation_thread = threading.Thread(target=self._animate)
        self.animation_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the loading animation."""
        self.stop_event.set()
        if self.animation_thread:
            self.animation_thread.join()


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
