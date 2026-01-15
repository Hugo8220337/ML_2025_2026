import os
import shutil

def clear_cache_files():
    cache_dir = 'files/cache'
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
    else:
        print(f"Directory not found: {cache_dir}")

def clear_ga_logs():
    ga_logs_dir = 'files/ga_logs'
    if os.path.exists(ga_logs_dir):
        for item in os.listdir(ga_logs_dir):
            item_path = os.path.join(ga_logs_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                print(f"Deleted: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
    else:
        print(f"Directory not found: {ga_logs_dir}")

if __name__ == '__main__':
    clear_cache_files()
    clear_ga_logs()
    pass
