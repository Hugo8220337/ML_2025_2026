import os
import joblib
import hashlib
import pandas as pd

# smart: load if exists, else train. overwrite: force retrain. load_only: fail if no cache.
class CacheManager:
    def __init__(self, strategy='smart', module_name=None, base_dir='files/cache'):
        self.strategy = strategy
        
        if module_name:
            self.cache_dir = os.path.join(base_dir, module_name)
        else:
            self.cache_dir = base_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def execute(self, task_name, func, inputs=None, params=None):
        input_hash = self._get_obj_hash(inputs)
        param_hash = self._get_obj_hash(params) if params else "default"
        
        filename = f"{task_name}_{input_hash[:10]}_{param_hash[:10]}.pkl"
        filepath = os.path.join(self.cache_dir, filename)

        if self.strategy == 'load_only':
            if os.path.exists(filepath):
                print(f"[CACHE] LOAD  | {task_name} (from {self.cache_dir})")
                return joblib.load(filepath)
            else:
                raise FileNotFoundError(f"Cache required but not found: {filepath}")

        elif self.strategy == 'overwrite':
            print(f"[CACHE] FORCE | {task_name} (Overwriting in {self.cache_dir})")
            result = func()
            joblib.dump(result, filepath)
            return result

        else: 
            if os.path.exists(filepath):
                try:
                    data = joblib.load(filepath)
                    print(f"[CACHE] HIT   | {task_name} (from {self.cache_dir})")
                    return data
                except Exception:
                    print(f"[CACHE] ERROR | {task_name} cache is corrupt. Re-computing...")
            
            print(f"[CACHE] MISS  | {task_name} (Computing...)")
            result = func()
            joblib.dump(result, filepath)
            return result

    def _get_obj_hash(self, obj):
        if obj is None: 
            return "0"
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
            return joblib.hash(obj)
        except:
            return hashlib.md5(str(obj).encode('utf-8')).hexdigest()