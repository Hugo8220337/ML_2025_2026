import os
import joblib
import hashlib
import pandas as pd
import shutil


class CacheManager:
    def __init__(self, module_name=None, base_dir='files/cache'):
        env_strategy = os.getenv("CACHE_STRATEGY")
        self.default_strategy = env_strategy or 'smart'
        self.module_name = module_name or "general"
        self.base_dir = base_dir
        
        if module_name:
            self.cache_dir = os.path.join(base_dir, module_name)
        else:
            self.cache_dir = base_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def execute(self, task_name, func, inputs=None, params=None, strategy=None):
        strategy = strategy or self.default_strategy
        
        input_hash = self._get_obj_hash(inputs)
        param_hash = self._get_obj_hash(params) if params else "default"
        
        filename = f"{task_name}_{input_hash[:10]}_{param_hash[:10]}.pkl"
        filepath = os.path.join(self.cache_dir, filename)

        if strategy == 'load_only':
            if os.path.exists(filepath):
                print(f"[Cache|{self.module_name}] Loading '{task_name}' (strategy: load_only)")
                return joblib.load(filepath)
            else:
                raise FileNotFoundError(f"Cache required but not found: {filepath}")

        elif strategy == 'overwrite':
            print(f"[Cache|{self.module_name}] Overwriting '{task_name}'...")
            result = func()
            joblib.dump(result, filepath)
            return result

        else: 
            if os.path.exists(filepath):
                print(f"[Cache|{self.module_name}] Loading: '{task_name}'")
                try:
                    return joblib.load(filepath)
                except Exception:
                    print(f"[Cache|{self.module_name}] Warning: Corrupt file for '{task_name}'.")
            
            result = func()
            joblib.dump(result, filepath)
            return result

    def clear(self, module_name=None):
        target = self.base_dir
        if module_name:
            target = os.path.join(self.base_dir, module_name)

        if os.path.exists(target):
            print(f"[Cache|{self.module_name}] Deleting: {target}")
            shutil.rmtree(target)

    def _get_obj_hash(self, obj):
        if obj is None: return "0"
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
            return joblib.hash(obj)
        except:
            return hashlib.md5(str(obj).encode('utf-8')).hexdigest()