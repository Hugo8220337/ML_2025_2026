import os
import joblib
import hashlib
import pandas as pd


class CacheManager:
    def __init__(self, default_strategy='smart', module_name=None, base_dir='files/cache'):
        self.default_strategy = default_strategy
        self.module_name = module_name or "general"
        
        if module_name:
            self.cache_dir = os.path.join(base_dir, module_name)
        else:
            self.cache_dir = base_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def execute(self, task_name, func, inputs=None, params=None, strategy=None):
        effective_strategy = strategy if strategy else self.default_strategy
        
        input_hash = self._get_obj_hash(inputs)
        param_hash = self._get_obj_hash(params) if params else "default"
        
        filename = f"{task_name}_{input_hash[:10]}_{param_hash[:10]}.pkl"
        filepath = os.path.join(self.cache_dir, filename)

        if effective_strategy == 'load_only':
            if os.path.exists(filepath):
                print(f"[Cache|{self.module_name}] Loading '{task_name}' (strategy: load_only)")
                return joblib.load(filepath)
            else:
                raise FileNotFoundError(f"Cache required but not found: {filepath}")

        elif effective_strategy == 'overwrite':
            print(f"[Cache|{self.module_name}] Overwriting '{task_name}'...")
            result = func()
            joblib.dump(result, filepath)
            return result

        else: 
            if os.path.exists(filepath):
                print(f"[Cache|{self.module_name}] Hit: '{task_name}'")
                try:
                    return joblib.load(filepath)
                except Exception:
                    print(f"[Cache|{self.module_name}] Warning: Corrupt file for '{task_name}'.")
            
            print(f"[Cache|{self.module_name}] Miss: '{task_name}' (Computing...)")
            result = func()
            joblib.dump(result, filepath)
            return result

    def _get_obj_hash(self, obj):
        if obj is None: return "0"
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
            return joblib.hash(obj)
        except:
            return hashlib.md5(str(obj).encode('utf-8')).hexdigest()