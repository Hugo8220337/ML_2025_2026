import streamlit as st
import sys
import io
import contextlib
import time
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASKS = {
    "Fake News Detection": {
        "module": "fake_news_detection.fake_news_detection",
        "func": "fake_news_detection",
        "models": ['logistic_regression', 'neural_network', 'svm', 'random_forest', 'xgboost', 'cnn', 'naive_bayes'],
        "defaults": ['logistic_regression', 'random_forest']
    },
    "Topic Classification": {
        "module": "topic_classification.topic_classification",
        "func": "topic_classification",
        "models": ['nmf', 'kmeans', 'hdbscan', 'gmm'],
        "defaults": ['nmf']
    },
    "Anomaly Detection": {
        "module": "anomaly_detection.anomaly_detection",
        "func": "anomaly_detection",
        "models": ['random_forest', 'svm', 'neural_network', 'isolation_forest', 'one_class_svm', 'dense_autoencoder', 'embedding_autoencoder'],
        "defaults": ['isolation_forest', 'one_class_svm']
    },
    "Stance Detection": {
        "module": "stance_detection.stance_detection",
        "func": "stance_detection",
        "models": ['svm', 'random_forest', 'xgboost', 'neural_network'],
        "defaults": ['svm', 'random_forest']
    },
    "Clickbait Detection": {
        "module": "clickbait_detection.clickbait_detection",
        "func": "clickbait_detection",
        "models": ['xgboost', 'cnn', 'svm', 'random_forest', 'neural_network'],
        "defaults": ['xgboost', 'cnn']
    }
}

class StreamlitCapture(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def run_task(task_name, selected_models, output_placeholder):    
    task_info = TASKS[task_name]
    module_name = task_info["module"]
    function_name = task_info["func"]
    
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except ImportError as e:
        output_placeholder.error(f"Failed to import module {module_name}: {e}")
        return
    except AttributeError:
        output_placeholder.error(f"Function {function_name} not found in {module_name}")
        return

    output_buffer = []
    
    class RealTimeCapture:
        def __init__(self, placeholder):
            self.placeholder = placeholder
            self.buffer = ""
            self.original_stdout = sys.stdout

        def write(self, text):
            self.buffer += text
            self.original_stdout.write(text)
            self.placeholder.code(self.buffer, language="bash")

        def flush(self):
            self.original_stdout.flush()

    output_placeholder.info(f"Starting {task_name}...")
    
    original_stdout = sys.stdout
    capturer = RealTimeCapture(output_placeholder)
    sys.stdout = capturer
    
    try:
        result = func(models=selected_models)
        
        sys.stdout = original_stdout
        output_placeholder.success("Execution Finished Successfully!")
         
        if isinstance(result, tuple):
             result_data = result[0]
        else:
             result_data = result
             
        if isinstance(result_data, dict) and 'best_model' in result_data:
            st.metric("Best Model", result_data['best_model'])
            st.json(result_data.get('metrics', {}))

    except Exception as e:
        sys.stdout = original_stdout
        output_placeholder.error(f"Error during execution: {e}")
        import traceback
        st.code(traceback.format_exc())


def main():
    st.set_page_config(page_title="Inference Motor", page_icon="⚙️", layout="wide")
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50; 
            color: white;
            font-weight: bold;
        }
        .console-box {
            background-color: #0e1117;
            color: #00ff00;
            font-family: 'Courier New', Courier, monospace;
            padding: 10px;
            border-radius: 5px;
            height: 400px;
            overflow-y: scroll;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚙️ Inference Motor")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        selected_task = st.selectbox(
            "Select Task",
            options=list(TASKS.keys())
        )
        
        task_config = TASKS[selected_task]
        available_models = task_config["models"]
        default_defaults = [m for m in task_config["defaults"] if m in available_models]
        
        selected_models = st.multiselect(
            "Select Models",
            options=available_models,
            default=default_defaults,
            help="Choose the models to train."
        )
        
        if st.button("Train Model", type="primary"):
            if not selected_models:
                st.warning("Please select at least one model.")
            else:
                st.session_state['run_task'] = True
                st.session_state['task_name'] = selected_task
                st.session_state['models'] = selected_models

    with col2:
        st.subheader("Console Output")
        output_container = st.empty()
        
        if 'run_task' not in st.session_state:
             output_container.code("#", language="bash")
        
        if st.session_state.get('run_task', False):
            run_task(
                st.session_state['task_name'],
                st.session_state['models'],
                output_container
            )
            st.session_state['run_task'] = False

if __name__ == "__main__":
    main()
