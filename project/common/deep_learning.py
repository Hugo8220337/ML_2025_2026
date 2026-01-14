import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .metrics import get_classification_metrics 

def train_dense_network(
    df,
    target_column,
    feature_columns,
    test_size=0.2,
    random_state=42,
    hidden_layers=[64, 32],  
    activation='relu',
    output_activation='softmax', 
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'],
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
):
    
    if target_column not in df.columns:
        print(f"Error: Target '{target_column}' not found.")
        return None
    
    
    X = df[feature_columns].values
    y = df[target_column].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(X_train.shape[1],)))
    
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
     
    num_classes = len(np.unique(y))
    output_units = 1 if num_classes == 2 and output_activation == 'sigmoid' else num_classes
    
    model.add(Dense(output_units, activation=output_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    
    loss_val, accuracy_val = model.evaluate(X_test, y_test, verbose=0)

    return {
        "model": model,
        "history": history.history,
        "metrics": {"test_loss": loss_val, "test_accuracy": accuracy_val},
        "scaler": scaler
    }


def train_cnn(
    X, 
    y, 
    X_test=None, 
    y_test=None, 
    texts=None, 
    texts_test=None, 
    vocab_size=5000, 
    embedding_dim=50, 
    max_length=30, 
    filters=64, 
    kernel_size=3, 
    dense_units=32, 
    dropout=0.2, 
    learning_rate=0.001, 
    batch_size=32, 
    epochs=5, 
    **kwargs
):
    if texts is None:
        print("Error: CNN requires 'texts' (raw tokens) to be passed.")
        return None

    if X_test is None or y_test is None:
        if texts is not None:
            X_train_raw, X_test_raw, y_train, y_test, texts_train, texts_test = train_test_split(
                X, y, texts, test_size=0.2, random_state=42
            )
        else:
             return None
    else:
        y_train = y
        texts_train = texts

    if texts_test is None:
        texts_test = []

    train_strs = [" ".join(t) if isinstance(t, list) else str(t) for t in texts_train]
    test_strs = [" ".join(t) if isinstance(t, list) else str(t) for t in texts_test]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_strs)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_strs), maxlen=max_length, padding='post', truncating='post')
    
    validation_data = None
    if len(test_strs) > 0:
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_strs), maxlen=max_length, padding='post', truncating='post')
        validation_data = (X_test_seq, y_test)

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=['accuracy']
    )
    
    try:
        history = model.fit(
            X_train_seq, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0 
        )
    except Exception as e:
        print(f"CNN Training Error: {e}")
        return None

    metrics = {}
    y_pred = None
    
    if validation_data:
        y_prob = model.predict(X_test_seq, verbose=0)
        y_pred = (y_prob > 0.5).astype("int32")
        
        if y_prob.shape[1] == 1:
            y_prob_combined = np.hstack([1 - y_prob, y_prob])
        else:
            y_prob_combined = y_prob

        metrics = get_classification_metrics(y_test, y_pred, y_prob_combined)

    return {
        "model": model,
        "metrics": metrics,
        "history": history.history,
        "tokenizer": tokenizer,
        "test_data": {"y_test": y_test, "predictions": y_pred} if validation_data else {}
    }

def train_dense_autoencoder(X, y, X_test=None, y_test=None, test_size=0.2, random_state=42, **kwargs):
    encoding_dim = kwargs.get('encoding_dim', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    batch_size = kwargs.get('batch_size', 64)
    epochs = kwargs.get('epochs', 5)
    dropout = kwargs.get('dropout', 0.1)
    threshold_sigma = kwargs.get('threshold_sigma', 2.0)

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X_train, y_train = X, y

    if hasattr(X_train, 'toarray'):
        real_indices = np.where(y_train == 0)[0]
        X_train_clean = X_train[real_indices].toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_clean = X_train[y_train == 0]
        X_test_dense = X_test

    input_dim = X_train_clean.shape[1]

    scaler = MaxAbsScaler()
    X_train_clean = scaler.fit_transform(X_train_clean) 
    if hasattr(X_test_dense, 'toarray'): 
         X_test_dense = scaler.transform(X_test_dense)
    else:
         X_test_dense = scaler.transform(X_test_dense)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(encoding_dim, activation='relu'),
        Dropout(dropout),
        Dense(128, activation='relu'),
        Dense(input_dim, activation='linear')   
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    try:
        model.fit(X_train_clean, X_train_clean, epochs=epochs, batch_size=batch_size, verbose=0)
    except Exception as e:
        print(f"Error training Dense AE: {e}")
        return None

    train_preds = model.predict(X_train_clean, verbose=0)
    train_mse = np.mean(np.power(X_train_clean - train_preds, 2), axis=1)
    
    threshold = np.percentile(train_mse, 95)
    
    model.threshold_ = threshold

    test_preds_raw = model.predict(X_test_dense, verbose=0)
    test_mse = np.mean(np.power(X_test_dense - test_preds_raw, 2), axis=1)
    
    y_pred_binary = (test_mse > threshold).astype(int)

    metrics = get_classification_metrics(y_test, y_pred_binary)

    return {
        "model": model, 
        "metrics": metrics,
        "test_data": {"y_test": y_test, "predictions": y_pred_binary},
        "threshold": threshold
    }


def train_embedding_autoencoder(
    X, 
    y, 
    X_test=None, 
    y_test=None, 
    texts=None,
    texts_test=None, 
    test_size=0.2, 
    random_state=42, 
    vocab_size=5000, 
    embedding_dim=50, 
    max_length=100,
    encoding_dim=32,
    batch_size=64,
    epochs=10,
    **kwargs
):
    if texts is None:
        print("Error: Embedding Autoencoder requires 'texts' (raw tokens).")
        return None

    if texts_test is None or len(texts_test) == 0:
        if len(texts) == len(y):
             pass 
        else:
             print("Warning: Length mismatch between X and texts")
    
    real_indices = [i for i, label in enumerate(y) if label == 0]
    real_texts = [texts[i] for i in real_indices]

    train_strs = [" ".join(t) if isinstance(t, list) else str(t) for t in real_texts]
    test_strs = [" ".join(t) if isinstance(t, list) else str(t) for t in (texts_test if texts_test else [])]
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_strs)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_strs), maxlen=max_length, padding='post', truncating='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_strs), maxlen=max_length, padding='post', truncating='post')

    X_train_target = tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences(train_strs), mode='binary')
    X_test_target = tokenizer.sequences_to_matrix(tokenizer.texts_to_sequences(test_strs), mode='binary')

    inputs = Input(shape=(max_length,))
    
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = GlobalAveragePooling1D()(x)
    encoded = Dense(encoding_dim, activation='relu')(x)
    
    decoded = Dense(vocab_size, activation='sigmoid')(encoded)
    
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    try:
        model.fit(X_train_seq, X_train_target, epochs=epochs, batch_size=batch_size, verbose=0)
    except Exception as e:
        print(f"Error training Emb AE: {e}")
        return None

    def calculate_error(model, sequences, targets):
        preds = model.predict(sequences, verbose=0)
        epsilon = 1e-7
        preds = np.clip(preds, epsilon, 1. - epsilon)
        bce = - (targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
        return np.mean(bce, axis=1)

    train_loss = calculate_error(model, X_train_seq, X_train_target)
    
    threshold = np.percentile(train_loss, 95)
    model.threshold_ = threshold
    model.tokenizer_ = tokenizer


    test_loss = calculate_error(model, X_test_seq, X_test_target)
    y_pred_binary = (test_loss > threshold).astype(int)

    metrics = get_classification_metrics(y_test, y_pred_binary)

    return {
        "model": model,
        "metrics": metrics,
        "test_data": {"y_test": y_test, "predictions": y_pred_binary},
        "threshold": threshold,
        "tokenizer": tokenizer
    }