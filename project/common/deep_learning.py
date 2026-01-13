import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.keras.utils import to_categorical
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
    df,
    target_column,
    feature_columns,
    reshape_dims, 
    test_size=0.2,
    random_state=42,
    filters=32,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    activation='relu',
    dense_units=64,
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
):

    if target_column not in df.columns:
        print(f"Error: Target '{target_column}' not found.")
        return None
        
    X_flat = df[feature_columns].values
    y = df[target_column].values

    try:
        X_flat = X_flat / 255.0 
        X_reshaped = X_flat.reshape(-1, reshape_dims[0], reshape_dims[1], reshape_dims[2])
    except ValueError as e:
        print(f"Error reshaping data. Check if feature columns match {reshape_dims}. Details: {e}")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=random_state
    )

    model = Sequential()
    
    model.add(Conv2D(filters, kernel_size, activation=activation, input_shape=reshape_dims))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Conv2D(filters * 2, kernel_size, activation=activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Flatten())
    model.add(Dense(dense_units, activation=activation))
    
    num_classes = len(np.unique(y))
    model.add(Dense(num_classes, activation='softmax'))

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
        "metrics": {"test_loss": loss_val, "test_accuracy": accuracy_val}
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
    
    threshold = np.mean(train_mse) + (threshold_sigma * np.std(train_mse))
    
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