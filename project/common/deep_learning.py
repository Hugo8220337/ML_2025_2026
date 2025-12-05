import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical



def train_keras_dense_network(
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

def train_keras_autoencoder(
    df,
    feature_columns,
    test_size=0.2,
    random_state=42,
    encoding_dim=32, 
    activation='relu',
    optimizer='adam',
    loss='mse',
    epochs=50,
    batch_size=256,
    shuffle=True,
    verbose=1
):

    X = df[feature_columns].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    encoded = Dense(encoding_dim, activation=activation)(input_layer)
    
    decoded = Dense(input_dim, activation='sigmoid')(encoded) # sigmoid outputs 0-1 (matches scaled data)

    autoencoder = Model(input_layer, decoded)
    
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    try:
        history = autoencoder.fit(
            X_train, X_train, # X is both input and ground truth
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(X_test, X_test),
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    return {
        "model": autoencoder,
        "encoder_model": encoder,
        "history": history.history,
        "scaler": scaler
    }

def train_keras_cnn(
    df,
    target_column,
    feature_columns,
    reshape_dims, # Tuple required: (height, width, channels) e.g., (28, 28, 1)
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