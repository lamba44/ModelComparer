import os
import time
from datetime import datetime
import numpy as np

# Keras / TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
except Exception as e:
    raise ImportError(
        "tensorflow is required for lstm trainer. Install with `pip install tensorflow`"
    ) from e

from sklearn.preprocessing import LabelEncoder
from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _model_filename(base_name: str, task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{task}_{ts}.keras"


def _reshape_for_sequence(X):
    """
    Convert 2D array (n_samples, n_features) -> (n_samples, timesteps=n_features, 1)
    """
    X = np.asarray(X, dtype=np.float32)
    return X.reshape((X.shape[0], X.shape[1], 1))


def train(X_train, y_train, X_val, y_val, config=None):

    if config is None:
        config = {}

    task = config.get("task")
    if task not in ("classification", "regression"):
        raise ValueError("config['task'] must be 'classification' or 'regression'")

    return_model = bool(config.get("return_model", False))

    epochs = int(config.get("epochs", 30))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 0.001))
    lstm_units = int(config.get("lstm_units", 64))
    dense_units = int(config.get("dense_units", 64))
    patience = int(config.get("patience", 5))
    model_dir = config.get("model_dir", "models")
    random_seed = int(config.get("random_seed", 42))
    base_name = config.get("base_name", "lstm")

    _ensure_dir(model_dir)

    # reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Convert to numpy
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # reshape for LSTM: (samples, timesteps, 1)
    X_train_seq = _reshape_for_sequence(X_train)
    X_val_seq = _reshape_for_sequence(X_val)
    input_shape = X_train_seq.shape[1:]  # (timesteps, 1)

    # handle classification target encoding
    label_encoder = None
    num_classes = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_val_enc = label_encoder.transform(y_val)
        unique_labels = np.unique(y_train_enc)
        num_classes = len(unique_labels)
        if num_classes <= 2:
            loss = "binary_crossentropy"
            final_activation = "sigmoid"
        else:
            loss = "sparse_categorical_crossentropy"
            final_activation = "softmax"
    else:
        y_train_enc = y_train.astype(np.float32)
        y_val_enc = y_val.astype(np.float32)
        loss = "mse"
        final_activation = None

    # build model
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=False)(inp)
    x = layers.Dense(dense_units, activation="relu")(x)

    if task == "regression":
        out = layers.Dense(1, activation="linear")(x)
        model = models.Model(inputs=inp, outputs=out)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
    else:
        if final_activation == "sigmoid":
            out = layers.Dense(1, activation="sigmoid")(x)
            model = models.Model(inputs=inp, outputs=out)
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=[])
        else:
            out = layers.Dense(num_classes, activation="softmax")(x)
            model = models.Model(inputs=inp, outputs=out)
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=[])

    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # train
    t0 = time.time()
    history = model.fit(
        X_train_seq,
        y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    train_time = time.time() - t0

    # predictions
    if task == "regression":
        y_pred = model.predict(X_val_seq).flatten()
        y_proba_for_metrics = None
    else:
        probs = model.predict(X_val_seq)
        if num_classes <= 2:
            y_proba_for_metrics = probs.flatten()
            y_pred = (y_proba_for_metrics >= 0.5).astype(int)
        else:
            y_proba_for_metrics = probs
            y_pred = np.argmax(probs, axis=1)

        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred)

    # metrics
    if task == "regression":
        computed_metrics = metrics_module.compute_metrics(
            y_val, y_pred, task="regression"
        )
    else:
        computed_metrics = metrics_module.compute_metrics(
            y_val, y_pred, task="classification", y_proba=y_proba_for_metrics
        )

    # build meta: include classes and label_encoder for downstream combos
    meta = {}
    if task == "classification":
        if label_encoder is not None:
            # label_encoder holds mapping from original labels -> ints
            meta["classes"] = label_encoder.classes_
            meta["label_encoder"] = label_encoder

    # save
    filename = _model_filename(base_name, task)
    model_path = os.path.join(model_dir, filename)
    model.save(model_path, include_optimizer=False)

    model_size = os.path.getsize(model_path)
    param_count = model.count_params()

    result = {
        "model_name": f"LSTM ({'Classifier' if task == 'classification' else 'Regressor'})",
        "model_path": model_path,
        "metrics": computed_metrics,
        "train_time": train_time,
        "model_size": model_size,
        "extra": {
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "epochs_trained": len(history.history.get("loss", [])),
            "param_count": param_count,
        },
    }

    if return_model:
        return result, model, meta

    return result
