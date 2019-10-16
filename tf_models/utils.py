from typing import Any, List, BinaryIO
import os
import pickle
import logging
from typing import Tuple

from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

logging.getLogger().setLevel(logging.INFO)


def build_mlp_model(
    input_dim: int, layers: List[int], output_dim: int, dropout_rate: float = 0
) -> Model:
    """Return a MLP model with given parameter settings"""
    # Input layer
    X = Input(shape=(input_dim,))

    # Hidden layer(s)
    H = X
    for layer in layers:
        H = Dense(layer, activation="relu")(H)
        if dropout_rate > 0:
            H = Dropout(rate=dropout_rate)(H)

    # Output layer
    activation_func = "softmax" if output_dim > 1 else "sigmoid"

    Y = Dense(output_dim, activation=activation_func)(H)
    return Model(inputs=X, outputs=Y)


def train(
    training_corpus: str, pos_label: str = "", root: str = ""
) -> Tuple[Model, History, TfidfVectorizer]:
    """
    Train a MLP model on given `training_corpus`. For simple demo purpose, I didn't expose parameters for
    the model for tuning here. Feel free to play with these parameters and build a larger corpus for better model.
    """
    # Load data from corpus file
    df = pd.read_csv(os.path.join(root, training_corpus))
    train_text, train_label = df["text"], df["label"]

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_x = tfidf_vectorizer.fit_transform(train_text)
    train_y = train_label.apply(lambda x: 1 if x == pos_label else 0)

    print(train_y)

    input_dim = len(tfidf_vectorizer.vocabulary_)

    # Build a mlp model for binary classification
    mlp_model = build_mlp_model(
        input_dim=input_dim, layers=[10], output_dim=1, dropout_rate=0.2
    )

    print(mlp_model.summary())

    mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    CPK_PATH = os.path.join(root, "model_cpk.hdf5")  # path to store checkpoint
    model_cpk_hook = ModelCheckpoint(
        CPK_PATH, monitor="acc", save_best_only=True  # Only keep the best model
    )

    history = mlp_model.fit(train_x, train_y, epochs=100, callbacks=[model_cpk_hook])

    mlp_model.load_weights(CPK_PATH)
    os.remove(CPK_PATH)

    return mlp_model, history, tfidf_vectorizer


def plot_history(his: History, metrics: List[str]):
    """
    Given a history object returned from `fit` and the name of metrics,
    plot the curve of metrics against number of epochs.
    """
    for metric in metrics:
        plt.plot(his.history[metric], label=metric)
    plt.legend()


def save_model(
    model: Model, vectorizer, name: str, root: str = ""
) -> Tuple[str, str, str]:
    """Save the trained model (structure, weights) and vectorizer to files."""
    json_file, h5_file, vec_file = (
        os.path.join(root, "{}.{}".format(name, ext)) for ext in ("json", "h5", "pkl")
    )

    with open(json_file, "w") as fp:
        fp.write(model.to_json())
    model.save_weights(h5_file)

    with open(vec_file, "wb") as bfp:  # type: BinaryIO
        pickle.dump(vectorizer, bfp)

    logging.info("Model is being written to {}".format(root + "/"))

    return json_file, h5_file, vec_file


def load_model(name: str, root: str = "") -> Tuple[Model, Any]:
    """Load the trained model (structure, weights) and vectorizer from files."""
    json_file, h5_file, vec_file = (
        os.path.join(root, "{}.{}".format(name, ext)) for ext in ("json", "h5", "pkl")
    )

    with open(json_file) as fp:
        model = model_from_json(fp.read())  # type: Model
    model.load_weights(h5_file)

    with open(vec_file, "rb") as bfp:  # type: BinaryIO
        vectorizer = pickle.load(bfp)

    logging.info("Model loaded from {}".format(root + "/"))

    return model, vectorizer
