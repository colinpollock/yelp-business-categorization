from tensorflow.keras.optimizers import Adam

from sklearn.linear_model import LogisticRegression
from tensorflow import keras as K


class Model:
    def __init__(self):
        self.model = None

    def fit(self, features, labels):
        raise NotImplementedError()

    def predict(self, features):
        return self.model.predict(features)


class DenseTextualModel(Model):
    def __init__(
        self,
        vocab_size,
        input_length,
        embedding_dimension,
        hidden_dimension,
        num_classes,
        learning_rate,
        epochs,
        batch_size,
    ):
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.model = K.models.Sequential(
            [
                K.layers.Embedding(
                    vocab_size, embedding_dimension, input_length=input_length
                ),
                K.layers.GlobalAvgPool1D(),
                K.layers.Dense(hidden_dimension, activation="relu"),
                K.layers.Dense(num_classes, activation="sigmoid"),
            ]
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

    def fit(
        self, train_features, train_labels, validation_features, validation_labels,
    ):
        self.model.fit(
            train_features,
            train_labels,
            validation_data=(validation_features, validation_labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )


class RnnTextualModel(Model):
    def __init__(
        self,
        vocab_size,
        input_length,
        embedding_dimension,
        rnn_dimension,
        num_classes,
        learning_rate,
        epochs,
        batch_size,
    ):
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.model = K.models.Sequential(
            [
                K.layers.Embedding(
                    vocab_size, embedding_dimension, input_length=input_length
                ),
                K.layers.LSTM(rnn_dimension),
                K.layers.Dense(num_classes, activation="sigmoid"),
            ]
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

    # TODO: this can probably just be inherited
    def fit(
        self, train_features, train_labels, validation_features, validation_labels,
    ):
        self.model.fit(
            train_features,
            train_labels,
            validation_data=(validation_features, validation_labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
