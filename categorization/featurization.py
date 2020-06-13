import numpy as np
from pandas import Series
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class Featurizer:
    def fit(self, examples):
        raise NotImplementedError()

    def transform(self, examples):
        raise NotImplementedError()

    def fit_transform(self, examples):
        self.fit(examples)
        return self.transform(examples)


class SequenceFeaturizer(Featurizer):
    def __init__(self, max_vocab_size, max_name_length, max_review_length):
        super().__init__()
        self._name_featurizer = KerasTokenizer(max_vocab_size, max_name_length)
        self._review_featurizer = KerasTokenizer(max_vocab_size, max_review_length)

    def fit(self, examples):
        names, reviews = _get_name_and_review_texts(examples)
        self._name_featurizer.fit(names)
        self._review_featurizer.fit(reviews)

    def transform(self, examples):
        names, reviews = _get_name_and_review_texts(examples)
        name_features = self._name_featurizer.transform(names)
        review_features = self._review_featurizer.transform(reviews)
        return np.concatenate((name_features, review_features), axis=1)


class TfidfBowFeaturizer(Featurizer):
    """Simple featurizer that extracts text features separately from the review and business name."""

    def __init__(self):
        self.name_vectorizer = TfidfVectorizer()
        self.review_vectorizer = TfidfVectorizer()

    def fit(self, examples):
        names, reviews = _get_name_and_review_texts(examples)
        self.name_vectorizer.fit(names)
        self.review_vectorizer.fit(reviews)

    def transform(self, examples):
        names, reviews = _get_name_and_review_texts(examples)
        name_features = self.name_vectorizer.transform(names)
        review_features = self.review_vectorizer.transform(reviews)

        return hstack([name_features, review_features]).toarray()


class KerasTokenizer:
    """Simple featurizer that wraps Keras's Tokenizer. Pads and truncates each vector."""

    OOV_TOKEN = "<OOV>"

    def __init__(self, max_vocab_size, max_length):
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=self.OOV_TOKEN)
        self.vocab_size = max_vocab_size
        self.max_length = max_length

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

        self._index_to_word = {
            index: word for (word, index) in self.tokenizer.word_index.items()
        }

    def word_to_index(self, word):
        return self.tokenizer.word_index.get(word)

    def index_to_word(self, index):
        return self._index_to_word.get(index)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            sequences, maxlen=self.max_length, padding="post", truncating="post"
        )


def _get_review_texts(examples):
    return [example.review.text for example in examples]


def _get_name_texts(examples):
    return [example.business_name for example in examples]


def _get_name_and_review_texts(examples):
    return _get_name_texts(examples), _get_review_texts(examples)
