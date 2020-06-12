from pandas import Series
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Featurizer:
    def fit(self, examples):
        raise NotImplementedError()

    def transform(self, examples):
        raise NotImplementedError()


class NameAndReviewTextFeaturizer(Featurizer):
    def __init__(self, max_vocab_size, max_length):
        super().__init__()
        self._text_featurizer = TextFeaturizer(max_vocab_size, max_length)

    def fit(self, examples):
        texts = (make_text_from_example(example) for example in examples)
        self._text_featurizer.fit(texts)

    def transform(self, examples):
        texts = (make_text_from_example(example) for example in examples)
        return self._text_featurizer.transform(texts)


def make_text_from_example(example):
    """Glues together the business name with all of the reviews into a long string."""
    return example.business_name + "\n".join(review.text for review in example.reviews)


from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeaturizer(Featurizer):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, examples):
        texts = [make_text_from_example(example) for example in examples]
        self.vectorizer.fit(texts)

    def transform(self, examples):
        texts = [make_text_from_example(example) for example in examples]
        return self.vectorizer.transform(texts)


class TextFeaturizer:
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
