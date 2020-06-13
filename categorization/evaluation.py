
from typing import NamedTuple, List

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


import numpy as np
import pandas as pd
from .featurization import Featurizer
from .model import Model
from .data import Example

class PredictedExample(NamedTuple):
    example: Example
    label: str
    prediction: str

class Experiment(NamedTuple):
    featurizer: Featurizer
    labelizer: LabelEncoder
    model: Model

    train_examples: List[Example]
    train_features: np.ndarray
    train_labels: List[str]
    train_predictions: List[str]

    validation_examples: List[Example]
    validation_features: np.ndarray
    validation_labels: str
    validation_predictions: List[str]

    @property
    def confusion_matrix(self):
        raise NotImplementedError()

    def errors_for_label(self, input_label, use_train):
        if use_train:
            examples = self.train_examples
            labels = self.train_labels
            predictions = self.train_predictions
        else:
            examples = self.validation_examples
            labels = self.validation_labels
            predictions = self.validation_predictions

        tp = []
        fp = []
        tn = []
        fn = []

        for example, label, prediction in zip(examples, labels, predictions):
            predicted_example = PredictedExample(example, label, prediction)

            if input_label == label:
                if prediction == label:
                    tp.append(predicted_example)
                else:
                    fn.append(predicted_example)
            else:
                if input_label == prediction:
                    fp.append(predicted_example)
                else:
                    tn.append(predicted_example)

        return {'fp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


        # tp = []
        # fp = []
        # tn = []
        # fn = []

        # for predicted_example in self.predicted_examples:
        #     example, label, prediction = predicted_example
        #     if input_label == label:
        #         if prediction == label:
        #             tp.append(predicted_example)
        #         else:
        #             fn.append(predicted_example)
        #     else:
        #         if input_label == prediction:
        #             fp.append(predicted_example)
        #         else:
        #             tn.append(predicted_example)

        # return {'fp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    @property
    def errors(self):
        labels = {predicted_example.label for predicted_example in self.predicted_examples}
        return { label: self.errors_for_label(label) for label in labels }


def model_experiment(
    featurizer, model,
    train_examples, train_features, train_labels,
    validation_examples, validation_features, validation_labels,
    probability_threshold=0.5, verbose=True
):
    labelizer = LabelEncoder()
    labelizer.fit(train_labels + validation_labels)
    encoded_train_labels = labelizer.transform(train_labels)
    encoded_validation_labels = labelizer.transform(validation_labels)

    model.fit(train_features, encoded_train_labels)
    train_predictions = _evaluate_model(model, labelizer, train_features, train_labels, probability_threshold, verbose)
    validation_predictions = _evaluate_model(model, labelizer, validation_features, validation_labels, probability_threshold, verbose)

    return Experiment(
        featurizer, labelizer, model,
        train_examples, train_features, train_labels, train_predictions,
        validation_examples, validation_features, validation_labels, validation_predictions
    )


def experiment(
    featurizer,
    model,
    train_examples,
    train_labels,
    validation_examples,
    validation_labels
):
    """
    - featurizer: a Featurizer
    """
    featurizer.fit(train_examples)
    train_features = featurizer.transform(train_examples)
    validation_features = featurizer.transform(validation_examples)


    return model_experiment(featurizer, model, train_examples, train_features, train_labels, validation_examples, validation_features, validation_labels)


def _evaluate_model(model, labelizer, features, labels, probability_threshold, verbose):
    predictions = labelizer.inverse_transform(model.predict(features))
    if verbose:
        report = classification_report(labels, predictions, zero_division=1)
        print(report)

    return predictions




# def group_by_result(examples, labels, predictions):
#     by_result = {result: [] for result in ('tp', 'fp', 'tn', 'fn')}

#     for example, label, prediction in zip(examples, label_sets, prediction_sets):
#         assert label in (True, False)
#         if label is True:
#             if prediction is True:
#                 result = 'tp'
#             else:
#                 result = 'fn'
#         else:
#             if prediction is True:
#                 result = 'fp'
#             else:
#                 result = 'tn'

#         by_result[result].append(PredictedExample(example, label, prediction))
#     return by_result

# def ___evaluate(label_sets, prediction_sets):
#     all_labels = _get_all_labels(label_sets)

#     results = defaultdict(
#         lambda: {result: 0 for result in ("tp", "fp", "tn", "fn", "support")}
#     )
#     for labels, predictions in zip(label_sets, prediction_sets):
#         for label in all_labels:
#             if label in labels:
#                 results[label]["support"] += 1
#             if label in labels:
#                 if label in predictions:
#                     results[label]["tp"] += 1
#                 else:
#                     results[label]["fn"] += 1
#             else:
#                 if label in predictions:
#                     results[label]["fp"] += 1
#                 else:
#                     results[label]["tn"] += 1
#     df = pd.DataFrame(results).T
#     df["precision"] = (df.tp / (df.tp + df.fp)).fillna(1.0)
#     df["recall"] = (df.tp / (df.tp + df.fn)).fillna(1.0)
#     df["f1"] = 2 * df.precision * df.recall / (df.precision + df.recall)

#     return df.sort_values("f1")
