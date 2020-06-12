
from typing import NamedTuple

import numpy as np
import pandas as pd
from collections import defaultdict
from .featurization import Featurizer
from .model import Model


class Experiment(NamedTuple):
    featurizer: Featurizer
    model: Model
    train_features: np.ndarray
    validation_features: np.ndarray
    train_metrics: pd.DataFrame
    validation_metrics: pd.DataFrame

def model_experiment(featurizer, model, labelizer, train_features, train_label_sets, validation_features, validation_label_sets, probability_threshold=0.5):
    train_labels = labelizer.fit_transform(train_label_sets)
    validation_labels = labelizer.transform(validation_label_sets)

    model.fit(train_features, train_labels, validation_features, validation_labels)
    train_metrics = evaluate_model(model, train_features, train_label_sets, labelizer, probability_threshold)
    validation_metrics = evaluate_model(model, validation_features, validation_label_sets, labelizer, probability_threshold)

    print("Train Macro:")
    print(train_metrics[["precision", "recall", "f1"]].mean())
    print()
    print("Validation Macro:")
    print(validation_metrics[["precision", "recall", "f1"]].mean())
    return Experiment(featurizer, model, train_features, validation_features, train_metrics, validation_metrics)


def experiment(
    featurizer,
    model,
    labelizer,
    train_examples,
    train_label_sets,
    validation_examples,
    validation_label_sets,
):
    """
    - featurizer: a Featurizer
    """
    featurizer.fit(train_examples)
    train_features = featurizer.transform(train_examples)
    validation_features = featurizer.transform(validation_examples)

    return model_experiment(featurizer, model, labelizer, train_features, train_label_sets, validation_features, validation_label_sets)


def evaluate_model(model, features, label_sets, labelizer, probability_threshold):
    predictions = labelizer.inverse_transform(model.predict(features) > probability_threshold)
    evaluations = evaluate(label_sets, predictions)
    return evaluations


def _get_all_labels(label_sets):
    return {label for labels in label_sets for label in labels}

def group_by_result(examples, label_sets, prediction_sets):
    all_labels = _get_all_labels(label_sets)
    label_to_result_to_data = defaultdict(lambda: {result: [] for result in ('tp', 'fp', 'tn', 'fn')})

    for example, labels, predictions in zip(examples, label_sets, prediction_sets):
        for label in all_labels:
            if label in labels:
                if label in predictions:
                    result = 'tp'
                else:
                    result = 'fn'
            else:
                if label in predictions:
                    result = 'fp'
                else:
                    result = 'tn'
            label_to_result_to_data[label][result].append((example, labels, predictions))
    return label_to_result_to_data

def evaluate(label_sets, prediction_sets):
    all_labels = _get_all_labels(label_sets)

    results = defaultdict(
        lambda: {result: 0 for result in ("tp", "fp", "tn", "fn", "support")}
    )
    for labels, predictions in zip(label_sets, prediction_sets):
        for label in all_labels:
            if label in labels:
                results[label]["support"] += 1
            if label in labels:
                if label in predictions:
                    results[label]["tp"] += 1
                else:
                    results[label]["fn"] += 1
            else:
                if label in predictions:
                    results[label]["fp"] += 1
                else:
                    results[label]["tn"] += 1
    df = pd.DataFrame(results).T
    df["precision"] = (df.tp / (df.tp + df.fp)).fillna(1.0)
    df["recall"] = (df.tp / (df.tp + df.fn)).fillna(1.0)
    df["f1"] = 2 * df.precision * df.recall / (df.precision + df.recall)

    return df.sort_values("f1")
