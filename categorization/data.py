from datetime import datetime
from collections import defaultdict
import json
import typing

import pandas as pd


BUSINESS_FILEPATH = "data/business.json"
REVIEW_FILEPATH = "data/review.json"
USER_FILEPATH = "data/user.json"

CATEGORIES_FILEPATH = "data/categories.json"


class Review(typing.NamedTuple):
    review_id: str
    user_id: str
    business_id: str
    stars: int
    useful: int
    funny: int
    cool: int
    text: str
    date: datetime


class Example(typing.NamedTuple):
    """Represents a business with all the metadata potentially useful in classification."""

    business_id: str
    business_name: str
    review_count: int
    stars: float
    state: str
    city: str
    reviews: typing.List[Review]


def load_examples(business_ids, min_reviews=0, accepted_categories=None):
    """Return a list of `Example`s, one for each business.

    - min_reviews: mininum reviews a business must have to be included.
    """
    if accepted_categories is None:
        accepted_categories = frozenset()

    all_reviews = load_reviews(business_ids)
    business_id_to_reviews = defaultdict(list)
    for review in all_reviews:
        business_id_to_reviews[review.business_id].append(review)

    examples = []
    labels = []
    for business in load_json(BUSINESS_FILEPATH):
        if business["business_id"] not in business_ids:
            continue

        if business["review_count"] < min_reviews:
            continue

        categories = _parse_categories_string(business["categories"])
        valid_categories = accepted_categories.intersection(categories)
        if not valid_categories:
            continue

        reviews = business_id_to_reviews[business["business_id"]]
        examples.append(
            Example(
                business["business_id"],
                business["name"],
                business["review_count"],
                business["stars"],
                business["city"],
                business["state"],
                reviews,
            )
        )

        labels.append(valid_categories)

    return examples, labels


def load_reviews(business_ids):
    """Return a list of Review objects for all reviews from the given business IDs."""
    reviews = []
    for review in load_json(REVIEW_FILEPATH):
        if review["business_id"] not in business_ids:
            continue

        reviews.append(
            Review(
                review["review_id"],
                review["user_id"],
                review["business_id"],
                review["stars"],
                review["useful"],
                review["funny"],
                review["cool"],
                review["text"],
                review["date"],
            )
        )

    return reviews


def load_reviews_df(business_ids, limit=None):
    records = []
    for review in load_json(REVIEW_FILEPATH):
        # TODO: can optimize this with a regex that checks for
        # any of the valid business IDs before loading the json.
        # E.g. "business_id":"ujmEBvifdJM6h6RLv4wQIg|..."
        business_id = review["business_id"]
        if business_id in business_ids:
            records.append(review)

        if limit is not None and len(records) > limit:
            print("breaking:", limit, len(records))
            break

    df = pd.DataFrame.from_records(records)
    return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_json(filepath):
    with open(filepath, "r") as fh:
        for line in fh:
            yield json.loads(line)


def load_users_df(user_ids):
    records = []
    for user in load_json(USER_FILEPATH):
        if user["user_id"] in user_ids:
            records.append(user)

    return pd.DataFrame.from_records(records)


def load_business_df():
    records = []
    for business in load_json(BUSINESS_FILEPATH):
        if business["is_open"] == 1:
            records.append(
                {
                    "business_id": business["business_id"],
                    "business_name": business["name"],
                    "review_count": business["review_count"],
                    "stars": business["stars"],
                    "state": business["state"],
                    "city": business["city"],
                    "categories": _parse_categories_string(business["categories"]),
                }
            )

    business_df = pd.DataFrame.from_records(records)
    return business_df


def _parse_categories_string(category_str):
    """Return a list of categories (as strings) from a CSV of categories."""
    if category_str is None:
        return []

    return [category.strip() for category in category_str.split(",")]


class CategoryTree:
    def __init__(self):
        with open(CATEGORIES_FILEPATH, "r") as fh:
            self._categories = json.load(fh)

    @property
    def root_categories(self):
        """Return a set of category titles (strings) for all root categories."""
        return {
            category["title"]
            for category in self._categories
            if not category["parents"]
        }
