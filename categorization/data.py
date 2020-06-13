import json
import random
import typing
from datetime import datetime
from collections import defaultdict

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

    # Note that this is a single review because we're classifying based on just one review.
    review: Review


def load_examples(business_ids, accepted_categories, reviews_per_business):
    """Return a list of `Example`s, one for each business.

    Args:
    - business_ids: only return businesses from this set
    - accepted_categories: only return businesses that have one of htese categories
    - reviews_per_business: the number of reviews to sample from each business, each of
      which becomes an Example.

    Note that exactly one of business's categories must be in `accepted_categories`.
    """

    all_reviews = load_reviews(business_ids)
    business_id_to_reviews = defaultdict(list)
    for review in all_reviews:
        business_id_to_reviews[review.business_id].append(review)

    examples = []
    labels = []
    for business in _load_json(BUSINESS_FILEPATH):
        if business_ids is not None and business["business_id"] not in business_ids:
            continue

        categories = _parse_categories_string(business["categories"])
        valid_categories = accepted_categories.intersection(categories)
        if len(valid_categories) != 1:
            continue
        (valid_category,) = valid_categories

        reviews = business_id_to_reviews[business["business_id"]]
        if len(reviews) == 0:
            continue

        for review in random.sample(reviews, reviews_per_business):
            examples.append(
                Example(
                    business["business_id"],
                    business["name"],
                    business["review_count"],
                    business["stars"],
                    business["city"],
                    business["state"],
                    review,
                )
            )

            labels.append(valid_category)

    return examples, labels


def load_reviews(business_ids):
    """Return a list of Review objects for all reviews from the given business IDs."""
    reviews = []
    for review in _load_json(REVIEW_FILEPATH):
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


# def load_reviews_df(business_ids, limit=None):
#     records = []
#     for review in _load_json(REVIEW_FILEPATH):
#         # TODO: can optimize this with a regex that checks for
#         # any of the valid business IDs before loading the json.
#         # E.g. "business_id":"ujmEBvifdJM6h6RLv4wQIg|..."
#         business_id = review["business_id"]
#         if business_id in business_ids:
#             records.append(review)

#         if limit is not None and len(records) > limit:
#             print("breaking:", limit, len(records))
#             break

#     df = pd.DataFrame.from_records(records)
#     return df
#     df["date"] = pd.to_datetime(df["date"])
#     return df


# def load_users_df(user_ids):
#     records = []
#     for user in _load_json(USER_FILEPATH):
#         if user["user_id"] in user_ids:
#             records.append(user)

#     return pd.DataFrame.from_records(records)


def load_business_df():
    records = []
    for business in _load_json(BUSINESS_FILEPATH):
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


def _load_json(filepath):
    with open(filepath, "r") as fh:
        for line in fh:
            yield json.loads(line)
