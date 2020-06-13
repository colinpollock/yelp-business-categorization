Yelp Business Categorization
============================

I'm using the Yelp Academic Dataset [1] to play with Keras and TensorFlow. The prediction problem is
to assign categories (like "Restaurant", "Sushi Bar", "Hardware Store", etc.) to Yelp businesses. I'm
making the following simplifying assumptions:
* Only a business's root category is used. For example, if a business has the categories "Sushi", "Japanese", and
  "Restaurant" I'd only be considering "Restaurant". This makes the problem much simpler since it's much easier
  to differentiate restaurants and plumbers than e.g. Mexican food from Tex Mex.
* I'm only considering businesses with a single root category. This allos me to turn a multilabel problem into
  a much simpler multiclass problem.
* I'm classifying the business based on just a single review. This makes the problem harder since some proportion
  of reviews will contain no hints of the category, but this is useful since in the real world businesses with
  many reviews are likely to already have accurate categories.


Setup
-----
1. Get the data (see section below Data Setup)
2. `$ python3 -m venv .venv`
3. `$ source .venv/bin/activate`
4. `$ pip install --upgrade pip`
5. `pip install -r requirements.txt`
6. `jupyter notebook --port 8888`
7. Open localhost:8888 in your browser and open to the notebook Categorization


Data Setup
----------
1. Download the Yelp Academic Dataset. Uncompress it and put the individual json files in data/.
2. Get the Yelp category tree and put it in data. Get it from https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json.


[1]: https://www.yelp.com/dataset
