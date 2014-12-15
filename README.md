recsys_challenge
================

http://2015.recsyschallenge.com/challenge.html

Downloading and preparing the data:

```
git clone git@github.com:pilipolio/recsys_challenge.git
cd recsys_challenge
mkdir data
wget --directory-prefix=data https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
7z e data/yoochoose-data.7z -o data
python recsys_challenge/loader.py
```

Train and run a predictor

```
mkdir solutions
PYTHONPATH=. python recsys_challenge/predictors/item_pop.py
```
