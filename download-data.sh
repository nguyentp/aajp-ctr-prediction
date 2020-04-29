#!/bin/bash

echo "[Start] Prepare directories"
mkdir -p ./data/raw/avazu
mkdir -p ./data/raw/avazu/sample
mkdir -p ./data/raw/ml-1m
mkdir -p ./data/raw/ml-1m/sample
echo "[Finish] Prepare directories"

echo "[Start] Download Avazu CTR datasets"
curl -o ./data/raw/avazu/avazu-ctr-test.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/test.gz
curl -o ./data/raw/avazu/avazu-ctr-train.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/train.gz
echo "[Finish] Download Avazu CTR datasets"

echo "[Start] Download MovieLens 1M datasets"
curl -o ./data/raw/ml-1m/movielens-1m.zip https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/movielens-1m/ml-1m.zip
echo "[Finish] Download MovieLens 1M datasets"

echo "[Start] Prepare raw datasets"
unzip -oj ./data/raw/ml-1m/movielens-1m.zip -d ./data/raw/ml-1m && rm ./data/raw/ml-1m/movielens-1m.zip

head -n 1000 ./data/raw/ml-1m/movies.dat > ./data/raw/ml-1m/sample/movies.dat
head -n 1000 ./data/raw/ml-1m/users.dat > ./data/raw/ml-1m/sample/users.dat
head -n 1000 ./data/raw/ml-1m/ratings.dat > ./data/raw/ml-1m/sample/rating.dat

gunzip -c ./data/raw/avazu/avazu-ctr-test.gz > ./data/raw/avazu/test
gunzip -c ./data/raw/avazu/avazu-ctr-train.gz > ./data/raw/avazu/train

head -n 1000 ./data/raw/avazu/train > ./data/raw/avazu/sample/train
head -n 1000 ./data/raw/avazu/test > ./data/raw/avazu/sample/test
echo "[Finish] Prepare raw datasets"
