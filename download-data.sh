#!/bin/bash

echo "[Start] Download all Datasets"

echo "[Start] Download Avazu CTR datasets"
curl -o ./data/raw/avazu-ctr-test.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/test.gz
curl -o ./data/raw/avazu-ctr-train.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/train.gz
echo "[Finish] Download Avazu CTR datasets"

echo "[Start] Download MovieLens 1M datasets"
curl -o ./data/raw/movielens-1m.zip https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/movielens-1m/ml-1m.zip
unzip -o ./data/raw/movielens-1m.zip -d ./data/raw
rm ./data/raw/movielens-1m.zip
echo "[Finish] Download MovieLens 1M datasets"

echo "[Start] Sample 1000 rows from some datasets"
gzip -cd ./data/raw/avazu-ctr-test.gz | head -n 1000 > ./data/raw/sample-avazu-ctr-test.txt
gzip -cd ./data/raw/avazu-ctr-train.gz | head -n 1000 > ./data/raw/sample-avazu-ctr-train.txt
echo "[Finish] Sample 1000 rows from some datasets"

echo "[Finish] Download all Datasets"
