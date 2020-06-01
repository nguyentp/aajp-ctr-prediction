#!/bin/bash

echo "[Start] Download Avazu CTR datasets"
mkdir -p ./data/raw/avazu
curl -o ./data/raw/avazu/avazu-ctr-test.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/test.gz
curl -o ./data/raw/avazu/avazu-ctr-train.gz https://aaajp-internal-ctr.s3-ap-northeast-1.amazonaws.com/datasets/avazu-ctr/train.gz
echo "[Finish] Download Avazu CTR datasets"

echo "[Start] Prepare raw datasets"
gunzip -c ./data/raw/avazu/avazu-ctr-test.gz > ./data/raw/avazu/test
gunzip -c ./data/raw/avazu/avazu-ctr-train.gz > ./data/raw/avazu/train
rm ./data/raw/avazu/avazu-ctr-test.gz
rm ./data/raw/avazu/avazu-ctr-train.gz

mkdir -p ./data/raw/avazu/sample
head -n 1000 ./data/raw/avazu/train > ./data/raw/avazu/sample/train
head -n 1000 ./data/raw/avazu/test > ./data/raw/avazu/sample/test
echo "[Finish] Prepare raw datasets"
