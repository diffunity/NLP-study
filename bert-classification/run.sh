python main.py \
    --model bert \
    --optimizer adamW \
    --tokenizer bert \
    --epochs 1 \
    --learning_rate 0.00001 \
    --seq_len 128 \
    --batch_size 2

# Modelling and training Specs

# Data : PyTorch IMDB
# Task : Sentiment Analysis (['pos','neg'])
# Model : BERT base ("bert-base-uncased")

# training data size: 16675
# validation data size: 8325
# test data size : 25000

# accuracy : 0.92 