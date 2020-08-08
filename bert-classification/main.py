import argparse
import logging
import os
import torch

from data import dataloader
from classification import train, evaluate, set_seed
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMTokenizer, XLMForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification
)
from transformers import AdamW

models = {
    "bert" : BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # "xlm" : XLMForSequenceClassification.from_pretrained("xlm-mlm-en-2048"),
    # "xlnet" : XLNetForSequenceClassification.from_pretrained("xlnet-base-uncased")
}

tokenizers = {
    "bert": BertTokenizer.from_pretrained("bert-base-uncased")
    # "xlm" : XLMTokenizer.from_pretrained("xlm-mlm-en-2048"),
    # "xlnet" : XLNetTokenizer.from_pretrained("xlnet-base-uncased")
}

optimizers = {
    "adam": torch.optim.Adam,
    "adamW": AdamW
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
                        default = None, 
                        type = str, 
                        required = True, 
                        help = "Type in the model")
    parser.add_argument("--optimizer",
                        default = None,
                        type = str,
                        required = True,
                        help = "Type in the optimizer")
    parser.add_argument("--tokenizer",
                        default = None,
                        type = str,
                        required = True,
                        help = "Type in the tokenizer")
    
    parser.add_argument("--epochs", 
                        default = None,
                        required = True, 
                        type = int, 
                        help = "Epochs for training")

    parser.add_argument("--learning_rate",
                        default = None,
                        required = True,
                        type = float, 
                        help = "Learning rate")

    parser.add_argument("--gradient_accumulation_steps",
                        default = 1,
                        type = int,
                        help = "Batch size = batch_size * gradient_accumulation_steps")

    parser.add_argument("--seed", default = 42, type = int, help = "Random seed")

    parser.add_argument("--batch_size", default = 5, type = int, help = "Batch size for training")

    parser.add_argument("--seq_len", default = 256, type = int, help = "Maximum Sequnece Length for input")
    
    args = parser.parse_args()

    set_seed(args)
    
    model = models[args.model]
    tokenizer = tokenizers[args.tokenizer]
    optimizer = optimizers[args.optimizer](
                model.parameters(), lr = args.learning_rate)

    train_loader, valid_loader, test_loader = dataloader(tokenizer, args)
    
    for i in range(args.epochs):
        train(model, optimizer, train_loader, args)
        evaluate(model, valid_loader)

    evaluate(model, test_loader)