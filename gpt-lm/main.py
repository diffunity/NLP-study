import os
import random
import logging
import argparse
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader, random_split
# Pre-setting
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)

def seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def dataloader(args):
    data = pd.read_csv(
        "./movie_lines.tsv", sep="\t", error_bad_lines=False, header=None
    )
    data = data[4].dropna().reset_index(drop=True).values[:5000]
    train_loader = DataLoader(
        dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    return train_loader

def train(model, optimizer, scheduler, train_data, tokenizer, args):
    model = model.to(device)
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    for e, text in enumerate(train_data):
        tokenized = tokenizer(
            text,
            max_length=128,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        input_ids, attention_mask = (
            tokenized["input_ids"].to(device),
            tokenized["attention_mask"].to(device),
        )

        loss, logits = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )[:2]

        loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if e % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if e % 500 == 0:
            logger.info(f"Batch {e} Loss: {loss.item()}")


def main(args):

    set_seed(args)

    model = models[args.model]

    tokenizer = tokenizers[args.tokenizer]

    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.learning_rate)

    # scheduler = ReduceLROnplateau(optimizer, 'min')

    dl = dataloader(args)

    print(
        "Total Training Steps: ",
        args.epochs
        * len(dl.dataset)
        / (args.batch_size * args.gradient_accumulation_steps),
    )

    for epochs in range(args.epochs):
        train(model, optimizer, dl, tokenizer, args)

    seq = "Hey would you like to "
    tokenized_seq = tokenizer([seq], return_tensors="pt")["input_ids"].to(device)
    generated_text = model.generate(
        tokenized_seq, max_length=len(tokenized_seq[0]) + 5, num_beams=3
    )[0]

    print(tokenizer.decode(generated_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default=None, type=str, required=True, help="Type in the model"
    )

    parser.add_argument(
        "--optimizer",
        default=None,
        type=str,
        required=True,
        help="Type in the optimizer",
    )

    parser.add_argument(
        "--tokenizer",
        default=None,
        type=str,
        required=True,
        help="Type in the tokenizer",
    )

    parser.add_argument(
        "--learning_rate", default=None, required=True, type=float, help="Learning rate"
    )

    parser.add_argument(
        "--epochs", default=None, required=True, type=int, help="Epochs for training"
    )

    parser.add_argument(
        "--batch_size", default=5, type=int, help="Batch size for training"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=10,
        type=int,
        help="Gradient Accumulation Step",
    )

    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    parser.add_argument(
        "--seq_len", default=256, type=int, help="Maximum Sequnece Length for input"
    )

    args = parser.parse_args()

    main(args)
