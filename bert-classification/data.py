from torchtext import data
from torchtext import datasets
from transformers import BertTokenizerFast
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch

TEXT = data.Field()
LABEL = data.Field()
train, test = datasets.IMDB.splits(TEXT, LABEL)

train, validation = train.split(split_ratio=0.666)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, data, tokenizer):
        super(Data, self).__init__()
        self.data = data
        self.text = list(self.data.text)
        self.label = list(self.data.label)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        tokenized = self.tokenizer(
            " ".join(self.text[ix]),
            max_length=512,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        label = torch.tensor([1 if self.label[ix][0] == "pos" else 0])
        return tokenized, label


def dataloader(tokenizer, args):
    train_set = Data(train, tokenizer=tokenizer)
    valid_set = Data(validation, tokenizer=tokenizer)
    test_set = Data(test, tokenizer=tokenizer)

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    return train_loader, valid_loader, test_loader
