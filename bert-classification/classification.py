import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(model, optimizer, dataloader, args):
  # print(type(model))
  # print(type(optimizer))
  model = model.to(device)
  model.train()

  for e, (text, label) in enumerate(dataloader):
      y = label.to(device)
      input_ids, attention_mask = text["input_ids"].squeeze(1).to(device), \
                                  text["attention_mask"].squeeze(1).to(device)
      # print(input_ids.shape)
      # print(attention_mask.shape)
      # # print(token_type_ids.shape)
      # print(y.shape)
      loss, logits = model(input_ids = input_ids,
                           attention_mask = attention_mask,
                           labels = y)

      loss.backward()

      # loss = loss/args.gradient_accumulation_steps

      optimizer.step()
      
      optimizer.zero_grad()

      if e % 500 == 0:
          print(f"Batch {e} Loss : {loss.item()}")
          training_acc = (torch.argmax(logits,1) == y.squeeze(1)).sum().item() / len(logits)
          print(f"Batch {e} Training Accuracy: {training_acc}")
      
  # loss.detach()


def evaluate(model, val_iter):
#   val_res = []
  res = 0
  model.eval()
#   corrects, total_loss = 0, 0
  for e, (text, label) in enumerate(val_iter):
    with torch.no_grad():
      y = label.to(device)
      input_ids, attention_mask = text["input_ids"].squeeze(1).to(device), \
                                  text["attention_mask"].squeeze(1).to(device)

      logits = model(input_ids = input_ids,
                    attention_mask = attention_mask)[0]
      res += (torch.argmax(logits,1) == y.squeeze(1)).sum().item() / len(y)
      
      if e % 500 == 0:
        print(f"Batch{e} Accuracy: {res/(e+1)}")
