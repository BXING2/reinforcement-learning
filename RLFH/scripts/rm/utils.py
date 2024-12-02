# 
import os
import numpy as np
import pandas as pd
import torch 
from transformers import BertTokenizer, BertForSequenceClassification


class Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root,
        max_length=512,
    ):
        
        self.root = root
        self.max_length = max_length

        # load moive review data
        data = np.load(
            os.path.join(root, "movie_reviews.npy"),
            allow_pickle=True,
        ).item()

        # load bert tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
        )
        
        # tokenize
        input_ids, attention_mask = [], []
        for reviews in [data["positive"], data["negative"]]:
            for review in reviews:
        
                encoding = tokenizer(
                    review,    
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
    
                input_ids.append(encoding["input_ids"])
                attention_mask.append(encoding["attention_mask"])
    
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_mask = torch.cat(attention_mask, dim=0)

        labels = [1] * len(data["positive"]) + [0] * len(data["negative"])
        self.labels = torch.tensor(labels, dtype=torch.int64)

        print(self.input_ids.shape, self.attention_mask.shape, self.labels.shape)


    def __len__(self):
        # number of data in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return one instance 
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def load_model(
    n_classes,
    n_encoder_layers,
):

    # load bert model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=n_classes,
    )

    # freeze embedding layers and encoder layers for fine tuning
    for params in model.bert.embeddings.parameters():
        params.requires_grad = False
    for params in model.bert.encoder.layer[:n_encoder_layers].parameters():
        params.requires_grad = False
    
    return model

