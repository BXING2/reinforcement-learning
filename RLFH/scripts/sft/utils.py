# 
import os
import numpy as np

import torch 
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root, # root directory of data
        max_length=512, # max sequence length 
    ):
        
        self.root = root
        self.max_length = max_length

        # load movie review data 
        data = np.load(
            os.path.join(root, "movie_reviews.npy"),
            allow_pickle=True,
        ).item()

            
        # load gpt2 tokenizer  
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<|unknown|>",
            pad_token="<|pad|>",
        )
        
        # tokenize
        input_ids, attention_mask = [], []
        for reviews in [data["positive"], data["negative"]]:
            for review in reviews:

                # add start token and end token to each sentence
                review = "<|startoftext|>" + review + "<|endoftext|>"
                
                # encode sentence
                encoding = self.tokenizer(
                    review,   
                    padding="max_length",        
                    max_length=self.max_length,  
                    truncation=True,             
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids.append(encoding["input_ids"])
                attention_mask.append(encoding["attention_mask"])
        
        # input ids, attention mask, labels (labels are identical to input_ids for self-supervised learning)  
        self.input_ids = torch.cat(input_ids)
        self.attention_mask = torch.cat(attention_mask)
        self.labels = torch.cat(input_ids)

        # set ids to -100 belonging to prompt tokens
        for i in range(len(self.labels)):
            prompt_length = int(0.1 * torch.sum(self.attention_mask[i])) # prompt length is set to 10% of the sentence length
            self.labels[i, :prompt_length] = -100

        # set labels to -100 for padding tokens so that they do not contribute to loss functions 
        self.labels[~self.attention_mask.bool()] = -100


    def __len__(self):

        # return the number of instance in dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return one instance 
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def load_model(
    n_tokens, # number of tokens in the vocabulary
    n_decoder_layers, # number of decoder layers to freeze
):

    # load gpt2 model
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
    )

    # resize embedding dimension due to added special tokens
    model.resize_token_embeddings(n_tokens)

    # freeze position embedding layer
    for params in model.transformer.wpe.parameters():
        params.requires_grad = False
    # freeze decoder layers
    for params in model.transformer.h[:n_decoder_layers].parameters():
        params.requires_grad = False

    return model

