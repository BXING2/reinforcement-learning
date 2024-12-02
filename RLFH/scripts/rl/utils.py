# 
import os
import numpy as np

import torch 
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

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

        # load gpt tokenizer
        self.sft_tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            unk_token="<|unknown|>",
            pad_token="<|pad|>",
        )

        # load bert tokenizer
        self.rm_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
        )
        
        # tokenize
        self.input_ids = []
        for reviews in [data["positive"], data["negative"]]:
            for review in reviews:

                # add start token and end token to each sentence
                review = "<|startoftext|>" + review + "<|endoftext|>"
                
                # encode sentence
                encoding = self.sft_tokenizer(
                    review,   
                    padding="max_length",        
                    max_length=self.max_length,  
                    truncation=True,             
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                
                # store prompt for generating sentence from sft and rl model
                prompt_length = int(0.1 * torch.sum(encoding["attention_mask"]))
                self.input_ids.append(encoding["input_ids"][0][:prompt_length])


    def __len__(self):

        # return the number of instance in dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return one instance 
        return self.input_ids[idx]



def load_sft_model(
    n_tokens, # number of tokens in the vocabulary
    n_decoder_layers, # number of decoder layers to freeze
):

    # load gpt2 model
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
    )

    # resize embedding dimension due to added special tokens
    model.resize_token_embeddings(n_tokens)

    '''
    # freeze position embedding layer
    for params in model.transformer.wpe.parameters():
        params.requires_grad = False
    # freeze decoder layers
    for params in model.transformer.h[:n_decoder_layers].parameters():
        params.requires_grad = False
    '''

    # add value head to gpt model
    model = AutoModelForCausalLMWithValueHead(
        pretrained_model=model,
    )


    return model


def load_reward_model(
    n_classes,
):

    # load pre-trained bert model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=n_classes,
    )

    return model

