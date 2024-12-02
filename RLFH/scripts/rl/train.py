import os
import time
import collections
import numpy as np

import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

import collections
import utils

def train():

    # --- load device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ---load dataset --- #
    dataset = utils.Dataset(
        root="../../",
        max_length=512,
    )

    # split train/valid/test subset
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(dataset)).tolist()

    train_split, valid_split, test_split = 0.8, 0.1, 0.1
    train_index = int(len(indices) * train_split)
    valid_index = int(len(indices) * valid_split) + train_index

    dataset_train = torch.utils.data.Subset(dataset, indices[: train_index])            
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_index: valid_index])
    dataset_test = torch.utils.data.Subset(dataset, indices[valid_index:])

    # train/valid/test data loaders
    def collate_func(batch):
        return [instance.to(device) for instance in batch]
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_func,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_func,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_func,
    )

    # sft/rm tokenizer
    sft_tokenizer = dataset.sft_tokenizer
    rm_tokenizer = dataset.rm_tokenizer
   

    # --- load sft model --- #

    # sft model (gpt2 model) to be finetuned
    n_decoder_layers = 10
    model = utils.load_sft_model(
        n_tokens=len(dataset.sft_tokenizer),
        n_decoder_layers=n_decoder_layers,
    ).to(device) 

    # sft model (gpt2 model) as a reference
    ref_model = utils.load_sft_model(
        n_tokens=len(dataset.sft_tokenizer),
        n_decoder_layers=n_decoder_layers,
    ).to(device)

    # load model weights
    model_weight_path = "../../SFT/"
    for curr_model in [model, ref_model]:
        curr_model.pretrained_model.transformer.wte.load_state_dict(torch.load(model_weight_path + "token_embedding.pt", weights_only=True))
        curr_model.pretrained_model.transformer.h[n_decoder_layers:].load_state_dict(torch.load(model_weight_path + "decoder_layer.pt", weights_only=True))
        curr_model.pretrained_model.transformer.ln_f.load_state_dict(torch.load(model_weight_path + "layer_norm.pt", weights_only=True))


    # --- load reward model (bert model) --- #
    n_classes = 2 # number of classes
    reward_model = utils.load_reward_model(
        n_classes,
    ).to(device)

    # load model weights
    model_weight_path = "../../RM/best_model_weights.pt"
    reward_model.load_state_dict(torch.load(model_weight_path, weights_only=True))

    #for name, params in model.named_parameters():
    #    print(name, params.shape, params.requires_grad)


    # --- load PP0 trainer --- #

    # PPO configs
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=100,
        ppo_epochs=2,
        mini_batch_size=10,
    )

    print(config.batch_size, config.mini_batch_size, config.ppo_epochs)

    # load trainer function
    # both model and ref model are needed for comparing action probability 
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        sft_tokenizer,
    )


    # --- load optimizer --- #
    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-6,
    )


    # --- finetune sft model using ppo algorithm --- # 

    for iterations, queries in enumerate(data_loader_train):
        
        # collect batch of data 
        batch =collections.defaultdict(list)
        for query_ids in queries:
        
            # generate query response using multinormial sampling 
            query_response_ids = ppo_trainer.generate(
                query_ids,
                max_length=512,
                do_sample=True,
                top_k=0, #50,
                top_p=1.0,
                pad_token_id=sft_tokenizer.pad_token_id,
                eos_token_id=sft_tokenizer.eos_token_id,
            )[0]
           
            response_length = len(query_response_ids) - len(query_ids)
            response_ids = query_response_ids[-response_length:]
            
            # store
            batch["query_ids"].append(query_ids)
            batch["response_ids"].append(response_ids)
            batch["pairs"].append(sft_tokenizer.decode(query_response_ids.squeeze(), skip_special_tokens=True,))
        
        # get reward from bert model 
        for pair in batch["pairs"]:
           
            encoding = rm_tokenizer(
                pair,                      # Sentence to encode.
                padding="max_length",
                truncation=True,
                max_length=512,       # Pad & truncate all sentences
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",         # Return pytorch tensors.
            )
        
            outputs = reward_model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
            
            # logits of positive class as reward
            batch["rewards"].append(outputs.logits.detach()[0, 1])
       
        print(np.mean([reward.item() for reward in batch["rewards"]]))

        # finetune via ppo trainer
        ppo_trainer.step(batch["query_ids"], batch["response_ids"], batch["rewards"])

    # save model
    torch.save(model.pretrained_model.state_dict(), "rl_model.pt")

def main():
    train()

if __name__ == "__main__":
    main()
