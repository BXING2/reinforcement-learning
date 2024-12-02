# import modules 
import time
import numpy as np

import torch

import utils

def infer():
    '''
    Pipeline for testing the GPT model
    '''

    # --- load device --- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- load dataset --- #
    path = "../../" 
    batch_size = 8

    dataset = utils.Dataset(
        root=path,
        max_length=512,
    )

    # train/valid/test subsets
    torch.manual_seed(0) # set torch random seed
    indices = torch.randperm(len(dataset)).tolist()

    train_split, valid_split, test_split = 0.8, 0.1, 0.1
    train_index = int(len(indices) * train_split)
    valid_index = int(len(indices) * valid_split) + train_index

    dataset_train = torch.utils.data.Subset(dataset, indices[: train_index])            
    dataset_valid = torch.utils.data.Subset(dataset, indices[train_index: valid_index])
    dataset_test = torch.utils.data.Subset(dataset, indices[valid_index:])

    # define train/valid/test data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    # --- load model --- #
    n_tokens = len(dataset.tokenizer)
    n_decoder_layers = 10
    model = utils.load_model(
        n_tokens=n_tokens,
        n_decoder_layers=n_decoder_layers,
    ).to(device)
   
    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)
    
    # load model weights
    model_weight_path = "../model_2/"
    model.transformer.wte.load_state_dict(torch.load(model_weight_path + "token_embedding.pt", weights_only=True))
    model.transformer.h[n_decoder_layers:].load_state_dict(torch.load(model_weight_path + "decoder_layer.pt", weights_only=True))
    model.transformer.ln_f.load_state_dict(torch.load(model_weight_path + "layer_norm.pt", weights_only=True))

    # --- model inference --- #
    model.eval()

    data_loaders = [data_loader_train, data_loader_valid, data_loader_test]
    splits = ["train", "valid", "test"]

    metric = {}
    for data_loader, split in zip(data_loaders, splits):
        # total loss 
        loss = 0
        for i, batch in enumerate(data_loader):
            # move data to cuda device
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss += outputs.loss.item()

        # loss 
        loss /= len(data_loader)

        # perplexity
        perplexity = np.exp(loss)

        # update metrics
        metric["{}_loss".format(split)] = loss
        metric["{}_perp".format(split)] = perplexity

    np.save("metric_summary.npy", metric)
    print(metric)


def main():
    # implement infer function
    infer()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
