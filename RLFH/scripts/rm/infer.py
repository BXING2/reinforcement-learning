#!/usr/bin/env python
# coding: utf-8

# import modules 
import os
import time
import numpy as np

import torch
import evaluate

import utils

def infer():
    '''
    Pipeline for testing BERT model on the test dataset
    '''

    # --- load device --- # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load dataset --- #
    n_classes = 2 # number of classes in the dataset
    path = "../../"
    batch_size = 16

    dataset = utils.Dataset(root=path)

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
        shuffle=True,
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
    model = utils.load_model(
        n_classes,
        n_encoder_layers=11,
    ).to(device)

    for name, params in model.named_parameters():
        print(name, params.shape, params.requires_grad)
    
    # load saved model weights
    model_weight_path = "../model_1/best_model_weights.pt"
    model.load_state_dict(torch.load(model_weight_path, weights_only=True))
    model.eval()


    # --- model evaluation --- #
    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    f1_metric = evaluate.load("f1")
    conf_metric = evaluate.load("confusion_matrix")

    names = ["train", "valid", "test"]
    dataloaders = [data_loader_train, data_loader_valid, data_loader_test]
    results = {}
    for name, dataloader in zip(names, dataloaders):
        
        truth, preds = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                
                # move data to cuda
                batch = {key: val.to(device) for key, val in batch.items()}            
                
                loss, logits = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["labels"],
                    return_dict=False,
                )
            
                pred_labels = torch.argmax(logits, axis=1)

                truth.extend(batch["labels"].cpu().numpy().tolist())
                preds.extend(pred_labels.cpu().numpy().tolist())
            
            # compute metris  
            accuracy = accuracy_metric.compute(
                predictions=preds,
                references=truth,
            ) 
            
            recall = recall_metric.compute(
                predictions=preds,
                references=truth,
                average="macro",
            ) 

            precision = precision_metric.compute(
                predictions=preds,
                references=truth,
                average="macro",
            ) 
            
            f1 = f1_metric.compute(
                predictions=preds,
                references=truth,
                average="macro",
            ) 

            conf_matrix = conf_metric.compute(
                predictions=preds,
                references=truth,
            ) 

            print(name, accuracy["accuracy"], recall["recall"], precision["precision"], f1["f1"], conf_matrix["confusion_matrix"])
            results[name] = [accuracy["accuracy"], recall["recall"], precision["precision"], f1["f1"], conf_matrix["confusion_matrix"]]
        
    np.save("metric_summary.npy", results)

def main():
    infer()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print("running time {}s".format(time_2 - time_1))
