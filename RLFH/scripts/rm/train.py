# import modules 
import time
import numpy as np

import torch

import utils

def train():
    '''
    Pipeline for training BERT model
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
    

    # --- load optimizer --- #
    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
    )

    # --- load learning rate scheduler --- #
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim,
        mode="min",
        min_lr=1e-6,
    )

    # --- train model --- #
    n_epochs = 10
    best_valid_loss = float("inf") # best validation loss 

    metric = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }

    for epoch in range(n_epochs):
       
        train_loss, train_acc = 0, 0
        # train 
        model.train()
        for i, batch in enumerate(data_loader_train):
            # move data to cuda 
            batch = {key: val.to(device) for key, val in batch.items()} 

            # forward process
            loss, logits = model(
                #outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"],
                return_dict=False,
            )
            
            # backward process 
            model.zero_grad()
            loss.backward()
            optim.step()

            # compute train loss/accuracy
            train_loss += loss.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc = torch.sum(pred_labels == batch["labels"]) / len(batch["labels"])
            train_acc += acc.item()

        train_loss /= len(data_loader_train)
        train_acc /= len(data_loader_train)

        # validation
        valid_loss, valid_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader_valid):
                
                # move data to cuda 
                batch = {key: val.to(device) for key, val in batch.items()} 
               
                # forward process
                loss, logits = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["labels"],
                    return_dict=False, 
                )
            
                # compute validation loss/accuracy 
                valid_loss += loss.item()
                pred_labels = torch.argmax(logits, axis=1)
                acc = torch.sum(pred_labels == batch["labels"]) / len(batch["labels"])
                valid_acc += acc.item()
                
            valid_loss /= len(data_loader_valid)
            valid_acc /= len(data_loader_valid)

        # update saved model as validation loss decreases. 
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            # save model weight
            print("best model saved at epoch {}".format(epoch))
            torch.save(model.state_dict(), "best_model_weights.pt")
        
        # update metrics
        metric["train_loss"].append(train_loss)
        metric["train_acc"].append(train_acc)
        metric["valid_loss"].append(valid_loss)
        metric["valid_acc"].append(valid_acc)
        
        # update the learning rate
        lr_scheduler.step(valid_loss)

        print(train_loss, valid_loss, train_acc, valid_acc)

    np.save("train_valid_metric.npy", metric)


def main():
    # implement train function
    train()

if __name__ == "__main__":

    time_1 = time.time()
    main()
    time_2 = time.time()

    print(time_2 - time_1)
