import torch
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
from Data_loader.dataLoader import data_loaders
import argparse
from tqdm import tqdm
import random

def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="C:\\Users\\IICT2\\Downloads\\Dataset_LCTSC")
    args.add_argument("--output_path", type=str, default="E:\\IICT5\\Output_Dice_CE")
    args.add_argument("--dataset", type=str, default="LCTSC")
    args.add_argument("--model_name", type=str, default="our")
    args.add_argument("--epoch_num", type=int, default=50)
    args = args.parse_args()
    return args

def main(conf):
    # Configuration options
    k_folds = 5
    num_epochs = 20
    loss_function = nn.CrossEntropyLoss()

    data_len = 70000

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=False)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(np.array([i for i in range(data_len)]))):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        loader_train, loader_valid = data_loaders(train_ids, test_ids, conf.data_path)
        loaders = {"train": loader_train, "valid": loader_valid}

        for epoch in tqdm(range(conf.epoch_num)):
            print("\n {epc} is running".format(epc=epoch))
            loss_train = []
            img_print = random.randint(500, 600)

            for phase in ["train", 'valid']:
                # if phase == "train":
                #     validation_predict = {}
                #     validation_true = {}
                #     model.train()
                # else:
                #     model.eval()

                for i, data in enumerate(loaders[phase]):
                    x, y_true = data
                    if i%5000==0:
                        print(f"{i} -> x = {x} y= {y_true}")
                    # x, y_true = x.to(device), y_true.to(device)

                    # optimizer.zero_grad()
                    #
                    # with torch.set_grad_enabled(phase == "train"):


    #     # Init the neural network
    #     network = SimpleConvNet()
    #     network.apply(reset_weights)
    #
    #     # Initialize optimizer
    #     optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    #
    #     # Run the training loop for defined number of epochs
    #     for epoch in range(0, num_epochs):
    #
    #         # Print epoch
    #         print(f'Starting epoch {epoch + 1}')
    #
    #         # Set current loss value
    #         current_loss = 0.0
    #
    #         # Iterate over the DataLoader for training data
    #         for i, data in enumerate(trainloader, 0):
    #
    #             # Get inputs
    #             inputs, targets = data
    #
    #             # Zero the gradients
    #             optimizer.zero_grad()
    #
    #             # Perform forward pass
    #             outputs = network(inputs)
    #
    #             # Compute loss
    #             loss = loss_function(outputs, targets)
    #
    #             # Perform backward pass
    #             loss.backward()
    #
    #             # Perform optimization
    #             optimizer.step()
    #
    #             # Print statistics
    #             current_loss += loss.item()
    #             if i % 500 == 499:
    #                 print('Loss after mini-batch %5d: %.3f' %
    #                       (i + 1, current_loss / 500))
    #                 current_loss = 0.0
    #
    #     # Process is complete.
    #     print('Training process has finished. Saving trained model.')
    #
    #     # Print about testing
    #     print('Starting testing')
    #
    #     # Saving the model
    #     save_path = f'./model-fold-{fold}.pth'
    #     torch.save(network.state_dict(), save_path)
    #
    #     # Evaluationfor this fold
    #     correct, total = 0, 0
    #     with torch.no_grad():
    #
    #         # Iterate over the test data and generate predictions
    #         for i, data in enumerate(testloader, 0):
    #             # Get inputs
    #             inputs, targets = data
    #
    #             # Generate outputs
    #             outputs = network(inputs)
    #
    #             # Set total and correct
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += (predicted == targets).sum().item()
    #
    #         # Print accuracy
    #         print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
    #         print('--------------------------------')
    #         results[fold] = 100.0 * (correct / total)
    #
    # # Print fold results
    # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    # print('--------------------------------')
    # sum = 0.0
    # for key, value in results.items():
    #     print(f'Fold {key}: {value} %')
    #     sum += value
    # print(f'Average: {sum / len(results.items())} %')

if __name__ == "__main__":
    main(conf())