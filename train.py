import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torchvision.datasets as datasets
from operator import truediv
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

from global_vars import *


def log_output(oa_ae, aa_ae, top2_acc,top5_acc, element_acc, lr, epoch, path):
    f = open(path, 'a')
    f.write(f"\n\n\nFor Learning_Rate : {lr} & Epochs : {epoch}   The Result is ->\n")
    sentence1 = 'OA(Top-1 Accuracy) is: ' + str(oa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'AA is: ' + str(aa_ae) +'\n'
    f.write(sentence2)
    sentence3 = 'Top-2 Accuracy is: '+ str(top2_acc) + '\n'
    f.write(sentence3)
    sentence4 = 'Top-5 Accuracy is: '+ str(top5_acc) + '\n'
    f.write(sentence4)
    element_mean = list(element_acc)
    sentence5 = "Class wise accuracy: " + str(element_mean) + '\n'
    f.write(sentence5)
    f.close()


class Identity(nn.Module):
  def __init__(self):
      super(Identity,self).__init__()
  def forward(self,x):
      return x
  
  

def build_model(model, CHECKPOINT_PATH, name, CLASSES=CLASSES, TRAIN_DIRECTORY=TRAIN_DIRECTORY, TEST_DIRECTORY=TEST_DIRECTORY, VALID_DIRECTORY=VALID_DIRECTORY, EPOCHS = 100, LEARNING_RATE = 0.0000001, measure_performance=True):
    # print(CLASSES)

    transform =transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    train_dataset = datasets.ImageFolder(root=TRAIN_DIRECTORY, transform=transform)
    valid_dataset = datasets.ImageFolder(root=VALID_DIRECTORY, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIRECTORY, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    nclass = len(CLASSES)
    epochs = EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Define the early stopping criteria
    best_loss = float('inf')
    best_accuracy = 0.0
    patience = 5
    counter = 0

    # Lists to store accuracy and loss values
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

#     # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in tqdm(train_loader, desc="Trainning", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

#       # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validating", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / len(valid_loader.dataset)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break


        # Save the model if it has the best validation accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Load the best model checkpoint for evaluation
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    top2_correct=0
    top5_correct=0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _,top2preds = torch.topk(outputs.data,2,1)
            _,top5preds = torch.topk(outputs.data,5,1)
            test_correct += (predicted == labels).sum().item()
            for i,label in enumerate(labels):
              if label in top5preds[i]:
                top5_correct += 1
              if label in top2preds[i]:
                top2_correct += 1

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        top2_acc = top2_correct/len(test_loader.dataset)
        top5_acc = top5_correct/len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Top-2 Acc: {top2_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")

    # Plot accuracy and loss curves
    plt.figure(figsize=(20, 20))
    plt.plot(train_acc_list, label='Train')
    plt.plot(val_acc_list, label='Validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('./RESULTS/Acc_'+name +'.png')
    plt.show()


    plt.figure(figsize=(20, 20))
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('./RESULTS/loss_'+name +'.png')
    plt.show()


    # Generate and display confusion matrix
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    confusion_mat = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(confusion_mat)
    df.to_csv('./RESULTS/confuse_matrix_'+name+'.csv')
    counter = confusion_mat.shape[0]
    list_diag = np.diag(confusion_mat)
    list_raw_sum = np.sum(confusion_mat, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    ax = sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(CLASSES)
    ax.yaxis.set_ticklabels(CLASSES)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('./RESULTS/confusion_matrix_'+name +'.png', dpi=400)
    plt.show()
    # print("Each class accuracy",each_acc)
    log_output(test_acc,average_acc,top2_acc,top5_acc,each_acc, LEARNING_RATE, EPOCHS, './RESULTS/results_'+name+'.txt')
    return model