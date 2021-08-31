import torch
from torch.utils.data import Dataset,DataLoader
from XBNet.training_utils import training
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class Data(Dataset):
    '''
    Dataset class for loading the data into dataloader
        :param X(numpy array): array of features
        :param y(numpy array): array of labels of the features
    '''
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_XBNET(X_train,y_train,model,criterion,optimizer,batch_size=16,epochs=100):
    N_splits = 5
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=0)
    accuracy = []
    cost = []
    val_accuracy = []
    val_cost = []
    for train_idx, val_idx in kf.split(X_train):
        trainDataLoad = DataLoader(Data(X_train[train_idx], y_train[train_idx]), batch_size=batch_size)
        valDataLoad = DataLoader(Data(X_train[val_idx], y_train[val_idx]), batch_size=batch_size)
        acc, loss, val_acc, val_loss = training(model,trainDataLoad,valDataLoad,criterion,optimizer,epochs)
        accuracy.append(acc)
        cost.append(loss)
        val_accuracy.append(val_acc)
        val_cost.append(val_loss)
    final_acc = [sum(col)/len(col) for col in zip(*accuracy)]
    final_cost = [sum(col)/len(col) for col in zip(*cost)]
    final_val_acc = [sum(col)/len(col) for col in zip(*val_accuracy)]
    final_val_cost = [sum(col)/len(col) for col in zip(*val_cost)]
    return model, final_acc, final_cost, final_val_acc, final_val_cost


def plot_feature(model):
    plt.figure()
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance Magnitude')
