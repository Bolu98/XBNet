import torch
from torch.utils.data import Dataset,DataLoader
from XBNet.training_utils import training
from sklearn.model_selection import KFold

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

# def run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,batch_size = 16,epochs=100):
#     '''
#     run_XBNET actually executes the entire training and validation methods for the models. Prints the metrics for the task
#     and plots the graphs of accuracy vs epochs and loss vs epochs.
#          :param X_train(numpy array): Features on which model has to be trained
#          :param y_train(numpy array): Labels of X_train i.e target variable
#          :param X_test(numpy array): Features on which model has to be validated
#          :param y_test(numpy array): Labels of X_test i.e target variable
#          :param model(XBNET Classifier/Regressor): model to be trained
#          :param criterion(object of loss function): Loss function to be used for training
#          :param optimizer(object of Optimizer): Optimizer used for training
#          :param batch_size(int,optional): Batch size used for training and validation. Default value: 16
#          :param epochs(int,optional): Number of epochs for training the model. Default value: 100
#       :return:
#          model object, list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
#     '''
#     trainDataload = DataLoader(Data(X_train, y_train), batch_size=batch_size)
#     testDataload = DataLoader(Data(X_test, y_test), batch_size=batch_size)
#     acc, lo, val_ac, val_lo = training(model, trainDataload, testDataload, criterion, optimizer, epochs)
#     return model,acc, lo, val_ac, val_lo

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



