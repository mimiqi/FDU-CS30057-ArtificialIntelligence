import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler

class SVMModel:
    # todo:
    """
        Initialize Support Vector Machine (SVM from sklearn) model.

    """
    """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """
    """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """
    def __init__(self):
        self.model = LinearSVC(C=1.0, max_iter=2000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)

class SVMFromScratch:
    def __init__(self, lr=0.1, epochs=300, lam=0.01, batch_size=64, decay=0.005):
        self.lr         = lr          # 初始学习率
        self.epochs     = epochs      # 训练轮数
        self.lam        = lam         # L2 正则化系数
        self.batch_size = batch_size  # Mini-batch 大小
        self.decay      = decay       # 学习率衰减率
        self.scaler     = StandardScaler()
        self.w = None
        self.b = 0.0

    def train(self, train_data, train_targets):
        X = self.scaler.fit_transform(train_data)  # 特征标准化
        n, d = X.shape
        y = np.where(train_targets == 1, 1, -1).astype(float)
        self.w = np.zeros(d)
        self.b = 0.0
        rng = np.random.default_rng(42)
        for epoch in range(self.epochs):
            lr_t = self.lr / (1.0 + self.decay * epoch)  # 学习率衰减
            idx  = rng.permutation(n)
            for start in range(0, n, self.batch_size):   # Mini-batch SGD
                batch   = idx[start : start + self.batch_size]
                Xb, yb  = X[batch], y[batch]
                margins = yb * (Xb @ self.w + self.b)
                mask    = margins < 1
                dw = self.lam * self.w - (Xb[mask] * yb[mask, None]).mean(axis=0) if mask.any() else self.lam * self.w
                db = -yb[mask].mean() if mask.any() else 0.0
                self.w -= lr_t * dw
                self.b -= lr_t * db

    def evaluate(self, data, targets):
        X     = self.scaler.transform(data)
        y     = np.where(targets == 1, 1, -1)
        preds = np.sign(X @ self.w + self.b)
        preds[preds == 0] = 1
        return np.mean(preds == y)
    

def data_preprocess():
    diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]
      
        ## todo: Try to load data/target
        train_mask = task_col.isin([1, 2])
        test_mask = task_col.isin([3, 4])

        train_data = diagrams[train_mask.values]
        test_data = diagrams[test_mask.values]

        train_targets = (task_col[train_mask] == 1).astype(int).values
        test_targets = (task_col[test_mask] == 3).astype(int).values

        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main():

    data_list, target_list = data_preprocess()

    task_acc_train = []
    task_acc_test = []
    
    ## Todo:Model Initialization 
    ## You can also consider other different settings

    model = SVMModel()
    # model = SVMFromScratch()
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    main()

