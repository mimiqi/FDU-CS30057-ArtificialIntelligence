"""
课外探索实验脚本
1. 超参数 C 对 LRModel / SVMModel 的影响
2. 改进手写版 LRFromScratch / SVMFromScratch
运行方式: python explore.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# 数据加载（复用 main_lr.py 的逻辑）
# ─────────────────────────────────────────
def data_preprocess():
    diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list, target_list = [], []
    for task in range(1, 56):
        task_col = cast.iloc[:, task]
        train_mask = task_col.isin([1, 2])
        test_mask  = task_col.isin([3, 4])
        train_data    = diagrams[train_mask.values]
        test_data     = diagrams[test_mask.values]
        train_targets = (task_col[train_mask] == 1).astype(int).values
        test_targets  = (task_col[test_mask]  == 3).astype(int).values
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    return data_list, target_list


def avg_accuracy(model_cls, data_list, target_list, **kwargs):
    """对55个任务求平均 train/test 准确率"""
    train_accs, test_accs = [], []
    for (train_data, test_data), (train_tgt, test_tgt) in zip(data_list, target_list):
        m = model_cls(**kwargs)
        m.fit(train_data, train_tgt)
        train_accs.append(m.score(train_data, train_tgt))
        test_accs.append(m.score(test_data, test_tgt))
    return np.mean(train_accs), np.mean(test_accs)


# ─────────────────────────────────────────
# 改进版 LRFromScratch
# ─────────────────────────────────────────
class LRFromScratchV2:
    """
    改进点：
    1. 特征标准化（StandardScaler）
    2. L2 正则化
    3. 学习率衰减
    """
    def __init__(self, lr=0.5, epochs=300, lam=0.01, decay=0.01):
        self.lr     = lr
        self.epochs = epochs
        self.lam    = lam      # L2 正则化系数
        self.decay  = decay    # 学习率衰减率
        self.scaler = StandardScaler()
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train(self, train_data, train_targets):
        X = self.scaler.fit_transform(train_data)   # 标准化
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for epoch in range(self.epochs):
            lr_t = self.lr / (1.0 + self.decay * epoch)  # 学习率衰减
            z     = X @ self.w + self.b
            y_hat = self._sigmoid(z)
            diff  = y_hat - train_targets
            # 加入 L2 正则梯度：lam * w
            self.w -= lr_t * ((X.T @ diff) / n + self.lam * self.w)
            self.b -= lr_t * diff.mean()

    def evaluate(self, data, targets):
        X     = self.scaler.transform(data)
        preds = (self._sigmoid(X @ self.w + self.b) >= 0.5).astype(int)
        return np.mean(preds == targets)


# ─────────────────────────────────────────
# 改进版 SVMFromScratch
# ─────────────────────────────────────────
class SVMFromScratchV2:
    """
    改进点：
    1. 特征标准化（StandardScaler）
    2. Mini-batch SGD（batch_size=64）
    3. 学习率衰减
    """
    def __init__(self, lr=0.1, epochs=300, lam=0.01, batch_size=64, decay=0.005):
        self.lr         = lr
        self.epochs     = epochs
        self.lam        = lam
        self.batch_size = batch_size
        self.decay      = decay
        self.scaler     = StandardScaler()
        self.w = None
        self.b = 0.0

    def train(self, train_data, train_targets):
        X = self.scaler.fit_transform(train_data)
        n, d = X.shape
        y = np.where(train_targets == 1, 1, -1).astype(float)
        self.w = np.zeros(d)
        self.b = 0.0
        rng = np.random.default_rng(42)

        for epoch in range(self.epochs):
            lr_t = self.lr / (1.0 + self.decay * epoch)
            # 随机打乱，按 batch 更新
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, yb = X[batch], y[batch]
                margins = yb * (Xb @ self.w + self.b)
                mask = margins < 1
                dw = self.lam * self.w - (Xb[mask] * yb[mask, None]).mean(axis=0) if mask.any() else self.lam * self.w
                db = -yb[mask].mean() if mask.any() else 0.0
                self.w -= lr_t * dw
                self.b -= lr_t * db

    def evaluate(self, data, targets):
        X = self.scaler.transform(data)
        y = np.where(targets == 1, 1, -1)
        preds = np.sign(X @ self.w + self.b)
        preds[preds == 0] = 1
        return np.mean(preds == y)


# ─────────────────────────────────────────
# 实验 1：C 对 LRModel 的影响
# ─────────────────────────────────────────
def exp_lr_C(data_list, target_list):
    print("\n" + "="*55)
    print("实验1：正则化系数 C 对 LRModel 的影响")
    print("="*55)
    print(f"{'C值':>10}  {'训练准确率':>12}  {'测试准确率':>12}")
    print("-"*40)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        tr, te = avg_accuracy(LogisticRegression, data_list, target_list,
                              C=C, max_iter=2000, solver='lbfgs')
        print(f"{C:>10.3f}  {tr:>12.4f}  {te:>12.4f}")


# ─────────────────────────────────────────
# 实验 2：C 对 SVMModel 的影响
# ─────────────────────────────────────────
def exp_svm_C(data_list, target_list):
    print("\n" + "="*55)
    print("实验2：正则化系数 C 对 SVMModel 的影响")
    print("="*55)
    print(f"{'C值':>10}  {'训练准确率':>12}  {'测试准确率':>12}")
    print("-"*40)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        tr, te = avg_accuracy(LinearSVC, data_list, target_list,
                              C=C, max_iter=3000)
        print(f"{C:>10.3f}  {tr:>12.4f}  {te:>12.4f}")


# ─────────────────────────────────────────
# 实验 3：手写版 v1 vs 改进版 v2
# ─────────────────────────────────────────
def run_scratch(model_cls, data_list, target_list, **kwargs):
    train_accs, test_accs = [], []
    for (train_data, test_data), (train_tgt, test_tgt) in zip(data_list, target_list):
        m = model_cls(**kwargs)
        m.train(train_data, train_tgt)
        train_accs.append(m.evaluate(train_data, train_tgt))
        test_accs.append(m.evaluate(test_data, test_tgt))
    return np.mean(train_accs), np.mean(test_accs)


# ── 原版手写 LR ──
class LRFromScratchV1:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr; self.epochs = epochs
        self.w = None; self.b = 0.0
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    def train(self, train_data, train_targets):
        n, d = train_data.shape
        self.w = np.zeros(d); self.b = 0.0
        for _ in range(self.epochs):
            diff = self._sigmoid(train_data @ self.w + self.b) - train_targets
            self.w -= self.lr * (train_data.T @ diff) / n
            self.b -= self.lr * diff.mean()
    def evaluate(self, data, targets):
        preds = (self._sigmoid(data @ self.w + self.b) >= 0.5).astype(int)
        return np.mean(preds == targets)

# ── 原版手写 SVM ──
class SVMFromScratchV1:
    def __init__(self, lr=0.01, epochs=200, lam=0.01):
        self.lr = lr; self.epochs = epochs; self.lam = lam
        self.w = None; self.b = 0.0
    def train(self, train_data, train_targets):
        n, d = train_data.shape
        y = np.where(train_targets == 1, 1, -1).astype(float)
        self.w = np.zeros(d); self.b = 0.0
        for _ in range(self.epochs):
            margins = y * (train_data @ self.w + self.b)
            mask = margins < 1
            dw = self.lam * self.w - (train_data[mask] * y[mask, None]).mean(axis=0) if mask.any() else self.lam * self.w
            db = -y[mask].mean() if mask.any() else 0.0
            self.w -= self.lr * dw; self.b -= self.lr * db
    def evaluate(self, data, targets):
        y = np.where(targets == 1, 1, -1)
        preds = np.sign(data @ self.w + self.b); preds[preds == 0] = 1
        return np.mean(preds == y)


def exp_scratch_compare(data_list, target_list):
    print("\n" + "="*55)
    print("实验3：手写版 v1（原版）vs v2（改进版）对比")
    print("="*55)
    print(f"{'模型':^28}  {'训练准确率':>10}  {'测试准确率':>10}")
    print("-"*55)

    tr, te = run_scratch(LRFromScratchV1, data_list, target_list)
    print(f"{'LRFromScratch  v1（原版）':^28}  {tr:>10.4f}  {te:>10.4f}")

    tr, te = run_scratch(LRFromScratchV2, data_list, target_list)
    print(f"{'LRFromScratch  v2（标准化+正则+衰减）':^28}  {tr:>10.4f}  {te:>10.4f}")

    tr, te = run_scratch(SVMFromScratchV1, data_list, target_list)
    print(f"{'SVMFromScratch v1（原版）':^28}  {tr:>10.4f}  {te:>10.4f}")

    tr, te = run_scratch(SVMFromScratchV2, data_list, target_list)
    print(f"{'SVMFromScratch v2（标准化+mini-batch）':^28}  {tr:>10.4f}  {te:>10.4f}")


# ─────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("加载数据中...")
    data_list, target_list = data_preprocess()
    print(f"数据加载完成，共 {len(data_list)} 个任务。\n")

    exp_lr_C(data_list, target_list)
    exp_svm_C(data_list, target_list)
    exp_scratch_compare(data_list, target_list)

    print("\n全部实验完成。")
