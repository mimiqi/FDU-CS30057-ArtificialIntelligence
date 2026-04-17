# 随堂练习1 完成规划

## 作业概览

| 项目 | 内容 |
|------|------|
| 截止时间 | 2026年4月19日 23:59 |
| 总分 | 3分 |
| 提交格式 | `学号_姓名_课堂练习1.zip`（代码 + 报告） |
| 核心任务 | 用 LR 和 SVM 对 1357 个蛋白质做 55 个二分类任务 |

---

## 数据理解

### 文件结构

```
data/
├── diagrams.npy                              # 特征矩阵，shape=(1357, 300)
├── SCOP40mini_sequence_minidatabase_19.cast  # 标签文件，1357行×56列（列0为蛋白ID，列1-55为55个任务）
└── SCOP40mini/*.ent                          # PDB结构文件（可视化用，非建模必须）
```

### `.cast` 标签含义

| 值 | 含义 | 转换目标 |
|----|------|----------|
| `1` | 正样本，训练集 | label=1, train |
| `2` | 负样本，训练集 | label=0, train |
| `3` | 正样本，测试集 | label=1, test |
| `4` | 负样本，测试集 | label=0, test |
| `0` | 不参与该任务 | 跳过 |

---

## 任务拆解与完成顺序

### ✅ Step 1：完成 `data_preprocess()`（两个文件都要改，基础）

**位置：** `main_lr.py` 第60行 / `main_svm.py` 第58行的 `## todo` 处

将 `## todo` 替换为以下代码：

```python
train_mask = task_col.isin([1, 2])
test_mask  = task_col.isin([3, 4])

train_data    = diagrams[train_mask.values]
test_data     = diagrams[test_mask.values]
train_targets = (task_col[train_mask] == 1).astype(int).values
test_targets  = (task_col[test_mask]  == 3).astype(int).values
```

#### 逐行解析

> **背景：** 外层循环 `for task in range(1, 56)` 每次处理一个分类任务。`task_col = cast.iloc[:, task]` 取出当前任务列，其中每个蛋白质对应一个值（0/1/2/3/4），表示它在本任务中的角色。

```python
train_mask = task_col.isin([1, 2])
```
- `task_col.isin([1, 2])` 对整列做逐元素判断，值为 `1`（正训练样本）或 `2`（负训练样本）时返回 `True`，其余为 `False`。
- `train_mask` 是一个长度为 1357 的布尔 Series，`True` 的位置即为"属于训练集"的蛋白质。

```python
test_mask = task_col.isin([3, 4])
```
- 同理，值为 `3`（正测试样本）或 `4`（负测试样本）的位置标记为 `True`。
- `test_mask` 对应"属于测试集"的蛋白质。

```python
train_data = diagrams[train_mask.values]
```
- `diagrams` 形状为 `(1357, 300)`，每行是一个蛋白质的 300 维特征向量。
- `.values` 将 Pandas Series 转为 NumPy 布尔数组，才能用于 NumPy 数组的行索引。
- 结果 `train_data` 是 `(N_train, 300)` 的特征矩阵，只保留训练集蛋白质的行。

```python
test_data = diagrams[test_mask.values]
```
- 同理，提取测试集蛋白质的特征，形状为 `(N_test, 300)`。

```python
train_targets = (task_col[train_mask] == 1).astype(int).values
```
- `task_col[train_mask]`：筛选出训练集蛋白质对应的标签值（只含 `1` 和 `2`）。
- `== 1`：值为 `1`（正样本）的位置返回 `True`，值为 `2`（负样本）返回 `False`。
- `.astype(int)`：将布尔值转为整数，`True→1`，`False→0`，完成标签二值化。
- 结果 `train_targets` 是长度为 `N_train` 的 0/1 标签数组，`1` 表示正类，`0` 表示负类。

```python
test_targets = (task_col[test_mask] == 3).astype(int).values
```
- 与上一行逻辑相同，但测试集的正样本标签为 `3`、负样本为 `4`，因此用 `== 3` 来区分正负。
- 结果 `test_targets` 是长度为 `N_test` 的 0/1 标签数组。

#### 数据流总结

```
diagrams.npy (1357×300)  +  cast 第 task 列 (1357个值)
        │                           │
        │         train_mask (值∈{1,2} → True)
        ├──────────────────────────►├── train_data    shape: (N_train, 300)
        │                           └── train_targets shape: (N_train,)  值: 1或0
        │
        │         test_mask (值∈{3,4} → True)
        └──────────────────────────►├── test_data     shape: (N_test, 300)
                                    └── test_targets  shape: (N_test,)   值: 1或0
```

---

### ✅ Step 2：完成 `LRModel`（第1分，sklearn）

**位置：** `main_lr.py` 第4-35行

```python
from sklearn.linear_model import LogisticRegression

class LRModel:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)
```

#### 逐行解析

```python
from sklearn.linear_model import LogisticRegression
```
- 从 sklearn 库中导入逻辑回归分类器。`sklearn`（scikit-learn）是 Python 最常用的机器学习工具包。

```python
def __init__(self):
    self.model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
```
- `__init__` 是 Python 类的**构造方法**，在 `LRModel()` 创建实例时自动调用。
- `self.model` 把 sklearn 模型对象保存为实例属性，后续 `train` 和 `evaluate` 都通过它来操作。
- 三个超参数：
  - `C=1.0`：正则化强度的**倒数**，`C` 越大正则化越弱（越允许过拟合），`C` 越小正则化越强。
  - `max_iter=1000`：梯度优化的最大迭代次数，防止不收敛时无限循环。
  - `solver='lbfgs'`：优化算法，L-BFGS 是一种拟牛顿法，适合中小规模数据。

```python
def train(self, train_data, train_targets):
    self.model.fit(train_data, train_targets)
```
- `fit(X, y)` 是 sklearn 统一的训练接口：传入特征矩阵 `X` 和标签向量 `y`，内部自动完成参数拟合。
- 训练完成后，模型的权重 `w` 和偏置 `b` 存储在 `self.model.coef_` 和 `self.model.intercept_` 中。

```python
def evaluate(self, data, targets):
    return self.model.score(data, targets)
```
- `score(X, y)` 是 sklearn 分类器的内置评估方法：先用训好的模型对 `X` 做预测，再与真实标签 `y` 对比，返回**准确率**（正确预测数 / 总数）。

---

### ✅ Step 3：完成 `SVMModel`（第1分，sklearn）

**位置：** `main_svm.py` 第4-34行

```python
from sklearn.svm import SVC

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel='linear', C=1.0)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)
```

#### 逐行解析

```python
from sklearn.svm import SVC
```
- 导入 sklearn 的支持向量分类器 `SVC`（Support Vector Classification）。

```python
def __init__(self):
    self.model = SVC(kernel='linear', C=1.0)
```
- `kernel='linear'`：使用**线性核**，即决策边界为超平面 $w \cdot x + b = 0$。线性核适合本任务中 300 维特征的中小数据集。其他可选核如 `'rbf'`（高斯核，非线性）、`'poly'`（多项式核）。
- `C=1.0`：惩罚系数，控制对误分类样本的惩罚力度。`C` 越大，模型越不容忍误分类（硬间隔方向）；`C` 越小，越允许部分样本在间隔内或被误分（软间隔方向）。

```python
def train(self, train_data, train_targets):
    self.model.fit(train_data, train_targets)
```
- 与 `LRModel` 接口完全一致。`fit` 内部通过**二次规划**求解支持向量和最优超平面。

```python
def evaluate(self, data, targets):
    return self.model.score(data, targets)
```
- 同样调用 `score` 返回准确率。SVC 的预测逻辑是：计算 $w \cdot x + b$ 的符号来判断类别。

> **Step 2 与 Step 3 的结构对比：** 两者的 `train/evaluate` 接口完全相同（sklearn 统一API），区别仅在 `__init__` 中创建的模型对象不同（`LogisticRegression` vs `SVC`），体现了不同的分类算法思想。

> 运行验证：`python main_lr.py` 和 `python main_svm.py` 均能输出准确率后再继续。

---

### ✅ Step 4：完成 `LRFromScratch`（第2分，手写梯度下降）

**数学原理：**
- 预测：$\hat{y} = \sigma(Xw + b)$，$\sigma(z) = \frac{1}{1+e^{-z}}$
- 梯度：$\nabla_w L = \frac{1}{n}X^T(\hat{y}-y)$，$\nabla_b L = \frac{1}{n}\sum(\hat{y}-y)$
- 更新：$w \leftarrow w - \eta \nabla_w L$

**位置：** `main_lr.py` 第38-47行

```python
class LRFromScratch:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def train(self, train_data, train_targets):
        n, d = train_data.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = train_data @ self.w + self.b
            y_hat = self._sigmoid(z)
            diff = y_hat - train_targets
            self.w -= self.lr * (train_data.T @ diff) / n
            self.b -= self.lr * diff.mean()

    def evaluate(self, data, targets):
        z = data @ self.w + self.b
        preds = (self._sigmoid(z) >= 0.5).astype(int)
        return np.mean(preds == targets)
```

#### 逐行解析

**`__init__`（初始化超参数）**

```python
def __init__(self, lr=0.1, epochs=200):
    self.lr = lr          # 学习率 η，控制每次参数更新的步幅
    self.epochs = epochs  # 训练轮数，整个数据集遍历多少次
    self.w = None         # 权重向量，训练时初始化
    self.b = 0.0          # 偏置标量
```
- `lr=0.1` 和 `epochs=200` 是**默认参数**，调用时可用 `LRFromScratch(lr=0.01, epochs=500)` 覆盖。
- `self.w = None` 先占位，因为特征维度 `d` 在 `train` 时才知道。

**`_sigmoid`（激活函数）**

```python
def _sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
```
- 实现 Sigmoid 函数：$\sigma(z) = \frac{1}{1+e^{-z}}$，将任意实数压缩到 $(0, 1)$ 区间，作为"属于正类的概率"。
- `np.clip(z, -500, 500)`：将 `z` 裁剪到 $[-500, 500]$ 范围内，防止 `np.exp` 溢出（`exp(710)` 在 float64 下会变成 `inf`）。
- 方法名前缀 `_` 是 Python 惯例，表示"仅供类内部使用的私有方法"。

**`train`（梯度下降训练）**

```python
n, d = train_data.shape
```
- **元组解包**：`train_data.shape` 返回 `(样本数, 特征维度)`，一步赋值给 `n` 和 `d`。

```python
self.w = np.zeros(d)
self.b = 0.0
```
- 将权重初始化为全零向量（长度 `d=300`），偏置初始化为 `0`。

```python
for _ in range(self.epochs):
```
- 循环 200 轮。`_` 是 Python 惯例，表示"不需要使用循环变量"。

```python
    z = train_data @ self.w + self.b
```
- `@` 是矩阵乘法运算符（Python 3.5+）。`train_data` 形状 `(n, d)` 乘以 `self.w` 形状 `(d,)`，得到 `z` 形状 `(n,)`。
- `+ self.b`：NumPy **广播**，标量自动加到向量每个元素上。
- 等价于数学公式 $z = Xw + b$。

```python
    y_hat = self._sigmoid(z)
```
- 将线性输出 `z` 通过 Sigmoid 转换为预测概率 $\hat{y} \in (0,1)$，形状仍为 `(n,)`。

```python
    diff = y_hat - train_targets
```
- 预测值与真实标签的逐元素差值，形状 `(n,)`。这是二元交叉熵损失对 $z$ 的梯度的核心部分。

```python
    self.w -= self.lr * (train_data.T @ diff) / n
```
- `train_data.T @ diff`：`(d, n) @ (n,)` → `(d,)`，即 $X^T(\hat{y}-y)$，是损失对 `w` 的梯度。
- `/ n`：除以样本数取均值。
- `self.w -= ...`：原地减法，即 $w \leftarrow w - \eta \cdot \frac{1}{n}X^T(\hat{y}-y)$，梯度下降更新。

```python
    self.b -= self.lr * diff.mean()
```
- `diff.mean()` 即 $\frac{1}{n}\sum(\hat{y}_i - y_i)$，是损失对 `b` 的梯度。
- 偏置同样按梯度下降更新。

**`evaluate`（预测与评估）**

```python
z = data @ self.w + self.b
preds = (self._sigmoid(z) >= 0.5).astype(int)
```
- 先算出概率 $\hat{y}$，然后用阈值 `0.5` 将概率转成 0/1 预测：$\geq 0.5$ 判为正类（`1`），否则判为负类（`0`）。
- `>= 0.5` 返回布尔数组，`.astype(int)` 转为整数。

```python
return np.mean(preds == targets)
```
- `preds == targets` 逐元素比较，返回布尔数组。`np.mean` 对布尔数组取均值就是"正确比例"，即**准确率**。

---

### ✅ Step 5：完成 `SVMFromScratch`（第3分，软间隔次梯度下降）

**数学原理（软间隔 SVM）：**
- 目标：$\min_{w,b} \frac{\lambda}{2}\|w\|^2 + \frac{1}{n}\sum\max(0, 1 - y_i(w \cdot x_i + b))$
- 标签转换：$y \in \{-1, +1\}$
- 次梯度：对违反间隔的样本（$1 - y_i(w\cdot x_i+b) > 0$），梯度加上 $-y_i x_i / n$

**位置：** `main_svm.py` 第36-45行

```python
class SVMFromScratch:
    def __init__(self, lr=0.01, epochs=200, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.w = None
        self.b = 0.0

    def train(self, train_data, train_targets):
        n, d = train_data.shape
        y = np.where(train_targets == 1, 1, -1).astype(float)
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            margins = y * (train_data @ self.w + self.b)
            mask = margins < 1
            dw = self.lam * self.w - (train_data[mask] * y[mask, None]).mean(axis=0) if mask.any() else self.lam * self.w
            db = -y[mask].mean() if mask.any() else 0.0
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def evaluate(self, data, targets):
        y = np.where(targets == 1, 1, -1)
        preds = np.sign(data @ self.w + self.b)
        preds[preds == 0] = 1
        return np.mean(preds == y)
```

#### 逐行解析

**`__init__`（初始化超参数）**

```python
def __init__(self, lr=0.01, epochs=200, lam=0.01):
    self.lr = lr        # 学习率
    self.epochs = epochs  # 训练轮数
    self.lam = lam      # 正则化系数 λ，控制 ||w||² 的惩罚力度
    self.w = None
    self.b = 0.0
```
- 比 LR 多了一个超参数 `lam`（正则化系数）。SVM 的目标函数中包含 $\frac{\lambda}{2}\|w\|^2$ 项，`lam` 越大则越倾向于让 `w` 更小（间隔更大），但可能欠拟合。
- `lr=0.01` 比 LR 的 `0.1` 小，因为 SVM 的次梯度不够光滑，大学习率容易震荡。

**`train`（次梯度下降训练）**

```python
y = np.where(train_targets == 1, 1, -1).astype(float)
```
- `np.where(条件, 真值, 假值)`：**SVM 要求标签为 $\{-1, +1\}$**，而数据预处理得到的是 $\{0, 1\}$，所以需要映射：`1→+1`，`0→-1`。
- `.astype(float)` 转成浮点数，方便后续矩阵运算。

```python
margins = y * (train_data @ self.w + self.b)
```
- `train_data @ self.w + self.b`：计算每个样本的决策值 $f(x_i) = w \cdot x_i + b$，形状 `(n,)`。
- `y * (...)`：逐元素相乘，得到**函数间隔** $y_i \cdot f(x_i)$。间隔越大，说明分类越正确且越有信心。

```python
mask = margins < 1
```
- 布尔掩码：找出所有**函数间隔 < 1** 的样本。这些样本要么被误分类（`margin < 0`），要么虽然分对但离决策边界太近（`0 < margin < 1`）。
- 在 SVM 的 Hinge Loss $\max(0, 1 - y_i f(x_i))$ 中，只有 `margin < 1` 的样本对梯度有贡献。

```python
dw = self.lam * self.w - (train_data[mask] * y[mask, None]).mean(axis=0) if mask.any() else self.lam * self.w
```
- 这是一个**三元表达式**（`A if 条件 else B`），分两种情况：
- **有违反间隔的样本时（`mask.any()` 为 `True`）：**
  - `y[mask, None]`：`None` 增加一维，从 `(k,)` 变成 `(k, 1)`，使得能与 `(k, d)` 的特征矩阵逐行相乘（广播机制）。
  - `train_data[mask] * y[mask, None]`：每行特征乘以对应标签，得到 `(k, d)`。
  - `.mean(axis=0)`：沿样本维度取均值，得到 `(d,)` 向量，即 Hinge Loss 对 `w` 的次梯度均值。
  - `self.lam * self.w - ...`：加上正则项的梯度（$\lambda w$）。
- **没有违反间隔的样本时：** 梯度仅有正则项 $\lambda w$。

```python
db = -y[mask].mean() if mask.any() else 0.0
```
- Hinge Loss 对偏置 `b` 的次梯度：违反间隔样本的标签均值的相反数。
- 若没有违反间隔的样本，偏置不更新。

```python
self.w -= self.lr * dw
self.b -= self.lr * db
```
- 标准梯度下降更新：参数沿负梯度方向移动。

**`evaluate`（预测与评估）**

```python
y = np.where(targets == 1, 1, -1)
```
- 评估时也需要把真实标签转为 $\{-1, +1\}$ 格式，以便与预测结果对比。

```python
preds = np.sign(data @ self.w + self.b)
```
- `np.sign` 取符号函数：正数→`+1`，负数→`-1`，零→`0`。即根据决策值 $w \cdot x + b$ 的符号来判断类别。

```python
preds[preds == 0] = 1
```
- 处理恰好 $w \cdot x + b = 0$ 的边界情况：将其归为正类（也可归为负类，这里只是一个约定）。

```python
return np.mean(preds == y)
```
- 与 LR 相同，逐元素比较后取均值得到准确率。

#### Step 4 与 Step 5 对比

| 对比项 | LRFromScratch | SVMFromScratch |
|--------|---------------|----------------|
| 标签格式 | $\{0, 1\}$ | $\{-1, +1\}$ |
| 核心函数 | Sigmoid → 概率 | 符号函数 → 决策边界 |
| 损失函数 | 二元交叉熵 | Hinge Loss + L2 正则 |
| 梯度类型 | 光滑梯度 | 次梯度（不可导点处取子梯度） |
| 预测方式 | 概率 ≥ 0.5 → 正类 | $w \cdot x + b > 0$ → 正类 |

---

### ✅ Step 6：课外探索（报告加分）

**超参数影响实验（LRModel）：**

```python
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = LRModel(C=C)
    # 记录55个任务的平均 train/test accuracy，画折线图
```

**手写版改进思路：**
- LR：增加 L2 正则项（损失加 $\frac{\lambda}{2}\|w\|^2$），或使用学习率衰减 `lr = lr0 / (1 + epoch * decay)`
- SVM：使用 Mini-batch SGD（每轮随机抽取部分样本），加快大数据集的收敛

**报告中建议包含的对比表：**

| 模型 | 平均训练准确率 | 平均测试准确率 |
|------|----------------|----------------|
| LRModel (sklearn) | | |
| LRFromScratch | | |
| SVMModel (sklearn) | | |
| SVMFromScratch | | |

---

## 提交前检查清单

- [ ] `main_lr.py`：`data_preprocess()` 补全
- [ ] `main_lr.py`：`LRModel` 补全（sklearn）
- [ ] `main_lr.py`：`LRFromScratch` 补全（手写）
- [ ] `main_svm.py`：`data_preprocess()` 补全
- [ ] `main_svm.py`：`SVMModel` 补全（sklearn）
- [ ] `main_svm.py`：`SVMFromScratch` 补全（手写）
- [ ] 两个文件均能无报错运行并输出 55 个任务的平均准确率
- [ ] 报告包含：运行结果截图、超参数实验图表、手写版与 sklearn 版对比分析
- [ ] 打包为 `学号_姓名_课堂练习1.zip`
