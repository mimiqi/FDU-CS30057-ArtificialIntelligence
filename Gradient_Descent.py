import numpy as np

# ========== 第一步：创建训练数据 ==========
# 这是我们用来训练模型的数据集
# 格式：[(x1, y1), (x2, y2), ...]
# 例如：x是输入特征，y是对应的输出标签
trainExamples = [
    (1, 2),      # 当输入是1时，输出是2
    (2, 4),      # 当输入是2时，输出是4
    (3, 6),      # 当输入是3时，输出是6
    (4, 8),      # 当输入是4时，输出是8
]

def trainLossLoop(w):
    """计算所有训练数据的总损失（均方误差）"""
    total_loss = 0
    for x, y in trainExamples:  # 遍历所有训练样本
        prediction = w.dot(phi(x))  # 预测值
        error = prediction - y  # 与真实值的差距
        total_loss += error ** 2  # 累加误差的平方
    return 1.0 / len(trainExamples) * total_loss  # 平均损失

def gradientTrainLossLoop(w):
    """计算损失函数的梯度（用于更新权重）"""
    total_gradient = np.zeros(2)  # 初始化梯度向量
    for x, y in trainExamples:  # 遍历所有训练样本
        prediction = w.dot(phi(x))
        error = prediction - y
        # 计算该样本的梯度贡献
        total_gradient += 2 * error * phi(x)
    return 1.0 / len(trainExamples) * total_gradient  # 平均梯度

def phi(x):
    """特征变换函数：将输入x转换为[1, x]的向量"""
    # 第一个1是偏置项（bias），第二个x是真实特征
    return np.array([1, x])

def initialWeightVector():
    """初始化权重向量，开始时都设为0"""
    return np.zeros(2)

def gradientDescent(F, gradientF, initialWeightvector):
    """
    梯度下降优化算法
    F: 损失函数
    gradientF: 梯度函数
    initialWeightvector: 初始权重
    """
    w = initialWeightVector()
    loss_list = []
    eta = 0.1  # 学习率：每次更新的步长大小
    
    for t in range(501):  # 总共迭代501次
        value = F(w)  # 计算当前损失
        gradient = gradientF(w)  # 计算梯度
        w = w - eta * gradient  # 更新权重：沿着梯度的反方向移动
        loss_list.append(value)
        
        # 定期打印训练进度
        if t % 25 == 0 or t < 10:
            print(f't = {t}, w = {w}, F(w) = {value}, gradientF = {gradient}')
    
    return w, loss_list

# ========== 第二步：执行梯度下降算法 ==========
print("开始训练模型...\n")
w_loop, loss_list_loop = gradientDescent(trainLossLoop, gradientTrainLossLoop, initialWeightVector)

# ========== 第三步：打印最终结果 ==========
print(f"\n训练完成！")
print(f"最优权重: {w_loop}")
print(f"最终损失: {loss_list_loop[-1]:.6f}")