# 五子棋AI项目

这个项目是一个使用PyTorch实现的五子棋AI，其中Player1是主要训练目标，而Player2作为陪练模型帮助Player1提升。项目利用深度学习技术，通过神经网络模型来模拟玩家下棋的策略。

## 特点

- 使用PyTorch实现神经网络模型
- 支持CPU同时支持CUDA加速
- Player1与Player2交替对战，Player1作为主要训练目标
- 模型权重可以在训练期间会定期保存
- 使用Adam优化器进行模型参数优化，交叉熵损失函数用于评估预测动作的准确性

## 依赖

确保你已经安装了以下依赖：

- Python 3.x
- PyTorch
- NumPy

## 使用方法

### 安装环境

```shell
conda create --name gomoku-ai python=3.8
conda activate gomoku-ai
pip install -r requirements.txt
```

### 训练模型

运行以下命令开始训练模型：
```shell
python train.py
```

在训练过程中，每一千个回合会保存一次Player1的模型权重，你可以在当前目录下找到名为gobang_model_player1_<episode>.pth的文件。

### 使用GPU

默认情况下，代码会自动检测是否有可用的GPU，并在有GPU时使用CUDA。如果你希望强制使用或不使用GPU，可以手动设置USE_GPU标志位。

```python
USE_GPU = True  # 使用GPU
USE_GPU = False  # 不使用GPU
```

### 加载模型

你可以使用以下代码加载保存的模型：

```python
import torch
from model import GomokuNet

model = GomokuNet()
model.load_state_dict(torch.load('gobang_best_model.pth'))
model.eval()
```

## 贡献

欢迎您贡献代码！如果你有任何改进建议或发现了问题，请提交Pull Request或者直接创建issue。
