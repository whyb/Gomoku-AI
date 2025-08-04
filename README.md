# 五子棋AI项目

这个项目是一个使用PyTorch实现的五子棋AI，其中Player1是主要训练目标，而Player2作为陪练模型帮助Player1提升。项目利用深度学习技术，通过神经网络模型来模拟玩家下棋的策略。

## 特点

- 使用 PyTorch 2.6.0+cu126 实现神经网络模型，支持 GPU 加速
- 训练及推理验证均通过 PyTorch 内部机制自动检测并使用 GPU
- Player1 与 Player2 交替对战，Player1 作为主要训练目标
- 模型权重在训练期间会定期保存，支持断点续训
- 使用 Adam 优化器进行模型参数优化，交叉熵损失函数用于评估预测动作的准确性
- 支持自定义棋盘尺寸和胜利条件


## 依赖

确保你已经安装了以下依赖：

- Python 3.10+
- torch 2.6.0+cu126（支持 CUDA 12.6 的 GPU 版本）
- numpy 1.26.4

## 使用方法

### 安装环境

```shell
conda create --name gomoku-ai python=3.10
conda activate gomoku-ai
pip install -r requirements.txt
```

### 训练模型

运行以下命令开始训练模型，需指定棋盘尺寸和胜利条件：
```shell
# 4x4棋盘，连4子胜利
python train.py --board_size 4 --win_condition 4

# 8x8棋盘，连5子胜利
python train.py --board_size 8 --win_condition 5

# 10x10棋盘，连5子胜利
python train.py --board_size 10 --win_condition 5

```

训练过程中，每 `config.SAVE_INTERVAL` 回合会保存一次 Player1 的模型权重，生成 `gobang_model_player1_*.pth` 文件

训练结束后会生成 `gobang_best_model.pth` 作为最终权重文件，支持从该文件继续训练


### 使用GPU

代码会通过 PyTorch 自动检测并使用可用的 GPU，无需手动配置：

* 若系统存在兼容的 NVIDIA GPU 且安装了对应 CUDA 版本，会自动启用 GPU 加速
* 若无 GPU，会自动 fallback 到 CPU 模式运行

### 加载模型

你可以使用以下代码加载保存的模型：

```python
import torch
from model import GomokuNetV2

# 需指定与训练时一致的棋盘尺寸
model = GomokuNetV2(board_size=4)
model.load_state_dict(torch.load('gobang_best_model.pth'))
model.eval()
```

### 转换ONNX

您可以将训练好的模型转换为 ONNX 格式 和 torchscript 模型（需指定棋盘尺寸和胜利条件）：

```shell
# 基础用法（4x4棋盘，连4子胜利）
python export_onnx.py gobang_best_model.pth --board_size 4 --win_condition 4

# 自定义输出路径
python export_onnx.py gobang_best_model.pth \
  --board_size 4 --win_condition 4 \
  --onnx_path ./output/model_4x4.onnx \
  --torchscript_path ./output/model_4x4.pt

```
执行成功后，会在output目录产生`model_4x4.pt`、`model_4x4.onnx`两个文件

### 打印棋盘

如果您想要查看每一次训练或验证阶段打印当下棋局结果，观察过程，分析训练效果，有一个`NEED_PRINT_BOARD`变量可以控制辅助到您，设置成`True`或`False`开启或关闭。
在验证阶段也可在val.py中设置`NEED_PRINT_BOARD = True`，可在每局结束后打印棋盘布局
棋盘打印效果如下：
```
. . . . X X X . . . . X O . .
. . O O . X . . . . . O . . .
O X X . X X O X X O . . . . .
X O O . . X X O X . . X . O .
O O X . . . O . X O X O . . X
. X . . X O . . O X . . O . X
O . X . X X O . . . X . X . X
O . . . O . O O . . . X X X .
. . X X . O . . . . O . X O .
O O . . . . . . O O . . . . .
. . X O O . O . X . X O . O .
O X . . X . . O O . X . . . .
. X . O . O . . . . . O . O O
X X . . . X X X O X . O O . .
X O O X . . X O X . . . . . O
```

## 欢迎贡献

欢迎您贡献代码！如果你有任何改进建议或发现了问题，请提交Pull Request或者直接在本仓库创建issue。
