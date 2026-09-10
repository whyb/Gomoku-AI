# Gomoku AI Project

[中文文档](README.md)

<p align="center">
    <a href="https://deepwiki.com/whyb/Gomoku-AI">
        <img src="https://img.shields.io/badge/DeepWiki-whyb%2FGomokuAI-yellow.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" />
    </a>
</p>

This project is a Gomoku AI implemented with PyTorch. Player 1 is the primary training target, while Player 2 acts as a sparring model that helps Player 1 improve. Inspired by the classic AlphaZero design, the project uses deep learning and a neural-network training pipeline based on a deep residual ResNet architecture and reinforcement learning to learn how to play Gomoku.

The new version (`train_alphazero.py`) uses AlphaZero-style two-stage training: it first distills knowledge from a traditional AI teacher (Kali-Hac) to rapidly acquire master-level positional judgment, then fine-tunes through MCTS self-play reinforcement learning to surpass the teacher. Compared with the older version, it offers higher training efficiency and a higher skill ceiling, and is **recommended**.

## Web Demo

[Play against the AI](https://whyb.github.io/Gomoku-AI/webdemo/)

[![Demo](webdemo/demo.png)](https://whyb.github.io/Gomoku-AI/webdemo/)

## Gomoku Rules

This project uses **Free-style Gomoku** rules:

- **Black (P1) moves first**, and the first move must be played at **tengen** (the exact center of the board).
- **White (P2) moves second** with no placement restrictions.
- A player wins by forming **five consecutive stones** in any direction.
- **Neither player is subject to forbidden moves**: Black may also win with double live threes, double blocked fours, an overline (at least 6 consecutive stones), and similar patterns.
- The default board is 15×15, and custom board sizes are supported.

> Unlike professional Renju, this project does not restrict Black's double-three, double-four, overline, or other forbidden patterns. Its rules are simpler and more general.

## Features

- Neural-network models are implemented with PyTorch 2.6.0+cu126 and support GPU acceleration.
- Training and inference validation automatically detect and use the GPU through PyTorch's native mechanisms.
- A residual network (ResNet) is used as the core architecture to learn board features effectively.
- Player 1 and Player 2 play against each other, with Player 1 as the primary training target.
- Model weights are saved periodically, allowing training to resume from the latest checkpoint for greater flexibility and stability.
- Training uses the Adam optimizer and jointly optimizes the model with cross-entropy loss for move policy and mean squared error loss for position value.
- Custom board sizes and win conditions make it easy to apply the model to different Gomoku variants.
- The dynamic-shape version allows the board sizes used during training and inference to differ, giving the model better generalization.
- Player 1 alternates first-move games against the Kali-Hac Gomoku AI, with Player 1 as the training target. (Legacy dynamic-model training method.)
- AlphaZero-style training combines MCTS self-play, an opponent pool, D4 symmetry augmentation, and a SE-ResNet architecture so the model can continue to improve.
- Two-stage training (recommended) first uses knowledge distillation from a traditional AI teacher (Kali-Hac) to quickly reach master-level positional judgment, then fine-tunes through MCTS self-play to surpass the teacher.

## Dependencies

Make sure the following dependencies are installed:

- Python 3.10+
- torch 2.6.0+cu126 (GPU build with CUDA 12.6 support)
- numpy 1.26.4

## Usage

### Set Up the Environment

```shell
conda create --name gomoku-ai python=3.10
conda activate gomoku-ai
pip install -r requirements.txt
```

### Train a Model

Run one of the following commands to start training. You must specify the board size and win condition.

#### Legacy Training (Not Recommended)

<details>
  <summary>Click to expand legacy training commands (not recommended)</summary>

```shell
# 8x8 board, five-in-a-row win condition, fixed-shape model; inference export is limited to 8x8
python train.py --board_size 8 --win_condition 5

# 15x15 board, five-in-a-row win condition, fixed-shape model; inference export is limited to 15x15
python train.py --board_size 15 --win_condition 5

# Train on an 8x8 board (any other size also works), five-in-a-row win condition, dynamic-shape model;
# export_onnx_dy.py can later export a model for any board size (recommended)
python train_dy.py --board_size 8 --win_condition 5
```

During training, Player 1's model weights are saved every `config.SAVE_INTERVAL` episodes as `gobang_model_player1_*.pth` (fixed-shape model) or `gobang_model_player1_dy_step_*.pth` (dynamic-shape model).

After training completes, `gobang_best_model.pth` (fixed-shape model) and `gobang_best_model_dy.pth` (dynamic-shape model) are generated as the final weights, and training can be resumed from either file.

</details>

#### AlphaZero-Style Training (Recommended)

This AlphaZero-based training pipeline replaces fixed-opponent training with **MCTS self-play**, allowing continuous improvement:

```shell
# Standard model on a 15×15 board (recommended for formal training)
python train_alphazero.py --board_size 15 --num_simulations 400 --model standard --fp16

# Small model on a 15×15 board (for quick experiments)
python train_alphazero.py --board_size 15 --num_simulations 400 --model small --fp16
```

### Training Visualization (TensorBoard)

`train_alphazero.py` enables TensorBoard logging by default, covering both the distillation and MCTS self-play stages. During training, metrics such as loss, Top-K accuracy, Elo rating, and throughput are written to the `runs/` directory:

```shell
tensorboard --logdir runs
```

Open `http://localhost:6006` in a browser to view the training curves.

Common options:

- `--log_dir`: Log directory (default: `runs`)
- `--no_tensorboard`: Disable logging

### Two-Stage Training (Recommended)

Starting MCTS reinforcement learning directly from random initialization converges very slowly because RL rewards are sparse. This project therefore uses **two-stage training**: first rapidly imitate a master through knowledge distillation from a traditional AI teacher, then surpass that teacher through self-play MCTS fine-tuning:

```text
Stage 1: Distillation (--distill)       Stage 2: MCTS Fine-Tuning (normal training)
┌───────────────────────┐              ┌──────────────────────┐
│ Teacher AI (Kali-Hac) │ Load weights │ MCTS + Self-play     │
│   ↓ Self-play          │ ──────────→  │   ↓                  │
│ (state, π_teacher, z) │ Auto-switch  │ RL fine-tuning       │
│   ↓                   │              │   ↓                  │
│ KL divergence + MSE   │              │ Surpass teacher      │
└───────────────────────┘              └──────────────────────┘
```

#### Stage 1: Knowledge Distillation — Rapidly Imitate the Master

**Goal**: Let a randomly initialized network quickly learn the teacher AI's positional judgment ("look like" the master).

| Property | Description |
|----------|-------------|
| Data source | Teacher AI (Kali-Hac) self-play, not MCTS |
| Loss function | `L = KL(teacher_soft \|\| student_soft) × T² + λ × MSE(v, z)` |
| Temperature T | Default 3.0 (`--distill_temperature`); teacher scores are log-compressed and temperature-scaled to produce a softer probability distribution |
| Value-head weight | Default 0.5 (`--distill_value_weight`); the teacher has no precise value estimates, so training focuses on policy matching |
| Random openings | Default 20% (`--distill_random_frac`); each selected game starts with 4–12 random moves, forcing the teacher to handle "messy" positions |
| Data efficiency | Pure supervised learning converges extremely quickly (50,000 games can reach Top-1 accuracy above 80%) |
| Recommended games | 10×10: 50,000 games; 15×15: 20,000–30,000 games; 5×5: 10,000 games |

#### Stage 2: MCTS Fine-Tuning — Surpass the Teacher

When `--distill` is disabled, normal training **automatically loads the distillation weights** as the initial weights (preferring `*_distill_best.pth`, then `*_distill.pth`) and performs standard AlphaZero MCTS self-play fine-tuning to break through the teacher's skill ceiling.

```shell
# Stage 1: Distillation (15×15, 20,000 games, approximately 1–3 hours)
python train_alphazero.py --board_size 15 --model standard --distill --distill_games 20000

# Stage 2: MCTS fine-tuning (automatically loads the distillation weights)
python train_alphazero.py --board_size 15 --model standard --num_simulations 400
```

#### Why Are Two Stages Needed?

| Dimension | Distillation Only | MCTS Only (Random Initialization) | Distillation → MCTS |
|-----------|-------------------|------------------------------------|---------------------|
| Training speed | Fast (supervised learning) | Slow (sparse RL rewards) | Fast → Slow |
| Skill ceiling | Equal to the teacher | Can surpass the teacher | **Can surpass the teacher** |
| Generalization | Poor (vulnerable to unsound moves) | Strong (exploration-driven) | **Strong** |
| State distribution | Narrow (teacher style) | Broad (MCTS exploration) | Narrow → Broad |

> **Distribution shift**: All distillation data comes from teacher self-play, so its state space is narrow. The student is prone to errors when facing opponents that do not follow the teacher's patterns, such as random play or unusual MCTS moves. `--distill_random_frac` can only partially mitigate this. The fundamental solution is MCTS fine-tuning, in which the model explores a massive number of new states through self-play.
>
> **Distillation collapse protection (implemented)**: Best-model tracking (saving `*_distill_best.pth` whenever Top-1 improves by at least 0.5%), three-level collapse detection (Top-1 falls to zero / drops 70% relative to the best value / loss increases 3×), automatic recovery from the best checkpoint, and early stopping after 20 consecutive evaluations without improvement.

### Training Volume Estimates (15×15 Board, 400 MCTS Simulations per Move)

| Scenario | Small Model (`--model small`) | Standard Model (`--model standard`) | Expected Strength |
| :--- | :--- | :--- | :--- |
| 🟢 Beginner | 5,000–10,000 | — | Can consistently beat random play |
| 🟡 Amateur | 30,000–50,000 | ~50,000 | Understands basic tactics such as live threes and blocked fours |
| 🟠 Strong amateur | 80,000–100,000 | 150,000–200,000 | Mature tactical awareness; beats most humans; **the small model approaches its capacity limit** |
| 🔴 Expert | >300,000 (not recommended) | 300,000–500,000 | Accurate tactical judgment with balanced attack and defense |
| 🏆 Superhuman | Not achievable | 1,000,000+ | Approaches the theoretical limit of this architecture |

> **Recommendation**: For a 15×15 board, use `--model standard` (128 channels, 10-layer SE-ResNet, approximately 3M parameters). The small model (approximately 460K parameters) is capacity-limited and will struggle to progress beyond strong-amateur level on a 15×15 board, regardless of how many games it is trained on.

#### Training Output Files

The following files are saved automatically during training (example based on the standard model):

| File | Description |
|------|-------------|
| `alpaz_standard_15x15_model.pth` | Pure model weights (used for ONNX export) |
| `alpaz_standard_15x15_checkpoint.pth` | Full checkpoint (includes optimizer and scheduler state; supports resuming) |
| `alpaz_standard_15x15_opponent_pool.pth` | Opponent pool (contains historical model snapshots) |
| `alpaz_standard_15x15_elo.json` | Elo rating history |
| `alpaz_standard_15x15_distill.pth` | Final distillation weights (loaded automatically when `--distill` is disabled; starting point for MCTS fine-tuning) |
| `alpaz_standard_15x15_distill_best.pth` | Best distillation weights (saved whenever Top-1 reaches a new high; restored automatically by collapse detection) |
| `alpaz_standard_15x15_best.pth` | Best model from the MCTS fine-tuning stage (saved whenever Elo reaches a new high) |

Compared with the traditional training approach, the AlphaZero pipeline provides the following features:

- **MCTS (Monte Carlo Tree Search)**: Uses PUCT selection and 200–800 simulations per move to generate a high-quality move policy π.
- **Self-play training**: The model plays against itself to generate training data and no longer depends on a fixed opponent.
- **Opponent pool**: Historical model snapshots are saved automatically. During training, there is a 50% chance of playing against a historical version to prevent catastrophic forgetting.
- **D4 symmetry augmentation**: Each experience is automatically transformed with 8 rotations and reflections, increasing the amount of data by 8×.
- **Elo rating system**: Tracks changes in model strength and automatically evaluates new versions against old ones.
- **Temperature annealing**: High temperatures encourage exploration early on, while low temperatures select the best moves later.
- **Full checkpoint resumption**: Automatically saves the model, optimizer, scheduler, training step, total number of games, and all other training state.

The AlphaZero version uses **SE-ResNet (Squeeze-Excitation Residual Network)**, adding a channel-attention mechanism to the standard residual block:

| Model | Channels | Layers | Parameters | Use Case |
|-------|----------|--------|------------|----------|
| GomokuNetAlphaZeroSmall | 64 | 6 | ~460K | Quick experiments and small boards |
| GomokuNetAlphaZero | 128 | 10 | ~3M | Formal training and large boards |

Unlike the traditional `CE × reward` loss, training uses the weighted sum of three losses:

```text
L = (z - v)² - π^T · log(p) + c · ||θ||²
    ─────   ─────────────   ───────
    value MSE  policy CE     L2 regularization
```

- `z`: The final game result (+1 win / -1 loss / 0 draw); `v`: The value predicted by the model.
- `π`: The policy distribution produced by MCTS; `p`: The policy probabilities predicted by the model.

Example training log:
```text
Game  42 | 127 moves | winner=Black | opponent=history | 896 samples | 4.2s
  MCTS: 2.1s | NN: 1.5s | symm: 0.4s | avg: 30 sims/s
```
- `opponent=history`: The opponent comes from the historical model pool; `opponent=self`: The opponent is the latest current model.
- `samples`: The number of training samples generated by this game (including symmetry augmentation).

### AI Reward Mechanism Explained

Basic Gomoku concepts:

Note: In the diagrams below, X represents the player's stone, O represents the opponent's stone, and . represents an empty intersection.

* Blocked Two (Two in a row with one end blocked)

  **Meaning**: Two consecutive stones with one end blocked. It has the lowest value, but provides a foundation for future development.
```text
. . . . .
O X X . .
. . . . .
```
* Live Two

  **Meaning**: Two consecutive stones with neither end blocked. This is the most basic attacking shape and receives a small reward.
```text
. . . . .
. . X X .
. . . . .
```

* Blocked Three (Three in a row with one end blocked)

  **Meaning**: Three consecutive stones with one end blocked. It requires two more moves to reach five but still has attacking value.
```text
. . . . . .
. O X X X .
. . . . . .
```

* Live Three

  **Meaning**: Three consecutive stones with neither end blocked. It can develop into a live four or blocked four and is an important latent threat.
```text
. . . . . . .
. . . X X X .
. . . . . . .
```

* Blocked Four (Four in a row with one end blocked)

  **Meaning**: Four consecutive stones with one end blocked. One additional stone can complete five, making this an important attacking shape.
```text
. . . . . . . . .
. . O X X X X . .
. . . . . . . . .
```

* Live Four

  **Meaning**: Four consecutive stones with neither end blocked. This is a forced win because the opponent cannot defend both winning ends at the same time.
```text
. . . . . . . . .
. . . X X X X . .
. . . . . . . . .
```

* Double Live Three

  **Meaning**: A single move creates two live threes at the same time. This shape usually creates a forced win because the opponent cannot defend both attacking directions simultaneously.
```text
. . . . . . .
. . . . . . .
. . X . X X .
. . . X . . .
. . . X . . .
. . . . . . .
```

* Blocked Four + Live Three

  **Meaning**: A single move creates both a blocked four and a live three. This is a very powerful combination in Gomoku and has a very high reward; it usually means the player can win on the next move.
```text
. . . . . . . .
. . . . . . . .
. . X . . . . .
. O X X X X . .
. . X . . . . .
. . . . . . . .
```

* Double Blocked Four

  **Meaning**: A single move creates two blocked fours. This is a forced winning move with a very high reward and means the player can win on the next move.
```text
. . . . . . . .
. . . . . X . .
. . . . X . . .
. O X X X X . .
. . X . . . . .
. O . . . . . .
. . . . . . . .
```

For more fundamental knowledge, see: [Gomoku terminology](https://baike.baidu.com/item/%E4%BA%94%E5%AD%90%E6%A3%8B%E6%9C%AF%E8%AF%AD/11009079) (Chinese).

### Model Validation (`val_az.py`)

Validate a trained AlphaZero model with `val_az.py`. The script prints two core result lines: win rate and draw rate only for the **validated model**, grouped by whether it moves first or second. In self-play mode, the models are displayed as "Validated Model 1 / Validated Model 2".

```shell
# Model self-play (target=self by default; P1 and P2 use the same model)
python val_az.py --board_size 15 --model standard --model_path alpaz_standard_15x15_best.pth --target self

# Play against the Kali-Hac teacher (validated model moves first as Black=P1; teacher is White=P2)
python val_az.py --board_size 15 --model standard --model_path alpaz_standard_15x15_best.pth --target teacher
```

Common options:

- `--target self|teacher`: Self-play (default) or play against the teacher.
- `--model small|standard`: Model size, which must match the size used during training.
- `--model_path`: Supports pure weights (`*_model.pth`, `*_best.pth`, `*_distill.pth`) and full checkpoints (the script automatically reads `model_state_dict`).
- `--total_rounds`: Number of validation games (default: 200).
- `--epsilon`: Random exploration rate for model moves (default: 0, pure greedy play; recommended for evaluation).
- `--seed N`: Fix the random seed for fully reproducible results.
### Model Architecture and Inputs/Outputs

The recommended AI models in this project are the AlphaZero-style dual-head networks `GomokuNetAlphaZero` (standard) and `GomokuNetAlphaZeroSmall` (small model for quick experiments). **Both are residual models**, specifically **SE-ResNet (Squeeze-Excitation Residual Network)** models: each residual block consists of two 3×3 convolutions, batch normalization, and SE channel attention, combined through an `out + residual` skip connection. Unlike the older `GomokuNetV3`, the new version **does not use a Transformer**. It is a purely convolutional architecture, and its fully convolutional design plus global average pooling give it native support for any board size.

| Model | Channels | SE Residual Blocks | Parameters | Use Case |
|-------|----------|--------------------|------------|----------|
| GomokuNetAlphaZeroSmall | 64 | 6 | ~460K | Quick experiments and small boards |
| GomokuNetAlphaZero | 128 | 10 | ~3M | Formal training and large boards |

Network structure:

```text
Input (B, 2, H, W)
  ↓
Stem: Conv3×3(2→C) → BN → ReLU
  ↓
Body: N × SEResBlock(Conv3×3 → BN → ReLU → Conv3×3 → BN → SE → residual add → ReLU)
  ↓
┌────────────────┬────────────────┐
│ Policy Head    │ Value Head     │
│ Conv1×1→32     │ Conv1×1→32     │
│ BN → ReLU      │ BN → ReLU      │
│ Conv1×1→1      │ AvgPool → FC   │
│ → (B, H×W)     │ → (B, 1)       │
└────────────────┴────────────────┘
```

The model's inputs and outputs are designed as follows:

#### 1. Model Input

- **Meaning**: The model input is a representation of the current Gomoku position. A multi-channel representation lets the model distinguish the two players' stones.
- **Shape**: `(batch_size, 2, board_size, board_size)`
    - **`batch_size`**: The number of game positions processed at once. It is usually greater than 1 during training and 1 during prediction.
    - **`2`**: The number of input channels. The representation uses the **current player's perspective**, making it view-symmetric so the same weights work for Black and White.
        - **Channel 0**: The current player's stones. The value is 1 where the current player has a stone and 0 otherwise.
        - **Channel 1**: The opponent's stones. The value is 1 where the opponent has a stone and 0 otherwise.
    - **`board_size`**: The side length of the board. The fully convolutional model has no fixed-size layers and supports any board size (for a 5x5 board, `board_size` is 5).

---

#### 2. Model Output - Policy Logits (`policy_logits`)

- **Meaning**: These are raw scores (logits) representing the "likelihood" of playing at each board position. A higher score means the model considers that move better.
- **Purpose**: Internally, these logits are usually converted into a probability distribution by **Softmax** to guide move selection. In practice, the valid move with the highest logit is usually selected.
- **Shape**: `(batch_size, board_size * board_size)`
    - **`batch_size`**: The same as the input.
    - **`board_size * board_size`**: A one-dimensional vector whose length equals the total number of board intersections. For a 5x5 board, the length is 25. Each element is the logit for one board position.
- **Value range**: These are raw scores with no fixed range; they can be any real number, positive or negative.

---

#### 3. Model Output - Value Output (`value_output`)

- **Meaning**: A scalar prediction of the current position's **win probability**.
- **Purpose**: The value head evaluates how favorable the current board position is. A value close to 1 means the model believes the current player has a very high chance of winning; a value close to -1 means the opponent has a very high chance of winning; a value close to 0 suggests a balanced position.
- **Shape**: `(batch_size)`
    - **`batch_size`**: The same as the input. Each game position has one value prediction.
    - **Perspective**: The value is evaluated from the **current player's** perspective (the output shape is `(batch_size,)`, and the trailing dimension is removed by `squeeze(-1)` in the code).
- **Value range**: A `Tanh` activation restricts the range to **$[-1, 1]$**.
    - **$1$**: The current player has definitely won.
    - **$-1$**: The current player has definitely lost.
    - **$0$**: The position is balanced.

With the policy and value heads, the model can perform both **decision-making (policy)** and **position evaluation (value)**, following the classic dual-head architecture used by deep reinforcement learning models such as AlphaGo and AlphaZero.

### Using the GPU

The code automatically detects and uses an available GPU through PyTorch; no manual configuration is required:

* If the system has a compatible NVIDIA GPU and the corresponding CUDA version is installed, GPU acceleration is enabled automatically.
* AMD GPUs can also use the ROCm build of PyTorch without changing a single line of code.
* If no GPU is available, the code automatically falls back to CPU mode.

### Converting to ONNX

You can convert a trained model to ONNX format and a TorchScript model. You must specify the board size and win condition.

#### Fixed-Shape Model (Legacy Version, Not Recommended)

<details>
  <summary>Click to expand</summary>

```shell
# Basic usage (8x8 board, five-in-a-row win condition)
python export_onnx.py gobang_best_model.pth --board_size 8 --win_condition 5

# Custom output path
python export_onnx.py gobang_best_model.pth --board_size 8 --win_condition 5 --onnx_path ./webdemo/model_bs8_win5.onnx
```

</details>

#### Dynamic-Shape Model (Legacy Version, Not Recommended)

<details>
  <summary>Click to expand</summary>

```shell
# Basic usage (15x15 board, five-in-a-row win condition)
python export_onnx_dy.py gobang_best_model_dy.pth --board_size 15 --win_condition 5

# Custom output path
python export_onnx_dy.py gobang_best_model_dy.pth --board_size 15 --onnx_path ./webdemo/model_bs15_win5.onnx
```

</details>

#### AlphaZero Model Export (Recommended)

```shell
# Export the Standard model
python export_onnx_az.py alpaz_standard_15x15_model.pth --board_size 15 --model standard

# Export the Small model
python export_onnx_az.py alpaz_small_15x15_model.pth --board_size 15 --model small

# Custom output path (for the Web Demo)
python export_onnx_az.py alpaz_standard_15x15_model.pth --board_size 15 --model standard --onnx_path ./webdemo/model_bs15_win5.onnx
```

After a successful ONNX export, `gobang_az_*_*x*.onnx` and `gobang_az_*_*x*.pt` files are generated in the directory. You can then use the human-vs-AI program under `webdemo/` to test the model.

## Projects using Gomoku-AI

* [AlphaZero-based TW game plugin](https://www.bilibili.com/video/BV1V5cozPELG/)
* [Gomoku move suggestion website](https://github.com/tkgg18201760958/WuZi)
* [Gomoku 5×5 must5src](https://github.com/732857315/must5src)

## Contributing

Contributions are welcome! If you have any improvement suggestions or find an issue, please submit a Pull Request or create an issue directly in this repository.
