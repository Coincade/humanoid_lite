# Humanoid Lite

A lightweight humanoid robot simulation and reinforcement learning framework for humanoid locomotion research. This repository provides tools for training and deploying humanoid locomotion policies using Isaac Sim and RSL-RL.

## Features

- **Humanoid Lite Robot**: A lightweight humanoid robot model with 22 degrees of freedom
- **Velocity Control**: Train policies for velocity-based locomotion tasks
- **Isaac Sim Integration**: Full integration with NVIDIA's Isaac Sim for high-fidelity simulation
- **RSL-RL Framework**: Reinforcement learning training using the RSL-RL framework
- **Policy Deployment**: Export trained policies for real-world deployment

## Prerequisites

- **Operating System**: Ubuntu 22.04 or later (recommended)
- **GPU**: NVIDIA GPU with CUDA support (required for Isaac Sim)
- **CUDA**: CUDA 12.8 or compatible version
- **Python**: Python 3.10
- **Conda**: Anaconda or Miniconda for environment management

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Coincade/humanoid_lite.git
cd humanoid_lite
```

### 2. Create Conda Environment

```bash
conda create -n humanoid-lite python=3.10 -y
conda activate humanoid-lite
```

### 3. Install System Dependencies

```bash
sudo apt update
sudo apt install cmake build-essential -y
```

### 4. Install Python Dependencies

Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install Isaac Sim:
```bash
pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab:
```bash
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```

### 5. Install Project Dependencies

Install the project packages in development mode:
```bash
# Install humanoid_lite package
cd source/humanoid_lite
pip install -e .

# Install humanoid_lite_assets package
cd ../humanoid_lite_assets
pip install -e .
```

## Usage

### Training

To train a humanoid locomotion policy, use the training script:

```bash
# Basic training command
python scripts/rsl_rl/train.py --task=Velocity-Humanoid-Lite-v0

# Training with custom parameters
python scripts/rsl_rl/train.py \
    --task=Velocity-Humanoid-Lite-v0 \
    --num_envs=4096 \
    --max_iterations=5000 \
    --seed=42

# Training with video recording
python scripts/rsl_rl/train.py \
    --task=Velocity-Humanoid-Lite-v0 \
    --video \
    --video_length=200 \
    --video_interval=2000
```

#### Training Parameters

- `--task`: Task name (default: Velocity-Humanoid-Lite-v0)
- `--num_envs`: Number of parallel environments (default: 4096)
- `--max_iterations`: Maximum training iterations (default: 5000)
- `--seed`: Random seed for reproducibility
- `--video`: Enable video recording during training
- `--video_length`: Length of recorded videos in steps
- `--video_interval`: Interval between video recordings

### Playing/Evaluation

To evaluate a trained policy:

```bash
# Basic play command (uses latest checkpoint)
python scripts/rsl_rl/play.py --task=Velocity-Humanoid-Lite-v0

# Play with specific checkpoint
python scripts/rsl_rl/play.py \
    --task=Velocity-Humanoid-Lite-v0 \
    --load_run=2025-07-31_09-13-39 \
    --load_checkpoint=model_5999.pt

# Play with video recording
python scripts/rsl_rl/play.py \
    --task=Velocity-Humanoid-Lite-v0 \
    --video \
    --video_length=500
```

#### Play Parameters

- `--task`: Task name (default: Velocity-Humanoid-Lite-v0)
- `--load_run`: Specific run directory to load from
- `--load_checkpoint`: Specific checkpoint file to load
- `--num_envs`: Number of environments for evaluation
- `--video`: Enable video recording during evaluation
- `--video_length`: Length of recorded videos in steps

### Available Tasks

Currently, the following tasks are available:

- **Velocity-Humanoid-Lite-v0**: Velocity control task for humanoid locomotion

## Project Structure

```
humanoid_lite/
├── configs/                 # Configuration files
├── logs/                    # Training logs and checkpoints
├── outputs/                 # Hydra outputs
├── scripts/                 # Training and evaluation scripts
│   └── rsl_rl/
│       ├── train.py        # Training script
│       └── play.py         # Evaluation script
├── source/                  # Source code
│   ├── humanoid_lite/      # Main package
│   └── humanoid_lite_assets/ # Robot assets
└── checkpoints/            # Pre-trained models
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/policy_humanoid.yaml`: Humanoid robot policy configuration
- `source/humanoid_lite/humanoid_lite/tasks/locomotion/velocity/config/humanoid/`: Task-specific configurations

## Training Logs

Training logs and checkpoints are saved in:
```
logs/rsl_rl/humanoid/{timestamp}/
```

Each training run creates a timestamped directory containing:
- Model checkpoints (`model_*.pt`)
- Training logs and metrics
- Exported policies (`exported/policy.pt`, `exported/policy.onnx`)
- Videos (if enabled)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `--num_envs` parameter
2. **Isaac Sim Launch Issues**: Ensure NVIDIA drivers are properly installed
3. **Import Errors**: Make sure all packages are installed in development mode

### Performance Tips

- Use a high-end GPU for faster training
- Adjust `num_envs` based on your hardware capabilities
- Use `--video` sparingly as it impacts performance

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request