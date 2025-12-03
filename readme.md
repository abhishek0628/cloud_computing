# Quick Start: VNF Placement with Transformer + ACO

This guide helps you quickly run training, evaluation, and generate plots for RL vs Random VNF placement.

---

## Requirements

- Python 3.10 or higher
- PyTorch
- Transformers
- python-dateutil

## Clone the project
```bash
mkdir project && cd project
git clone https://github.com/Abhishek0628/cloud_computing.git
cd code
```


```bash
<touch src/__init__.py
touch src/env/__init__.py
touch src/models/__init__.py
touch src/train/__init__.py
touch src/eval/__init__.py
``` -->
```


##  Install dependencies:

```bash
pip install torch numpy networkx matplotlib tqdm
```
```bash
export PYTHONPATH=$(pwd)/src    # macOS / Linux
# For Windows cmd: set PYTHONPATH=%cd%\src
# For Windows PowerShell: $env:PYTHONPATH="$(pwd)/src"
```

---

## Training (REINFORCE)

Run the training script:

```bash
python src/train/train_encoder_decoder.py
```

Default configuration:
- `num_epochs = 20` (adjustable)
- `sfc_per_epoch = 10`
- `samples_per_sfc = 2`
- `topology_nodes = 20`

The trained model is saved as:

```
experiments/encoder_decoder.pth
```

💡 Tip: Adjust `num_epochs` based on runtime and desired performance.

---

##  Evaluation & Plots

Run evaluation to compute RL vs Random placement and generate plots:

```bash
python src/eval/evaluate_encoder_decoder.py
python src/eval/evaluate_with_plots.py
```


Generates the following plots in `experiments/`:

| File                  | Description |
|-----------------------|-------------|
| `rl_vs_random.png`    | Boxplot comparing RL vs Random resource variance |
| `network_path.png`    | Network topology with chosen RL (green) and Random (red dashed) paths |

---


##  Optional: Reproducibility

Fix random seeds to ensure the same topology and SFCs:

```python
import random
import torch
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

---

##  Changing Hyperparameters

Modify these directly in the scripts:

- `num_epochs` → Number of training epochs  
- `sfc_per_epoch` → Number of SFCs per epoch  
- `samples_per_sfc` → REINFORCE samples per SFC  
- `topology_nodes` → Number of network nodes  
- `lr` → Learning rate  

---

##  Notes

- Network topology is randomly generated; paths may change each run unless random seed is fixed.  
- Plots allow visual comparison of RL policy performance vs random baseline.

---

## Author

Abhishek Kumar

