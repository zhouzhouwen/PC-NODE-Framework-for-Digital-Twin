# PC-NODE Framework for Digital Twin

> “a practical high-fidelity digital-twin framework for aerosol control”

A research repository for a **physics-constrained AI digital twin** designed for **radioactive aerosol scavenging** in **nuclear decommissioning** scenarios.  
This project implements a **Transformer-based Physics-Constrained Neural Ordinary Differential Equation (PC-NODE)** to model aerosol removal dynamics under coupled spray–mist operating conditions, with a focus on **physical consistency**, **real-time inference**, **wastewater minimization**, and **transferability across materials**.

---

## Project Overview

This repository accompanies the study:

**A Physics-Constrained AI Digital Twin for Radioactive Aerosol Scavenging and Wastewater Minimization in Nuclear Decommissioning**

The work addresses a practical challenge in aerosol control: existing approaches often force a trade-off between **physical credibility** and **computational efficiency**. Empirical correlations may fail in strongly coupled multiphase regimes, high-fidelity CFD is too expensive for real-time use, and unconstrained AI models may violate basic physical behavior.

To address this, the repository implements a **PC-NODE digital twin** in which:

- a **Transformer-based rate network** learns the nonlinear dependence of aerosol decay on time and operating conditions,
- a **Neural ODE formulation** models continuous-time concentration evolution,
- **physics constraints** enforce monotonic aerosol decay by construction,
- the trained surrogate supports fast prediction and operating-condition exploration.

The framework is developed around the **UTARTS** aerosol scavenging system and is intended as a research-oriented surrogate modeling workflow for digital twin studies in multiphase process engineering.

---

## Key Features

- **Physics-constrained neural surrogate**
  - Enforces non-negative decay rates and monotonic concentration decrease.
- **Transformer-enhanced Neural ODE**
  - Captures nonlinear interactions between time and static operating variables.
- **Continuous-time prediction**
  - Predicts normalized aerosol concentration trajectories over the full scavenging process.
- **Wastewater minimization workflow**
  - Includes scripts for analyzing operating regimes that reduce water usage.
- **Transfer learning experiments**
  - Supports cross-material transfer from pretraining to downstream fine-tuning.
- **Paper-oriented evaluation**
  - Includes scripts for plotting and comparing results used in the manuscript.
- **Research reproducibility scaffold**
  - Organizes training, evaluation, optimization, and transfer-learning pipelines in one place.

---

## Tech Stack

This project is implemented in **Python** and currently relies on the scientific Python / deep learning ecosystem inferred from the source code.

Core libraries include:

- **PyTorch** — model definition and training
- **NumPy** — numerical operations
- **Pandas** — tabular data handling
- **scikit-learn** — preprocessing / scaling
- **joblib** — serialization of scalers
- **Optuna** — hyperparameter optimization
- **Matplotlib** — figure generation for paper results

> Note: The repository does not currently provide a pinned `requirements.txt` or `environment.yml`, so dependency versions should be added for stricter reproducibility.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhouzhouwen/PC-NODE-Framwork-for-Digital-Twin.git
cd PC-NODE-Framwork-for-Digital-Twin
```

### 2. Create a Python environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Or using Conda:

```bash
conda create -n pc-node python=3.10 -y
conda activate pc-node
```

### 3. Install dependencies

Since a dependency file is not yet included, install the main packages manually:

```bash
pip install torch numpy pandas scikit-learn joblib matplotlib openpyxl optuna
```

> You may need to adjust the PyTorch installation command depending on your CUDA / CPU environment.

---

## Quick Start

### 1. Prepare the dataset

The current training script expects an **Excel dataset** with UTARTS experimental data and time-series concentration columns.

At present, the training code uses a hard-coded Excel path. Before running, update the dataset path in the script:

```python
EXCEL_FILE_PATH = "/path/to/your/data.xlsx"
```

### 2. Train the PC-NODE model

```bash
python train_ode_Cm_Q.py
```

This script trains the monotonic physics-constrained model and saves artifacts such as:

- trained model weights
- feature scaler
- prediction outputs
- evaluation metrics

### 3. Evaluate the trained model

```bash
python eval_ode_Cm_Q.py
```

### 4. Reproduce paper-related analysis

Plotting / analysis scripts currently included:

```bash
python paper_3.2_plot.py
python "paper_3.3_dm_Cm_for_saving water.py"
python transfer_compare.py
```

---

## Configuration

At the current stage, configuration is primarily script-based rather than centralized.

### Main configurable items

In the training / evaluation scripts, check and update:

- **dataset path**
  - `EXCEL_FILE_PATH`
- **sheet name**
  - `SHEET_NAME`
- **output file paths**
  - model checkpoints
  - scaler files
  - CSV metric outputs
- **training hyperparameters**
  - batch size
  - learning rate
  - number of epochs
  - model dimensions
- **feature / time column mappings**
  - static operating variables
  - time grid columns

### Dataset structure

The current code expects:

- **9 static operating variables**, including spray, mist, and aerosol properties
- concentration measurements at multiple time points such as:
  - `0min`, `1min`, `3min`, `5min`, `10min`, `15min`, `20min`, `30min`, `40min`, `50min`, `60min`

### Recommendation

For future improvement, it is recommended to move these settings into:

- a `config.yaml` or `config.json`
- CLI arguments
- an environment-based experiment configuration system

---

## Project Structure

```text
PC-NODE-Framwork-for-Digital-Twin/
├── README.md
├── train_ode_Cm_Q.py
├── eval_ode_Cm_Q.py
├── paper_3.2_plot.py
├── paper_3.3_dm_Cm_for_saving water.py
├── transfer_compare.py
└── transfer_runs/
```

### File descriptions

- **train_ode_Cm_Q.py**  
  Main training script for the physics-constrained monotonic PC-NODE model.

- **eval_ode_Cm_Q.py**  
  Evaluation / inference script for trained models.

- **paper_3.2_plot.py**  
  Plotting script associated with paper result visualization.

- **paper_3.3_dm_Cm_for_saving water.py**  
  Analysis script for mist operating conditions and water-saving optimization.

- **transfer_compare.py**  
  Transfer learning comparison workflow across particle/material datasets.

- **transfer_runs/**  
  Saved experiment outputs and transfer-learning run artifacts.

---

## Usage Example

A typical workflow is:

### Step 1 — Train on source-domain data

```bash
python train_ode_Cm_Q.py
```

### Step 2 — Evaluate model performance

```bash
python eval_ode_Cm_Q.py
```

### Step 3 — Generate figures / paper-style comparisons

```bash
python paper_3.2_plot.py
```

### Step 4 — Explore wastewater minimization settings

```bash
python "paper_3.3_dm_Cm_for_saving water.py"
```

### Step 5 — Run transfer-learning experiments

```bash
python transfer_compare.py
```

---

## Development Notes

This repository is currently organized as a **research codebase** rather than a fully packaged software library.

### Current characteristics

- script-driven workflow
- paper-oriented experiment organization
- hard-coded local paths in some files
- no centralized dependency management yet
- no packaged module structure yet

### Suggested future improvements

- add `requirements.txt` or `environment.yml`
- replace hard-coded paths with CLI arguments or config files
- add dataset format documentation
- provide pretrained checkpoints
- add a reproducible demo notebook
- standardize filenames and script naming
- package the model code into reusable modules
- add automated tests for data loading and inference

---

## FAQ

### What does PC-NODE stand for?

**Physics-Constrained Neural Ordinary Differential Equation**.

### What problem does this repository solve?

It models **radioactive aerosol scavenging dynamics** under spray–mist operating conditions for use in a **digital twin** workflow relevant to **nuclear decommissioning**.

### Why not just use CFD?

CFD can be highly accurate but is generally too computationally expensive for rapid digital twin deployment and operating-condition exploration.

### Why use a physics-constrained model?

A purely data-driven model may fit observations while violating physically expected behavior.  
This framework enforces physically meaningful decay behavior directly in the model structure.

### Is the dataset included in this repository?

At the current public repository state, the training dataset is **not bundled as a ready-to-use public data folder** in the root workflow. You may need to prepare the Excel dataset separately before training.

### Can I run this repository immediately after cloning?

Not fully. You will likely need to:

1. install dependencies manually,
2. prepare the dataset,
3. update local file paths in the scripts.

### Does this repository include a web demo?

The manuscript references a digital-twin demonstration endpoint, but long-term availability of a live demo may depend on deployment status.

### Is this repository production-ready?

No. It should currently be considered a **research / experimental codebase** intended for manuscript support, reproducibility, and further development.

---

## License

**No license file is currently included in the public repository.**

That means default copyright rules apply unless a license is explicitly added by the repository owner.

If you plan to make this project openly reusable, consider adding a standard license such as:

- MIT License
- Apache-2.0
- BSD-3-Clause

---

## Citation

If you use this repository in academic work, please cite the associated paper.

```bibtex
@article{zhou_pcnode_digital_twin,
  title   = {A Physics-Constrained AI Digital Twin for Radioactive Aerosol Scavenging and Wastewater Minimization in Nuclear Decommissioning},
  author  = {Zhou, Wen and Xu, Ruicong and Miwa, Shuichiro and Wang, Hanyu and Wang, Kai and Sharma, Avadhesh Kumar and Okamoto, Koji},
  journal = {To be updated},
  year    = {2026}
}
```

---

## Acknowledgment

This repository is developed for research on **physics-constrained AI surrogates**, **aerosol control**, and **digital twin modeling** for safety-critical multiphase engineering systems.
