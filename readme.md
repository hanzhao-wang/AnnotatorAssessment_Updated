## Setup

The code is tested on Python 3.10 and cuda 12.1. Please make sure you have installed cuda >= 11.6 and satisfy the minimal requirement for flash attention.
You can use either
```bash
conda env create -f py310_env.yaml
```
to directly create the conda environment or manually install the required packages by running
```bash
conda create -n rm_dev python=3.10.15
conda activate rm_dev    

pip3 install torch==2.1.2 torchvision torchaudio 
pip3 install numpy==1.26.4
pip3 install flash-attn==2.6.3
pip3 install accelerate==0.33.0 
pip3 install deepspeed==0.12.2
pip3 install transformers==4.43.4
pip3 install wandb peft click datasets sentencepiece bitsandbytes rewardbench loguru
pip3 install "fschat[model_worker,webui]"
pip3 install "huggingface_hub[cli]"
```

Then please login the wandb account by running `wandb login` and huggingface account by running `huggingface-cli login`.

## Usage

We use json configs to manage the experiment settings. You can find all the experiment configs in `paper_experiment_configs/`. To reproduce, first prepare the oracle-annotated dataset by running 
```bash
python prepare_oracle_data.py
```
It would download and annotate the dataset.

To run the experiments, 
```bash
python Section3.py
python Section4.py
```

## Reviewer Sensitivity Experiments (Figure 3)

`Section4.py` now supports assumption-compliant sensitivity analysis over disagreement probability (`delta`) and utility-function profiles:

- `G_a` is increasing and strictly concave.
- `E` is increasing and convex.
- `mu` is increasing and concave.

Default reviewer run:
```bash
python Section4.py
```

Quick smoke run:
```bash
python Section4.py --datasets PKU --n-list 1,11 --delta-values 0,0.02,0.1 --utility-profiles baseline --save-tag smoke
```

CLI options:

- `--delta-values`: comma-separated deltas (default: `0.0,0.01,0.02,0.05,0.1,0.2`)
- `--utility-profiles`: comma-separated profile names (default: `baseline,risk_averse_poly,high_cost_exp`)
- `--datasets`: comma-separated datasets (default: `PKU,sky,Helpsteer,Ultra`)
- `--n-list`: comma-separated `n` values; if omitted, dataset defaults are used
- `--save-tag`: optional output suffix

Reviewer summary artifacts:

- `fig_contract/sensitivity/figure3_delta_utility_summary.csv`
- `fig_contract/sensitivity/figure3_delta_utility_summary.eps`

## Acknowledgements
This codebase is built on top of [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm). Special thanks to its creators for their valuable contributions and insights.
