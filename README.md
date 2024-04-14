# Group 58 Reproducibility project: Data-driven feature tracking for event cameras

This repository is an adapted version of the original repository along with the changes required during the project.

# Reproducing our results

## Setting up dependencies 
Install the dependencies `pip install -r requirements.txt`. This wont work for torch, you need to supply the source for the dependency so 

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Adding large files
All large files used in this project are in this google drive folder: INSERT LINK. Simply download this file and extract its contents in the root directory (i.e. just the loose files and directories, they will match the structure needed)

## Generating the inferences
1. Update all instances of <path_to_repo> in `configs.eval_real_defaults.yaml`
2. Run `python evaluate_real.py`
3. Results of inference should be in `correlation3_unscaled/timestamp/`

N.B. We provide our results so this step is not necessary.

## Benchmarking the results 
1. Move the results into `gt/network_pred/` (We provide our own results in the drive)
2. Run `python -m scripts.benchmark`
3. Results will be written to `out/benchmarking_results.csv`