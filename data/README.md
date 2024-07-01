# Datasets preparation

## Clone from huggingface

Three required datasets should be cloned from three huggingface datasets.

* [Android Screen Hierarchical Layout dataset](https://huggingface.co/datasets/orlando23/screendata) (ASHL)
* [Screen Point-and-Read Benchmark](https://huggingface.co/datasets/orlando23/mobile_pc_web_osworld)
* [Mobile Trajectory Verification Data](https://huggingface.co/datasets/orlando23/failed_agent_trajectory)

Using the following commands to download them to this folder

```bash
git clone https://huggingface.co/datasets/orlando23/screendata
git clone https://huggingface.co/datasets/orlando23/mobile_pc_web_osworld
git clone https://huggingface.co/datasets/orlando23/failed_agent_trajectory
```

Double-check the data tracked by lfs downloaded after git clone. Otherwise, use `git lfs pull` to guarantee they have been synced.

## Training and Validation split for ASHL

We provide our used Training and Validation split for ASHL in [ScreenReaderData_train.json](ScreenReaderData_train.json) and [ScreenReaderData_val.json](ScreenReaderData_val.json). You can take it as your default setting in DINO training to reproduce our result.
