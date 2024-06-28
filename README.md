
**Read Anywhere Pointed: Layout-aware GUI Screen Reading with Tree-of-Lens Grounding**

[![Paper](https://img.shields.io/badge/Arxiv%20-Visit-red)](http://arxiv.org/abs/2406.19263)
[![Project Webpage](https://img.shields.io/badge/Project%20Webpage-Visit-blue)](screen-point-and-read.github.io)
[![Hugging Face Data Page](https://img.shields.io/badge/Hugging%20Face%20Data%20Page-Visit-orange)](https://huggingface.co/datasets/yfan1997/ScreenPR)

## Abstract

Graphical User Interfaces (GUIs) are central to our interaction with digital devices. Recently, growing efforts have been made to build models for various GUI understanding tasks. However, these efforts largely overlook an important GUI-referring task: screen reading based on user-indicated points, which we name the Screen Point-and-Read (ScreenPR) task. This task is predominantly handled by rigid accessible screen reading tools, in great need of new models driven by advancements in Multimodal Large Language Models (MLLMs). In this paper, we propose a Tree-of-Lens (ToL) agent, utilizing a novel ToL grounding mechanism, to address the ScreenPR task. Based on the input point coordinate and the corresponding GUI screenshot, our ToL agent constructs a Hierarchical Layout Tree. Based on the tree, our ToL agent not only comprehends the content of the indicated area but also articulates the layout and spatial relationships between elements. Such layout information is crucial for accurately interpreting information on the screen, distinguishing our ToL agent from other screen reading tools. We also thoroughly evaluate the ToL agent against other baselines on a newly proposed ScreenPR benchmark, which includes GUIs from mobile, web, and operating systems. Last but not least, we test the ToL agent on mobile GUI navigation tasks, demonstrating its utility in identifying incorrect actions along the path of agent execution trajectories. 

## Todos

- [ ] Add script inputs details
- [ ] Add detail setup instruction
- [ ] Add Evaluation scripts


## ToL agent
+ ### GUI region detection 
    + We train a GUI region detection model to detect the local and global regions for each GUI screenshot. The GUI region detection model is fine-tuned on the DINO detection model with [MMDetection](https://github.com/open-mmlab/mmdetection), which is [a git submodule]((https://github.com/llv22/tol_gui_region_detection/)) for our main project. You need to use the following commands to finish the initialization
        ```bash
        git submodule init
        git submodule sync
        git submodule update --remote
        ```

    + The details about training and inference, please check [src/models/tol_gui_region_detection/README.md](https://github.com/llv22/tol_gui_region_detection/).

+ ### GUI Hierarchical Layout Tree construction, Target Path Selection & Prompting with Multi-lens
    + Based on the output (a json file) from the previous GUI region detection, the pipeline of our ToL agent includes:

        1. construct the Hierarchical Layout Trees based on the detected regions. 

        2. select the target path in the tree based on the input point coordinate. 

        3. generate lenses as prompts.

    + We use the following script for the listed process:
        ```
        python src/ToL_after_region_detection.py
        ```

## Cycle consistency evaluation
