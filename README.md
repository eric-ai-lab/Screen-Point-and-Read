# Screen-Point-and-Read (ScreenPR)
**Read Anywhere Pointed: Layout-aware GUI Screen Reading with Tree-of-Lens Grounding**

[![Paper](https://img.shields.io/badge/Arxiv%20-Visit-red)](http://arxiv.org/abs/2406.19263)
[![Project Webpage](https://img.shields.io/badge/Project%20Webpage-Visit-blue)](screen-point-and-read.github.io)
[![Hugging Face Data Page](https://img.shields.io/badge/Hugging%20Face%20Data%20Page-Visit-orange)](https://huggingface.co/datasets/yfan1997/ScreenPR)

## Abstract
Graphical User Interfaces (GUIs) are central to our interaction with digital devices. Recently, growing efforts have been made to build models for various GUI understanding tasks. However, these efforts largely overlook an important GUI-referring task: screen reading based on user-indicated points, which we name the Screen Point-and-Read (ScreenPR) task. This task is predominantly handled by rigid accessible screen reading tools, in great need of new models driven by advancements in Multimodal Large Language Models (MLLMs). In this paper, we propose a Tree-of-Lens (ToL) agent, utilizing a novel ToL grounding mechanism, to address the ScreenPR task. Based on the input point coordinate and the corresponding GUI screenshot, our ToL agent constructs a Hierarchical Layout Tree. Based on the tree, our ToL agent not only comprehends the content of the indicated area but also articulates the layout and spatial relationships between elements. Such layout information is crucial for accurately interpreting information on the screen, distinguishing our ToL agent from other screen reading tools. We also thoroughly evaluate the ToL agent against other baselines on a newly proposed ScreenPR benchmark, which includes GUIs from mobile, web, and operating systems. Last but not least, we test the ToL agent on mobile GUI navigation tasks, demonstrating its utility in identifying incorrect actions along the path of agent execution trajectories. 


## Table of Contents
- [Hierarchical Layout Tree Construction](#hierarchical-layout-tree-construction)
- [Target Path Selection & Prompting with Multi-lens](#target-path-selection--prompting-with-multi-lens)
- [Cycle Consistency Evaluation](#evaluation)

(under construction)
<!-- 
## Hierarchical Layout Tree Construction

We train a GUI region detection model to detect the local and global regions for each GUI screenshot, then construct the Hierarchical Layout Trees accordingly. The GUI region detection model is fine-tuned on the DINO detection model with [MMDetection](https://github.com/open-mmlab/mmdetection). We detail the training, evaluation and inference code in a seperate repo [here]().


## Target Path Selection & Prompting with Multi-lens

Based on the output from the previous Hierarchical Layout Tree construction process, we select the target path in the tree based on the input point coordinate and then generate lenses as prompts.
 -->