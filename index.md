---
layout: page
title: This Web is under deving
subtitle: Learning Concept-Based Visual Causal Transition and Symbolic Reasoning for Visual Planning
---
<center style="font-weight: bold"> </center>

<div style="display: flex; justify-content: center; align-items: center;">
  <span class="link-block" style="text-align: center; display: block; margin: 0 10px;">
    <a href="" target="_blank" class="external-link button is-normal is-rounded is-dark">
      <span class="icon">
        <i class="fab fa-github"></i>
      </span>
      <span>Code (Coming Soon)</span>
    </a>
  </span>
  <span class="link-block" style="text-align: center; display: block; margin: 0 10px;">
    <a href="https://arxiv.org/abs/2310.03325" target="_blank" class="external-link button is-normal is-rounded is-dark">
      <span class="icon">
        <i class="far fa-file-alt"></i>
      </span>
      <span>Paper</span>
    </a>
  </span>
  <span class="link-block" style="text-align: center; display: block; margin: 0 10px;">
    <a href="" target="_blank" class="external-link button is-normal is-rounded is-dark">
      <span class="icon">
        <i class="fa fa-database"></i>
      </span>
      <span>Dataset (Comming Soon)</span>
    </a>
  </span>
</div>


<!-- <p align="center">
    <a href='https://github.com/jiemingcui/probio/', target="_blank">[Code]
    </a>
    <a href='https://arxiv.org/abs/<ARXIV PAPER ID>', target="_blank">[ArXiv]
    </a>
</p> -->
<!-- Github link -->

<br>

![](assets/img/1.png)

Visual planning simulates how humans make decisions to achieve desired goals in the form of searching for visual causal transitions between an initial visual state and a final visual goal state. It has become increasingly important in egocentric vision with its advantages in guiding agents to perform daily tasks in complex environments. In this paper, we propose an interpretable and generalizable visual planning framework consisting of **i)** a novel Substitution-based Concept Learner (**SCL**) that abstracts visual inputs into disentangled concept representations, **ii)** symbol abstraction and reasoning that performs task planning via the self-learned symbols, and **iii)** a Visual Causal Transition model (**ViCT**) that grounds visual causal transitions to semantically similar real-world actions. Given an initial state, we perform goal-conditioned visual planning with a symbolic reasoning method fueled by the learned representations and causal transitions to reach the goal state. To verify the effectiveness of the proposed model, we collect a large-scale visual planning dataset based on AI2-THOR, dubbed as *CCTP*. Extensive experiments on this challenging dataset demonstrate the superior performance of our method in visual task planning. Empirically, we show that our framework can generalize to unseen task trajectories and unseen object categories.


<hr>

## Video

<div class="extensions extensions--video">
<iframe width="920" height="580" src="" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" frameborder="0" scrolling="no" allowfullscreen></iframe>
</div>

<hr>

## Framework

### Substitution-based Concept Learner (SCL)

<div class="card bg-light border-light mb-3">
    <img class="card-img lazyload" data-src="assets/img/concept_learning.png" />
    <div class="card-body">
      <h5 class="card-title">Figure 1. Architecture of SCL.</h5>
    </div>
</div>



### Symbol Abstraction and Reasoning

<div class="card bg-light border-light mb-3">
    <img class="card-img lazyload" data-src="assets/img/symbol_reasoning.png" />
    <div class="card-body">
      <h5 class="card-title">Figure 2. Symbol Abstraction and Reasoning.</h5>
    </div>
</div>

### Visual Causal Transition Learning (ViCT)

<div class="card bg-light border-light mb-3">
    <img class="card-img lazyload" data-src="assets/img/causal_transition.png" />
    <div class="card-body">
      <h5 class="card-title">Figure 3. Architecture of ViCT.</h5>
    </div>
</div>

## Dataset

To facilitate the learning and evaluation of the concept-
based visual planning task, we collect a large-scale RGB-D
image sequence dataset named CCTP (Concept-based Causal
Transition Planning) based on AI2-THOR simulator.
We exclude scene transitions in each task by design to
focus more on concept and causal transition learning, i.e.,
each task is performed on a fixed workbench, although the
workbenches and scenes vary from task to task. The whole dataset consists
of a concept learning dataset and a visual causal planning
dataset.


<!-- ### Visualization of the ambiguous actions in BioLab. -->



<hr>

### Concept Revision Demo

<div class="card bg-light border-light mb-3">
    <img class="card-img lazyload" data-src="assets/img/fig_4_updated.png" />
    <div class="card-body">
      <h5 class="card-title">Figure 4. Fine-grained attribute level concept manipulation.</h5>
    </div>
</div>

## Download

Our dataset is distributed under the [CC BY-NC-SA (Attribution-NonCommercial-ShareAlike)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You can download our dataset from [Link here (Coming Soon)]().

<hr>

## Citation

```bibtex

```
