---
title: NeurIPS 2024
summary: General overview of the poster Clustering in Causal Attention Masking
date: 2024-12-13

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'My poster'

authors:
  - admin

tags:
  - Academic
  - Markdown
---

## Welcome 👋

I am happy to present my first graduate paper [Clustering in Causal Attention Masking](https://arxiv.org/abs/2411.04990) as a [poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95352.png?t=1734050775.9867222) at NeurIPS 2024. The main goal of this work is to better understand self-attention from theoretical perspective, by modeling propagation of tokens through layers as an interacting particle system.

## The main system
Similar to how ResNets are studied via neural ODEs, one can write down the dynamical system that models propagation of tokens 
through layers of a pre-trained transformer, see [Geshkovski et al](https://arxiv.org/abs/2312.10794) for a 
fundamental introduction into this model. After several assumptions we arrive to the main equation. Tokens, represented as vectors $x_k, k \in [n]$ on a $d$-dimensional unit sphere (this assumption comes from normalisation layer), evolve according to a system of non-linear differential equations
$$
x_k = P_{x_k}(\frac{1}{Z_k} \sum_{j=1}^k e^{\beta \langle W_Q x_k, W_K x_j, \rangle} W_V x_j),
$$
where $W_Q, W_K, W_V$ are standard query, key and value matrices in transformers architecture, $\beta$ is a hyperparameter, $Z_k$ is sum of attention weights (so that they sum up to one) and $P_{x}(y) = y - \langle y, x\rangle x$ ensures that tokens stay on a unit sphere. 

## Convergence to one cluster
This system, in a sense, is a generalisation of the famous Kuramoto model. In that sense, the following result is a continuation of a long line of works dedicated to synchronisation phenomena. For us, it can be seen as a negative property, because it tells us that deep transformers tend to collapse.

We prove that for $$W_V = I_d$$ almost any initial configuration of tokens converges to one point.
Moreover, we conjecture that for large timeframes, the largest real part of an eigenvalue of $W_V$ that we denote $\lambda_{\max}$ determines the complexity of the dynamics.


{{< toc mobile_only=true is_open=true >}}


