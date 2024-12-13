---
title: NeurIPS 2024
summary: General overview of the paper [Clustering in Causal Attention Masking](https://arxiv.org/abs/2411.04990)
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
fundamental introduction into this model. After several assumptions we arrive to
$$
x_k = P_{x_k}(\frac{1}{Z_k} \sum_{j=1}^k e^{\beta \langle W_Q x_k, W_K x_j, \rangle} W_V x_j).
$$



{{< toc mobile_only=true is_open=true >}}


