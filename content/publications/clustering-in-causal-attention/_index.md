---
title: Clustering in Causal Attention Masking
date: 2024-12-09
publication_types: 1
publication: NeurIPS 2024
abstract: This work presents a modification of the self-attention dynamics proposed by [Geshkovski et al] to better reflect the practically relevant, causally masked attention used in transformer architectures for generative AI. This modification translates into an interacting particle system that cannot be interpreted as a mean-field gradient flow. Despite this loss of structure, we significantly strengthen the results of [Geshkovski et al] in the following context. While previous rigorous results focused on cases where all three matrices (Key, Query, and Value) were scaled identities, we prove asymptotic convergence to a single cluster for arbitrary key-query matrices and a value matrix equal to the identity.
Additionally, we establish a connection to the classical R\'enyi parking problem from combinatorial geometry to make initial theoretical steps towards demonstrating the existence of meta-stable states.
---
