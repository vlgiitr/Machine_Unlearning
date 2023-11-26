# Unlearning_Challenge
Approach to Machine Unlearning on Cifar-10 using Resnet and Vit 

## Contents

1. [Overview](#1-overview)
2. 

## 1. Overview

 **Machine unlearning** is an emergent subfield of machine learning that aims to remove the influence of a specific subset of training examples — the "forget set" — from a trained model. Furthermore, an ideal unlearning algorithm would remove the influence of certain examples while maintaining other beneficial properties, such as the accuracy on the rest of the train set and generalization to held-out examples.

A straightforward way to produce this unlearned model is to retrain the model on an adjusted training set that excludes the samples from the forget set.

## 2. Approach

we have gone beyond traditional fine-tuning methods by exploring the impact of sparsity enforcement on the unlearning process. Our experimentation involves the utilization of  several pruning techniques, followed by unlearning with a choice of five advanced algorithms. Additionally, we have introduced support for LoRA, which significantly enhances the retraining speed.

## 3.
