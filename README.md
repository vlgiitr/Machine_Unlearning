# Unlearning_Challenge
Approach to Machine Unlearning on Cifar-10 using Resnet and Vit 

## Contents

1. [Overview](#1-overview)
2. [Approach](#2-Approach)
3. [Pruning](#3-Pruning)
4. [Unlearning](#4-Unlearning)
5. [Conclusion](#5-Conclusion)

## 1. Overview

 **Machine unlearning** is an emergent subfield of machine learning that aims to remove the influence of a specific subset of training examples — the "forget set" — from a trained model. Furthermore, an ideal unlearning algorithm would remove the influence of certain examples while maintaining other beneficial properties, such as the accuracy on the rest of the train set and generalization to held-out examples.

A straightforward way to produce this unlearned model is to retrain the model on an adjusted training set that excludes the samples from the forget set.

## 2. Approach

We have gone beyond traditional fine-tuning methods by exploring the impact of sparsity enforcement on the unlearning process. Our experimentation involves the utilization of  several pruning techniques, followed by unlearning with a choice of five advanced algorithms. Additionally, we have introduced support for LoRA, which significantly enhances the retraining speed. 

## 3. Pruning
One of the following commands may be run to train the chosen pre-trained model followed by pruning.

**OMP**

python -u main_imp.py --data ./data --dataset ${data} --arch ${arch} --prune_type rewind_lt --rewind_epoch 8 --save_dir ${save_dir} --rate ${rate} --pruning_times 2 --num_workers 8

**IMP**

python -u main_imp.py --data ./data --dataset ${data} --arch ${arch} --prune_type rewind_lt --rewind_epoch 8 --save_dir ${save_dir} --rate 0.2 --pruning_times ${pruning_times} --num_workers 8

## 4. Unlearning

Following pruning one of these may be run to carry out unlearning.

**FT**

python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10

**GA**

python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5

**FF**

python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn fisher_new --num_indexes_to_replace 4500 --alpha ${alpha}

**IU**

python -u main_forget.py --save_dir ${save_dir} --mask ${mask_path} --unlearn wfisher --num_indexes_to_replace 4500 --alpha ${alpha}

## 5. Conclusion

 We discovered that the introduction of sparsity can effectively enhance the accuracy of the unlearning algorithm. Moreover, the application of fine-tuning, bolstered by LoRA, not only substantiates this concept but also expedites the entire process.








