import os
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict

import utils
import unlearn
import pruner
from trainer import validate
import evaluation

import arg_parser

from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTForImageClassification, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from typing import List
import torch
import torch.nn as nn
from models.ResNet import *

# class ResnetConfig(PretrainedConfig):
#     model_type = "resnet"

#     def __init__(
#         self,
#         num_classes=1000,
#         zero_init_residual=False,
#         groups=1,
#         width_per_group=64,
#         replace_stride_with_dilation=None,
#         norm_layer=None,
#         imagenet=False,
#         **kwargs,
#     ):
#         if norm_layer is None:
#             self.norm_layer = nn.BatchNorm2d

#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             self.replace_stride_with_dilation = [False, False, False]



#         self.layers = [2, 2, 2, 2]
#         self.num_classes = num_classes
#         self.groups = groups
#         self.width_per_group = width_per_group
#         super().__init__(**kwargs)


# class ResnetModelForImageClassification(PreTrainedModel):
#     config_class = ResnetConfig

#     def __init__(self, config, pruned_model):
#         super().__init__(config)
#         self.model = pruned_model

#     def forward(self, tensor):
#         return self.model.forward(tensor)

def load_prunned_model(model, checkpoint):
    model_state_dict = model.state_dict()
    for key in checkpoint['state_dict'].keys():
        if "mask" in key or 'orig' in key:
            raw_key = key.split('_')[0]
            orig_w_key = raw_key + '_orig'
            mask_w_key = raw_key + '_mask'

            # Check if orig and mask keys exist in the checkpoint
            if orig_w_key not in checkpoint['state_dict'] or mask_w_key not in checkpoint['state_dict']:
                raise KeyError(f"Missing orig/mask keys for {raw_key}")

            # Extract original weight (A) and mask (B)
            A = checkpoint['state_dict'][orig_w_key]
            B = checkpoint['state_dict'][mask_w_key]

            # Check if A and B have compatible shapes
            if A.shape != B.shape:
                raise ValueError(f"Shapes of {orig_w_key} and {mask_w_key} do not match")

            # Perform pointwise multiplication and assign to the original key in the model's state_dict
            model_state_dict[raw_key] = A.mul(B)

        else:
            # Assign the same weight in the checkpoint to the model
            model_state_dict[key] = checkpoint['state_dict'][key]
            
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def add_lora(model,target_modules,r=8,lora_alpha=16,lora_dropout=0.1):  
    
    config = LoraConfig(
                        r=r,
                        lora_alpha=lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        modules_to_save=["classifier"],
                        )
    lora_model = get_peft_model(model, config)
    
    return lora_model


    
def main():
    args = arg_parser.parse_args()
    
    print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(
        args)
    model.cuda()

    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle)

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = - forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = - forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert(len(forget_dataset) + len(retain_dataset)
               == len(train_loader_full.dataset))
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = - forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True)
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True)
            print(len(retain_dataset))
            assert(len(forget_dataset) + len(retain_dataset)
                == len(train_loader_full.dataset))
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = - forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True)
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True)
            print(len(retain_dataset))
            assert(len(forget_dataset) + len(retain_dataset)
                == len(train_loader_full.dataset))

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        val=val_loader,
        test=test_loader)
        

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:    
        if args.hf_vit=="YES":
            print("Loading_vit_prunned_model")
            id2label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
            label2id = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
            model = ViTForImageClassification.from_pretrained('02shanky/vit-finetuned-cifar10',
                                                            id2label=id2label,
                                                            label2id=label2id)
            model.to(device)
            checkpoint = torch.load(args.mask, map_location=device)
            if args.lora=='YES':
                print("VIT_LoRA method")
                target_modules=["query", "value", "dense"]
                model = load_prunned_model(model, checkpoint)
                
        elif args.arch=="resnet18" and args.lora=='YES':
            print("RESNET_LoRA_method")
            target_modules=['conv1','conv2','fc']
            checkpoint = torch.load(args.mask, map_location=device)
            model = load_prunned_model(model, checkpoint)
        else:
            checkpoint = torch.load(args.mask, map_location=device)
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']
            current_mask = pruner.extract_mask(checkpoint)
            pruner.prune_model_custom(model, current_mask, args)
            pruner.check_sparsity(model,args)

            if args.unlearn != "retrain" and args.unlearn != "retrain_sam" and args.unlearn != "retrain_ls":
                model.load_state_dict(checkpoint, strict=False)
            
            pruner.check_sparsity(model, args)

        
        if args.lora=='YES':
            model = add_lora(model,target_modules,r=64,lora_alpha=16,lora_dropout=0.1)
            print_trainable_parameters(model)
        
        # print([name for name, m in model.named_modules()])
        # print([name for name, m in model.named_parameters()])
        print(model)
        
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        # print("unlearn_method: ",unlearn_method)

        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if 'new_accuracy' not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print("**********",name,"**********")
            utils.dataset_convert_to_test(loader.dataset,args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result['accuracy'] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ['MIA', 'SVC_MIA', 'SVC_MIA_forget']:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    '''forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)'''
    if 'SVC_MIA_forget_efficacy' not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset,args)
        utils.dataset_convert_to_test(forget_loader,args)
        utils.dataset_convert_to_test(test_loader,args)

        shadow_train = torch.utils.data.Subset(
            retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False)

        evaluation_result['SVC_MIA_forget_efficacy'] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader, shadow_test=test_loader,
            target_train=None, target_test=forget_loader,
            model=model, args=args)
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    '''training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)'''
    if 'SVC_MIA_training_privacy' not in evaluation_result:
        test_len = len(test_loader.dataset)
        retain_len = len(retain_dataset)
        num = test_len // 2

        utils.dataset_convert_to_test(retain_dataset,args)
        utils.dataset_convert_to_test(forget_loader,args)
        utils.dataset_convert_to_test(test_loader,args)

        shadow_train = torch.utils.data.Subset(
            retain_dataset, list(range(num)))
        target_train = torch.utils.data.Subset(
            retain_dataset, list(range(num, retain_len)))
        shadow_test = torch.utils.data.Subset(
            test_loader.dataset, list(range(num)))
        target_test = torch.utils.data.Subset(
            test_loader.dataset, list(range(num, test_len)))

        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=args.batch_size, shuffle=False)

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=args.batch_size, shuffle=False)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=args.batch_size, shuffle=False)

        evaluation_result['SVC_MIA_training_privacy'] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader, shadow_test=shadow_test_loader,
            target_train=target_train_loader, target_test=target_test_loader,
            model=model, args=args)
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == '__main__':
    main()
