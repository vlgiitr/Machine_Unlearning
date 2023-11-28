import numpy as np
import torch
import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

org_train_ds, org_test_ds = load_dataset('cifar10', split=['train', 'test'])
test_ds = org_test_ds.filter(lambda example: example['label']!=0)
forget_ds = org_train_ds.filter(lambda example: example['label']==0)
train_ds = org_train_ds.filter(lambda example: example['label']!=0)
split = train_ds.train_test_split(test_size=0.1)
retain_ds = split['train']
val_ds = split['test']

id2label = {id:label for id, label in enumerate(retain_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
print(id2label)

processor = ViTImageProcessor.from_pretrained('02shanky/vit-finetuned-cifar10')
model_vit = ViTForImageClassification.from_pretrained('02shanky/vit-finetuned-cifar10',
                                                  id2label=id2label,
                                                  label2id=label2id)


image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


# Set the transforms
retain_ds.set_transform(train_transforms)
forget_ds.set_transform(val_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
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

batch_size = 32
args = TrainingArguments(
    f"VIT-finetuned-lora-CIFAR10",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
    logging_dir='logs',
)



config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model_vit, config)
print_trainable_parameters(lora_model)

trainer = Trainer(
    lora_model,
    args,
    train_dataset=retain_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()
trainer.push_to_hub()
trainer.save_model('/content/drive/MyDrive/Colab_Notebooks/unlearning/lora_vit_finetuned_cifar10')


outputs_test = trainer.predict(test_ds)
print(outputs_test.metrics)

outputs_forget = trainer.predict(forget_ds)
print(outputs_forget.metrics)