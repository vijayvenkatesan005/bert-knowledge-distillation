#!/usr/bin/env python
# coding: utf-8

# In[65]:

import argparse

parser = argparse.ArgumentParser(
    prog="QQP Trainer Script",
    description="Trains BERT on the QQP dataset",
    epilog="Use at your own risk"
)

parser.add_argument("-lr", "--learning_rate", help="Controls the step size", type=float)
parser.add_argument("-bs", "--batch_size", help="Controls the number of examples processed at once", type=int)
parser.add_argument("-a", "--alpha", help="Controls the importance of cross entropy loss versus distillation loss", type=float)
parser.add_argument("-t", "--temperature", help="Controls how much softening is applied to the output probability distribution of the teacher model", type=int)

args = parser.parse_args()

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from transformers import AutoConfig
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# In[50]:


output_dir = './bert_fine_tuned_on-qqp'
num_train_epochs = 3
learning_rate = float(args.learning_rate) if args.learning_rate else 2e-05
batch_size = int(args.batch_size) if args.batch_size else 16
alpha = float(args.alpha) if args.alpha else 0.25
temperature = int(args.temperature) if args.temperature else 2
weight_decay = 0.2
num_labels = 2

# In[37]:


print(f"learning rate: {learning_rate}")
print(f"batch size: {batch_size}")
print(f"alpha: {alpha}")
print(f"temperature: {temperature}")

# In[13]:


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "qqp")


# In[14]:


raw_datasets

# In[15]:


raw_datasets["train"][0]

# In[16]:


def tokenize_input(example):
  question1 = example["question1"]
  question2 = example["question2"]
  return tokenizer(question1, question2, padding="max_length", max_length=128, truncation=True)

# In[17]:


tokenized_datasets = raw_datasets.map(tokenize_input, batched=True)

# In[19]:


model_name = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model.config)

# In[23]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="macro")
    rec = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    

# In[26]:


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

# In[32]:


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=batch_size):
        outputs_stu = model(**inputs)

        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits

        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits

        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1)
        )

        loss = self.args.alpha * loss_ce + (1 - self.args.alpha) * loss_kd
        return (loss, outputs_stu) if return_outputs else loss 
        

# In[42]:


student_training_args = DistillationTrainingArguments(
    output_dir=output_dir,
    report_to=['tensorboard'],
    logging_dir=f"log-lr-{learning_rate}-bs-{batch_size}-a-{alpha}-t-{temperature}",
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    alpha=alpha,
    temperature=temperature,
    weight_decay=weight_decay
)

# In[45]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[46]:


student_ckpt = 'distilbert-base-uncased'

# In[54]:


student_config = (
    AutoConfig.from_pretrained(student_ckpt, num_labels=num_labels)
)

# In[55]:


student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

# In[56]:


def student_init():
    return (
        AutoModelForSequenceClassification
        .from_pretrained(student_ckpt, config=student_config).to(device)
    )

# In[57]:


teacher_ckpt = 'bert-base-uncased'

# In[58]:


def teacher_init():
    return (
        AutoModelForSequenceClassification
        .from_pretrained(teacher_ckpt, num_labels=num_labels)
        .to(device)
    )

# In[59]:


teacher_model = teacher_init()

# In[60]:


id2label = {0: 'not paraphrase', 1: 'paraphrase'}
label2id = {'not paraphrase': 0, 'paraphrase': 1}

# In[61]:


distilbert_trainer = DistillationTrainer(
    model_init=student_init,
    teacher_model=teacher_model,
    args=student_training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    tokenizer=student_tokenizer
)

# In[66]:


distilbert_trainer.train()

# In[ ]:



