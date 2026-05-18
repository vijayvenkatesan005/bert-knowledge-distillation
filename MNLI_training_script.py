#!/usr/bin/env python 
 
# %% [markdown] 
# ## Mounting my google drive account to the colab notebook 
 
# %% [markdown] 
# ## Loading the MNLI dataset and tokenizer 
 
# %%

import argparse

parser = argparse.ArgumentParser(
  prog="MNLI Trainer Script",
  description="Trains BERT on the MNLI dataset",
  epilog="Use at your own risk"
)

parser.add_argument("-lr", "--learning_rate", help="The learning rate for training", type=float)
parser.add_argument("-bs", "--batch_size", help="The batch size for training", type=int)
parser.add_argument("-a", "--alpha", help="The alpha value for training", type=float)
parser.add_argument("-t", "--temperature", help="The temperature value for training", type=int)

args = parser.parse_args()


from datasets import load_dataset 
from transformers import AutoTokenizer, DataCollatorWithPadding 
 
raw_datasets = load_dataset("glue", "mnli") 
 
checkpoint = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
output_dir = './bert_fine_tuned_on-mnli' 
num_training_epochs = 3 
learning_rate = float(args.learning_rate) if args.learning_rate else 2e-5  
batch_size = int(args.batch_size) if args.batch_size else 16 
alpha = float(args.alpha) if args.alpha else 0.25 
temperature = int(args.temperature) if args.temperature else 2 
weight_decay=0.2 
num_labels = 3

print(f"learning rate: {learning_rate}")
print(f"batch size: {batch_size}")
print(f"alpha: {alpha}")
print(f"temperature: {temperature}")

# %% 
raw_datasets 
 
# %% [markdown] 
# ## Defining a function to tokenize the input 
 
# %% 
def tokenize_input(example): 
  premise = example['premise'] 
  hypothesis = example['hypothesis'] 
  return tokenizer(premise, hypothesis, padding="max_length", max_length=128, truncation=True) 
 
 
# %% [markdown] 
# ## Applying the function that will tokenize the input to every example from the dataset 
 
# %% 
tokenized_datasets = raw_datasets.map(tokenize_input, batched=True) 
 
# %% [markdown] 
# ## Loading the pretrained sequence classification model 
 
# %% 
from transformers import AutoModelForSequenceClassification 
 
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3) 
 
# %% [markdown] 
# ## Defining a function to compute classification metrics 
 
# %% 
import numpy as np 
# import evaluate 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
#  compute accuracy, precision, recall, and f1 score 
# do the distillation process 
 
# accuracy = evaluate.load("accuracy") 
# precision = evaluate.load("precision") 
# recall = evaluate.load("recall") 
# f1 = evaluate.load("f1") 
 
def compute_metrics(eval_pred): 
  logits, labels = eval_pred 
  predictions = np.argmax(logits, axis=1) 
  # acc = accuracy.compute(predictions=predictions, references=labels) 
  # prec = precision.compute(predictions=predictions, references=labels, average="micro") 
  # rec = recall.compute(predictions=predictions, references=labels, average="micro") 
  # f1_score = f1.compute(predictions=predictions, references=labels, average="micro")

  print(f"Predictions: {predictions}")
  print(f"Labels: {labels}")
  print()

  acc = accuracy_score(labels, predictions) 
  prec = precision_score(labels, predictions, average="macro") 
  rec = recall_score(labels, predictions, average="macro") 
  f1 = f1_score(labels, predictions, average="macro") 
  # return {"accuracy": acc["accuracy"], "precision": prec["precision"], "recall": rec["recall"], "f1": f1_score["f1"]} 
  return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1} 
 
 
 
# %% 
train_tokenized = tokenized_datasets["train"].shuffle(seed=42) 
val_tokenized = tokenized_datasets["validation_matched"].shuffle(seed=42) 
 
 
# %% [markdown] 
# ## Distillation Process 
 
# %% 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import Trainer 
 
# %% 
from transformers import TrainingArguments 
 
class DistillationTrainingArguments(TrainingArguments): 
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.alpha = alpha 
        self.temperature = temperature 
 
# %% 
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
 
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd 
        return (loss, outputs_stu) if return_outputs else loss  
             
         
 
# %% 
student_training_args = DistillationTrainingArguments( 
    output_dir=output_dir, 
    report_to=['tensorboard'],
    logging_dir=f"log-lr-{learning_rate}-bs-{batch_size}-a-{alpha}-t-{temperature}",
   # evaluation_strategy='epoch',
   # evaluation_strategy='steps',
   #eval_steps=1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=num_training_epochs, 
    learning_rate=learning_rate, 
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size, 
    alpha=alpha, 
    weight_decay=weight_decay 
) 
 
 
 
# %% 
import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
 
student_ckpt = 'distilbert-base-uncased' 
 
def student_init(): 
    return ( 
        AutoModelForSequenceClassification 
        .from_pretrained(student_ckpt, config=student_config).to(device) 
    ) 
     
 
 
# %% 
teacher_ckpt = 'bert-base-uncased' 
teacher_model = ( 
    AutoModelForSequenceClassification 
    .from_pretrained(teacher_ckpt, num_labels=num_labels) 
    .to(device) 
) 
 
# %% 
from transformers import AutoTokenizer 
 
student_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') 
 
# %% 
id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'} 
label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2} 
 
# %% 
from transformers import AutoConfig 
 
student_config = (AutoConfig 
                  .from_pretrained(student_ckpt, num_labels=num_labels)) 
 
# %% 
distilbert_trainer = DistillationTrainer( 
    model_init=student_init, 
    teacher_model=teacher_model, 
    args=student_training_args, 
    train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['validation_matched'], 
    compute_metrics=compute_metrics, 
    tokenizer=student_tokenizer 
) 
 
distilbert_trainer.train() 
 
# %% 
 
 
# %% 
 
 
 
