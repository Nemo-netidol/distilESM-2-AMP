from transformers import (
    EsmConfig,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)

"""
Training script for Knowledge Distillation of ESM-2 models.

This script implements a knowledge distillation process where distilESM-2-AMP, a smaller "student"
model, learns from a larger "teacher" ESM-2 model. It uses a custom trainer
to compute a combined loss function consisting of Masked Language Modeling (MLM)
loss and Kullback-Leibler (KL) Divergence loss between the teacher's and
student's logits.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import Dataset, load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import wandb
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_from_disk("../tokenized_data/")['train']
print(f"This file contains {dataset.shape[0]} rows")

# Tokenize function
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') # Tokenzier for ESM

def tokenize_function(examples):
  """
  Tokenizes protein sequences for the ESM model.

  Args:
      examples (dict): A dictionary containing the sequence data, expected to have a 'seq' key.

  Returns:
      dict: A dictionary containing input_ids and attention_mask padded to max_length.
  """
  return tokenizer(examples['seq'], padding="max_length", truncation=True, max_length=256)


# Student Model

# teacher model, ESM 8M
# Original Model: https://huggingface.co/facebook/esm2_t6_8M_UR50D
teacher_model = EsmForMaskedLM.from_pretrained('facebook/esm2_t6_8M_UR50D')

# Student Model
# More information on ESMConfig: https://huggingface.co/docs/transformers/en/model_doc/esm#transformers.EsmConfig
student_config = EsmConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')
student_config.num_hidden_layers = 3 # Transformer layer

student_model = EsmForMaskedLM(student_config)

# facebook/esm2_t36_3B_UR50D -> teacher model
def initialize_weight_from_teacher(student, teacher):
  """
  Initializes the student model's weights using the teacher model's weights.

  This function copies the embeddings, specific encoder layers (skipping every other layer
  to reduce depth), and the language modeling head from the teacher to the student.

  Args:
      student (EsmForMaskedLM): The student model to be initialized.
      teacher (EsmForMaskedLM): The teacher model acting as the source of weights.
  """
  student.esm.embeddings.load_state_dict(teacher.esm.embeddings.state_dict())

  # Copy Encoder Layers (student encoder index[0, 1, 2] = teacher encoder layer[0, 2, 4])
  for i in range(3):
    student.esm.encoder.layer[i].load_state_dict(teacher.esm.encoder.layer[i*2].state_dict())

  # lm_head for training MLM
  student.lm_head.load_state_dict(teacher.lm_head.state_dict())



# Customed Trainer

# For more information on Custom trainer: https://huggingface.co/docs/transformers/main/en/trainer
class DistilledESMTrainer(Trainer):
  """
  Custom Trainer for Knowledge Distillation.

  This trainer extends the Hugging Face Trainer to include a distillation loss.
  The total loss is a combination of the student's Cross-Entropy loss (MLM)
  and the KL Divergence between the student's and teacher's probability distributions.
  """
  def __init__(self, teacher, temperature=2, alpha=0.5, *args, **kwargs):
    """
    Initializes the DistilledESMTrainer.

    Args:
        teacher (nn.Module): The pre-trained teacher model.
        temperature (float, optional): Hyperparameter to soften the probability distribution. Defaults to 2.
        alpha (float, optional): Weight for the student's task-specific loss (MLM). Defaults to 0.5.
    """
    super().__init__(*args, **kwargs)
    self.teacher = teacher
    self.teacher.eval()
    self.temperature = temperature
    self.alpha = alpha

    self.train_preds = []
    self.train_labels = []
    self.steps_since_log = 0
    self.log_frequency = 100

  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """
    Computes the weighted training loss for knowledge distillation.

    Calculates the Masked Language Modeling (MLM) loss for the student and the
    KL Divergence loss between the student and teacher logits.

    Args:
        model (nn.Module): The student model being trained.
        inputs (dict): The input batch containing input_ids, attention_mask, and labels.
        return_outputs (bool, optional): Whether to return the model outputs along with the loss.
                                         Defaults to False.

    Returns:
        torch.Tensor: The computed loss.
    """
    labels = inputs.get("labels")
    outputs_student = model(**inputs, output_hidden_states=True)

    with torch.no_grad():
      outputs_teacher = self.teacher(**inputs, output_hidden_states=True)

      mask = labels != -100

    # MLM loss
    loss_mlm = outputs_student.loss

    # KL-divergence
    student_logits = outputs_student.logits / self.temperature
    teacher_logits = outputs_teacher.logits / self.temperature


    loss_kl = F.kl_div(
        F.log_softmax(student_logits[mask], dim=-1),
        F.softmax(teacher_logits[mask], dim=-1), reduction="batchmean") * (self.temperature ** 2)

    # Calculate 
    loss = self.alpha * loss_mlm + (1-self.alpha) * loss_kl

    with torch.no_grad():
      preds = torch.argmax(outputs_student.logits, dim=-1)
      preds_flat = preds.cpu().numpy().flatten()
      labels_flat = labels.cpu().numpy().flatten()

      mask_flat = labels_flat != -100

      self.train_preds.extend(preds_flat[mask_flat])
      self.train_labels.extend(labels_flat[mask_flat])

    self.steps_since_log += 1

    if self.steps_since_log >= self.log_frequency:
      if len(self.train_labels) > 0:
        train_accuracy = accuracy_score(self.train_labels, self.train_preds)

        self.log({
              "train/accuracy": train_accuracy,
              "train/loss_mlm": loss_mlm.item(),
              "train/loss_kl": loss_kl.item(),
            })
        
        self.train_preds = []
        self.train_labels = []
        self.steps_since_log = 0

    return (loss, outputs_student) if return_outputs else loss


# MLM Training Session

def main():
  """
  Main execution function for the training script.

  Sets up the environment, initializes models, configures the trainer, and starts the
  training process. Finally, saves the trained student model and tokenizer.
  """

  os.environ["WANDB_PROJECT"]="distil-esm2"
  os.environ["WANDB_WATCH"]="false"

  initialize_weight_from_teacher(student_model, teacher_model)

  student_model.to(device)
  teacher_model.to(device)

  # Set masking probability to 15%
  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer,
      mlm_probability=0.15,
  )

  def compute_metrics(eval_pred):
      logits, labels = eval_pred

      if isinstance(logits, (tuple, list)):
          logits = logits[0]

      # Get predictions
      preds = np.argmax(logits, axis=-1) 

      # Flatten both predictions and labels
      preds = preds.flatten()
      labels = labels.flatten()

      # Filter out ignored tokens (-100)
      mask = labels != -100  # -100 is the ignore index
      preds = preds[mask]
      labels = labels[mask]

      # Calculate accuracy
      acc = accuracy_score(labels, preds)
      return {"accuracy": acc}

  training_args = TrainingArguments(
      # Train batch size
      per_device_train_batch_size=256,
      # Training
      learning_rate=5e-5,
      num_train_epochs=3,
      warmup_steps=1000,
      # Logging
      logging_first_step=True,
      logging_strategy='steps',
      logging_steps=100, # frequency of logging to wandb
      # Saving
      save_strategy='steps',
      save_steps=50000,
      save_total_limit=3,
      output_dir='./distil-esm2',
      metric_for_best_model='loss',
      resume_from_checkpoint=True,
      greater_is_better=False, # False for loss
      gradient_checkpointing=True,
      load_best_model_at_end=False,
      # Memory Optimization
      gradient_accumulation_steps=1,
      # Others
      report_to='wandb',
      fp16=True,
      disable_tqdm=False,
      
  )
      
  trainer = DistilledESMTrainer(
      args=training_args,
      teacher=teacher_model,
      model=student_model,
      data_collator=data_collator,
      train_dataset=dataset,
      eval_dataset=None,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,

  )

  start = time.time()
  trainer.train()
  wandb.finish()
  elapsed = time.time() - start
  print(f"Training MLM using {elapsed/60:.2f} minutes")

  save_dir = "./student_model"

  student_model.save_pretrained(save_dir)
  tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
  main()
