# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import nltk
nltk.download('punkt')

# Load the CNN/Daily Mail dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')

# Display the first example from the dataset
print(dataset['train'][0])

# Preprocess the dataset
def preprocess_data(examples):
    articles = examples['article']
    summaries = examples['highlights']
    return articles, summaries

train_articles, train_summaries = preprocess_data(dataset['train'])
val_articles, val_summaries = preprocess_data(dataset['validation'])
test_articles, test_summaries = preprocess_data(dataset['test'])

# Load the pre-trained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to generate summaries
def generate_summary(article, max_length=130, min_length=30):
    inputs = tokenizer([article], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=max_length, min_length=min_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Test the model on a sample article
sample_article = test_articles[0]
print("Original Article:", sample_article)
print("Generated Summary:", generate_summary(sample_article))

# Fine-tuning the model (optional)
from transformers import Trainer, TrainingArguments

# Prepare the dataset for training
def tokenize_data(examples):
    inputs = tokenizer(examples['article'], max_length=1024, truncation=True, padding='max_length')
    targets = tokenizer(examples['highlights'], max_length=150, truncation=True, padding='max_length')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': targets['input_ids']}

tokenized_train = dataset['train'].map(tokenize_data, batched=True, remove_columns=['article', 'highlights'])
tokenized_val = dataset['validation'].map(tokenize_data, batched=True, remove_columns=['article', 'highlights'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=10,
    eval_steps=10,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on real-world articles
def evaluate_model(articles, summaries):
    generated_summaries = [generate_summary(article) for article in articles]
    for i in range(5):
        print(f"Article {i+1}:")
        print("Original Article:", articles[i])
        print("Original Summary:", summaries[i])
        print("Generated Summary:", generated_summaries[i])
        print("\n")

evaluate_model(test_articles[:5], test_summaries[:5])

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_bart')
tokenizer.save_pretrained('./fine_tuned_bart')