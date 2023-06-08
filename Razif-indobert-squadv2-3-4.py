from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, DefaultDataCollator
import torch
import numpy as np
import time
import shutil
import os
import wandb


# constant variable 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "rahmanfadhil/squad_v2_id"
MODEL_CHECKPOINT = "indobenchmark/indobert-base-p1"

BASE_MODEL = 'IndoBERT'
MODEL_NAME = "IndoBERT-SQuADv2"

HF_TOKEN = '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'
WANDB_KEY = 'f24435d851b3bd0bc0a590bb865ec8eb173bac59'

PROJECT_NAME = f"{MODEL_NAME}_{str(time.time()).split('.')[0]}"


# Load Dataset 
dataset = load_dataset(DATASET)

# load model 
# correct output shape
data_collator = DefaultDataCollator()
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
config.num_labels = 2

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT, config=config)
print(model)

# Preprocessing
context = dataset["train"][0]["context"]
question = dataset["train"][0]["question"]

inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue 
            
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


dataset_train_tokenized = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
dataset_eval_tokenized = dataset["validation"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)

# Wandb logging
wandb.login(key=WANDB_KEY)

# method
sweep_config = {
    'method': 'grid'
}

# full hyperparameters
# parameters_dict = {
#     'epochs': {
#         'values': [10]
#         },
#     'batch_size': {
#         'values': [8, 16]
#         },
#     'learning_rate': {
#         'values': [2e-5, 2e-6]
#         },
#     'weight_decay': {
#         'values': [0.01]
#     },
# }

# # hyperparameter experiment 1, 2
# parameters_dict = {
#     'epochs': {
#         'values': [5]
#         },
#     'batch_size': {
#         'values': [8]
#         },
#     'learning_rate': {
#         'values': [2e-5, 2e-6]
#         },
#     'weight_decay': {
#         'values': [0.01]
#     },
# }

# hyperparameter experiment 3, 4
parameters_dict = {
    'epochs': {
        'values': [5]
        },
    'batch_size': {
        'values': [16]
        },
    'learning_rate': {
        'values': [2e-5, 2e-6]
        },
    'weight_decay': {
        'values': [0.01]
    },
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

working_dir = str(os.getcwd()) + "/"

def train(config=None):
    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config
        
        # model name that will be saved on both wandb and huggingface
        saved_model_name = f"{PROJECT_NAME}-{config.batch_size}-{config.learning_rate}-{config.weight_decay}-{config.epochs}"
        
        training_args = TrainingArguments(
            output_dir=saved_model_name,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            weight_decay=config.weight_decay,
            hub_token=HF_TOKEN,
            report_to="wandb",
            push_to_hub=True,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model.to(DEVICE),
            args=training_args,
            train_dataset=dataset_train_tokenized,
            eval_dataset=dataset_eval_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model()
        shutil.rmtree(working_dir + saved_model_name)


wandb.agent(sweep_id, train)
wandb.finish()