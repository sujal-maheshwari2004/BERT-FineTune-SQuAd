import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("squad")

# Use only 1000 entries from the dataset
print("Selecting 1000 entries from the dataset...")
small_train_dataset = dataset["train"].select(range(100))
small_eval_dataset = dataset["validation"].select(range(100))

# Load the pre-trained model and tokenizer
print("Loading pre-trained model and tokenizer...")
model_name = "bert-base-uncased"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping").tolist()
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If the answer is not fully inside the context, label it (0, 0)
        if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise, find the start and end token positions
            start_token = None
            end_token = None
            for idx, (start, end) in enumerate(offset):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break
            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = len(offset) - 1
            start_positions.append(start_token)
            end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("Tokenizing training dataset...")
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True, remove_columns=small_train_dataset.column_names)
print("Tokenizing evaluation dataset...")
tokenized_eval_dataset = small_eval_dataset.map(preprocess_function, batched=True, remove_columns=small_eval_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train the model
print("Starting training...")
for epoch in tqdm(range(training_args.num_train_epochs), desc="Epochs"):
    trainer.train()
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs} completed.")

# Save the model
print("Saving the model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Save the metrics
print("Saving the metrics...")
metrics = trainer.state.log_history
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("training_metrics.csv", index=False)

print("Training completed and model saved.")