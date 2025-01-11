import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
import random

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_models_and_tokenizers():
    fine_tuned_model = BertForQuestionAnswering.from_pretrained("./model")
    fine_tuned_tokenizer = BertTokenizerFast.from_pretrained("./model")
    
    base_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    base_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    return (fine_tuned_model, fine_tuned_tokenizer), (base_model, base_tokenizer)

(fine_tuned_model, fine_tuned_tokenizer), (base_model, base_tokenizer) = load_models_and_tokenizers()

# Define a function for Question Answering
def answer_question(model, tokenizer, question, context):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely start and end token indices
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1  # End index is inclusive

    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )
    return answer

# Pre-fed questions and contexts
sample_data = [
    {"context": "The capital of France is Paris. It is known for the Eiffel Tower.", "question": "What is the capital of France?"},
    {"context": "Python is a popular programming language created by Guido van Rossum.", "question": "Who created Python?"},
    {"context": "The Great Wall of China is a historic structure located in China.", "question": "Where is the Great Wall located?"},
    {"context": "Albert Einstein developed the theory of relativity.", "question": "Who developed the theory of relativity?"},
    {"context": "Mount Everest is the tallest mountain in the world.", "question": "What is the tallest mountain in the world?"},
]

# Streamlit UI
st.title("Question Answering with Fine-Tuned and Base BERT Models")
st.write("Provide a context and ask a question, or select a random sample from the predefined list.")

# Option to get a random pre-fed question and context
if st.button("Get Random Sample"):
    random_sample = random.choice(sample_data)
    st.session_state.context = random_sample["context"]
    st.session_state.question = random_sample["question"]

# Input fields with session state
context = st.text_area("Context:", st.session_state.get("context", "Enter the context here..."))
question = st.text_input("Question:", st.session_state.get("question", "Enter your question here..."))

# Generate answers
if st.button("Get Answer"):
    if context and question:
        fine_tuned_answer = answer_question(fine_tuned_model, fine_tuned_tokenizer, question, context)
        base_answer = answer_question(base_model, base_tokenizer, question, context)
        
        st.subheader("Answers:")
        st.write("**Fine-Tuned Model Answer:**", fine_tuned_answer)
        st.write("**Base Model Answer:**", base_answer)
    else:
        st.warning("Please provide both a context and a question.")
