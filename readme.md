# T5 Text Summarization on CNN/DailyMail

This project demonstrates **fine-tuning T5-small for text summarization** using the CNN/DailyMail dataset in **TensorFlow/Keras**. The notebook includes full preprocessing, model training, evaluation with ROUGE scores, and inference.

---

## 📌 Project Overview

- **Task:** Text summarization (abstractive)  
- **Dataset:** [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)  
- **Model:** T5-small (fine-tuned)  
- **Framework:** TensorFlow 2.x with Hugging Face Transformers 4.57  
- **Evaluation:** ROUGE-1, ROUGE-2, ROUGE-L  

This project demonstrates **end-to-end workflow** from preprocessing to model evaluation and sample generation.

---

## 🛠 Features

1. **Data Preprocessing**
   - Tokenization of articles and summaries using T5 tokenizer
   - Input truncation (`max_length=512`) and target truncation (`max_length=150`)
   - Preparation of TensorFlow `tf.data.Dataset` for training and validation

2. **Model Training**
   - Fine-tuned T5-small using `TFT5ForConditionalGeneration`
   - Loss tracking and validation
   - Early stopping and checkpointing (weights saved as `.h5`)

3. **Inference**
   - Generate summaries on validation/test articles
   - Configurable beam search and maximum generation length
   - Example generation code included in notebook

4. **Evaluation**
   - ROUGE evaluation using `evaluate` library
   - Optional subset evaluation for fast results
   - Sample outputs compared with reference summaries

5. **Visualizations**
   - Article and summary length distributions
   - Token length distributions
   - Training & validation loss curves
   - ROUGE score trends (optional)
   - Sample article → reference → generated summary tables

---



## 📁 Repository Structure
T5-Text-Summarization/
│
├─ notebook.ipynb # Full training & evaluation notebook
├─ t5_best.weights.h5 # Saved model weights
├─ README.md # Project description
└─ requirements.txt # Required Python packages


---

## ⚡ Usage

### 1. Install Dependencies

```bash
pip install tensorflow transformers[torch] datasets evaluate rouge_score
```

### 2. Load Model and Weights

```bash
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

checkpoint = "t5-small"
model = TFT5ForConditionalGeneration.from_pretrained(checkpoint, from_pt=True)
model.load_weights("./t5_best.weights.h5")

tokenizer = T5Tokenizer.from_pretrained(checkpoint)
```


### 3. Generate Summaries


```bash
input_texts = ["Your input article text goes here."]
inputs = tokenizer(input_texts, return_tensors="tf", max_length=512, truncation=True)

generated_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=150,
    num_beams=2
)

summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(summaries)
```


### 4. Evaluate on Validation Set


```bash
import evaluate
rouge = evaluate.load("rouge")

predictions, references = [], []

for batch in val_dataset.take(50):  # Subset for fast evaluation
    input_ids = batch['input_ids'].numpy()
    attention_mask = batch['attention_mask'].numpy()
    labels = batch['labels'].numpy()

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_beams=1  # greedy decoding for speed
    )

    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions.extend(preds)
    references.extend(refs)

results = rouge.compute(predictions=predictions, references=references)
print(results)
```


### 📊 Expected Results


Validation loss after fine-tuning: ~0.7–0.8

ROUGE scores (T5-small, 3 epochs) on subset:

ROUGE-1: ~0.42–0.45

ROUGE-2: ~0.19–0.22

ROUGE-L: ~0.39–0.42

Scores may vary depending on the exact number of epochs, batch size, and generation parameters.



Expected Results

- Validation loss after fine-tuning: ~0.7–0.8

- ROUGE scores (T5-small, 3 epochs) on subset:

- ROUGE-1: ~0.42–0.45

- ROUGE-2: ~0.19–0.22

- ROUGE-L: ~0.39–0.42

Scores may vary depending on the exact number of epochs, batch size, and generation parameters.

