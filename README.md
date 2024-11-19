
# LLM Classification Fine-Tuning

This project demonstrates how to fine-tune a large language model (LLM) for a text classification task using the Hugging Face Transformers library. The dataset and task are based on a Kaggle competition where the goal is to classify the outputs of different models as either a win for model A, model B, or a tie.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Training Pipeline](#training-pipeline)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Requirements](#requirements)
8. [Future Work](#future-work)
9. [Contact](#contact)

---

## Overview
Fine-tuning large pre-trained language models on domain-specific tasks has become a common approach in NLP. This repository fine-tunes a pre-trained transformer model (e.g., BERT) to classify outputs from two models (`Model A` and `Model B`) into one of three categories:
- **1**: Model A wins
- **2**: Model B wins
- **3**: Tie

The training pipeline includes:
- Preprocessing text data using tokenizers.
- Combining multiple columns of data into a unified input format.
- Handling multi-class classification.
- Optimizing training with techniques like mixed precision and gradient accumulation.

---

## Dataset
The dataset includes:
- **Training Data**: Contains text from `prompt`, `response_a`, and `response_b`, along with the target labels (`winner_model_a`, `winner_model_b`, `winner_tie`).
- **Test Data**: Similar to training data but without target labels.

We combine the target labels into a single column:
- **1**: Model A wins
- **2**: Model B wins
- **3**: Tie

The dataset is preprocessed using Hugging Face tokenizers, where inputs are formatted as:
```
[CLS] Prompt text [SEP] Response A text [SEP] Response B text [SEP]
```

---

## Model
The project uses a pre-trained transformer model (e.g., `bert-base-uncased`) from Hugging Face. The model is fine-tuned for multi-class classification with a custom head for three output classes.

### Features:
- Pre-trained transformer backbone.
- Classification head with three output nodes.
- Mixed precision training for faster execution.

---

## Training Pipeline
The training pipeline includes:
1. **Data Preprocessing**:
   - Tokenization of input text using Hugging Face's `AutoTokenizer`.
   - Mapping multi-column inputs to a unified text format.

2. **Model Setup**:
   - Pre-trained model loaded with `AutoModelForSequenceClassification`.
   - Configured for three-class classification.

3. **Training**:
   - Optimized using techniques like gradient accumulation and mixed precision training.
   - Checkpoints are saved periodically based on `save_steps`.

4. **Evaluation**:
   - Optional: Splits the training dataset into a training and validation set.

5. **Prediction and Submission**:
   - Generates predictions for the test set in Kaggle submission format.

---

## Results
The model achieves competitive results by leveraging transfer learning. Fine-tuning pre-trained transformers allows for effective classification even with limited labeled data.

### Key Metrics:
- **Accuracy**: TBD
- **F1-Score**: TBD

---

## How to Run
### Clone the Repository
```bash
git clone https://github.com/Poulami-Nandi/LLM_response_categorization.git
cd LLM_response_categorization
```

### Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Training the Model
Run the training script:
```bash
python train.py
```

### Generating Predictions
Run the prediction script:
```bash
python predict.py
```

### Submit to Kaggle
Use the generated `submission.csv` file to submit predictions:
```bash
kaggle competitions submit -c llm-classification-finetuning -f submission.csv -m "First submission"
```

---

## Requirements
- Python 3.8+
- Libraries:
  - `transformers`
  - `torch`
  - `scikit-learn`
  - `pandas`
  - `numpy`
- GPU (recommended for faster training)

---

## Future Work
- Explore additional models like RoBERTa, DistilBERT, or GPT-based architectures.
- Incorporate hyperparameter tuning with tools like Optuna.
- Experiment with data augmentation techniques.
- Add more robust evaluation strategies (e.g., cross-validation).

---

## Contact
For questions or collaboration opportunities, feel free to reach out!

- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/poulami-nandi-a8a12917b)
- **Email**: nandi.poulami91@gmail.com
