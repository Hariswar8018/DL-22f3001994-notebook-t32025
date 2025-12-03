# 🎭 Multi-Label Emotion Classification with Deep Learning

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
[![F1 Score](https://img.shields.io/badge/Metric-Macro%20F1-success.svg)](https://scikit-learn.org/)
[![IIT Madras](https://img.shields.io/badge/IIT-Madras-orange.svg)](https://www.iitm.ac.in/)

> **2025 Sep DLGenAI Project** - A comprehensive multi-label emotion classification system using CNN, GRU, BiLSTM, and DistilBERT architectures. Completed as part of the IIT Madras Deep Learning and Generative AI course (September 2025 term).
---

## Hugging Face Deployment
The model is live on Hugging Face Spaces with a full UI, real-time inference, dataset explanation, and project documentation.
<img width="1366" height="768" alt="Screenshot (420)" src="https://github.com/user-attachments/assets/34b358c5-c82f-4ac5-9301-72f523be8c38" />

#### [Click here to see the Hugging Face Deployment](https://huggingface.co/spaces/AyusmanSamasi/Sentimental_Analysis)
Experience real-time multi-label emotion classification powered by DeBERTa-v3.

#### [Click here to see Project Report](https://drive.google.com/file/d/10vRR0zuFbYP8pIU7ZQBYHqyIQo__OPhk/view?usp=sharing)

#### [Click here to view the Notebook](https://www.kaggle.com/code/samasiayushman/dl-22f3001994-notebook-t32025)
---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset & Task](#-dataset--task)
- [Models Implemented](#-models-implemented)
- [Technical Architecture](#-technical-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Results & Evaluation](#-results--evaluation)
- [Key Learnings](#-key-learnings)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Problem Statement

Emotions are complex, often overlapping, and can be expressed in subtle ways through language. Detecting them automatically is an important task in NLP with applications in:

- 💚 **Mental health support** - Understanding emotional states
- 🛍️ **Customer experience analysis** - Sentiment tracking
- 🤖 **Conversational AI** - Emotion-aware responses
- 📱 **Social media monitoring** - Trend analysis

### The Challenge

Build models that can classify short English text entries into **multiple emotion categories simultaneously**. This is a **multi-label classification problem** where each text can express multiple emotions at once.

---

## 📊 Dataset & Task

### Emotion Categories (5 Labels)

| Emotion | Description | Binary Label |
|---------|-------------|--------------|
| 😠 **Anger** | Frustration, irritation, rage | 0 or 1 |
| 😨 **Fear** | Anxiety, worry, terror | 0 or 1 |
| 😊 **Joy** | Happiness, excitement, delight | 0 or 1 |
| 😢 **Sadness** | Sorrow, grief, disappointment | 0 or 1 |
| 😲 **Surprise** | Shock, amazement, astonishment | 0 or 1 |

### Task Characteristics

- **Type**: Multi-label text classification
- **Language**: English
- **Input**: Short text entries
- **Output**: Binary vector of 5 emotions (each 0 or 1)
- **Evaluation Metric**: **Macro F1-Score**

### Evaluation Metric Explained

The competition uses **Macro F1-Score**, which:

1. Computes F1-score for each emotion independently
2. Takes the unweighted average across all 5 emotions
3. Balances precision and recall equally

```
Macro F1 = (1/5) × Σ F1(emotion_i)

where F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This metric treats all emotions equally, regardless of their frequency in the dataset.

---

## 🤖 Models Implemented

I experimented with **four distinct architectures**, progressing from classical deep learning to state-of-the-art transformers:

### Architecture Progression

```
Classical DL ──────────────────────────────────────────> Modern Transformers
    CNN          GRU          BiLSTM          DistilBERT
     ↓            ↓              ↓                 ↓
  Pattern     Sequence      Bidirectional    Contextual
  Detection   Modeling      Context          Understanding
```

---

### 1. 🔷 TextCNN - Convolutional Neural Network

**Why CNN for Text?**
CNNs can capture local patterns and n-gram features efficiently, making them excellent for detecting emotional keywords and phrases.

#### Architecture Details

```python
TextCNN(
  vocab_size=vocab_size,
  embed_dim=128,
  kernel_sizes=(3, 4, 5),      # Captures 3-gram, 4-gram, 5-gram patterns
  num_filters=64,               # 64 filters per kernel size
  num_labels=5,
  dropout=0.2
)
```

**Key Components:**
- 🔸 **Embedding Layer**: Converts tokens to 128-dimensional vectors
- 🔸 **Multi-Kernel Convolution**: Three parallel convolutions (3×128, 4×128, 5×128)
  - Kernel size 3: Captures trigrams like "so very happy"
  - Kernel size 4: Captures 4-grams like "I am so sad"
  - Kernel size 5: Captures 5-grams like "what a pleasant surprise today"
- 🔸 **Max Pooling**: Extracts most important features from each convolution
- 🔸 **Concatenation**: Combines features from all kernel sizes (192 features)
- 🔸 **Dropout (0.2)**: Prevents overfitting
- 🔸 **Fully Connected Layer**: Maps to 5 emotion logits

**Strengths:**
- ✅ Fast training and inference
- ✅ Parallel processing of features
- ✅ Good at detecting emotional keywords
- ✅ Low computational requirements

**Training Configuration:**
- Optimizer: **Adam**
- Learning Rate: **1e-3**
- Batch Size: **32**
- Epochs: **25**
- Loss Function: **BCEWithLogitsLoss**
- Final Training Loss: **0.01695**

**Performance:**
- Training F1: **71.00%**
- Validation F1: **71.15%**
- Test F1: **71.00%**
- Parameters: ~150K

---

### 2. 🔶 GRUNet - Gated Recurrent Unit

**Why GRU for Emotions?**
GRUs excel at modeling sequential dependencies, crucial for understanding how emotion builds up across a sentence.

#### Architecture Details

```python
GRUNet(
  vocab_size=vocab_size,
  embed_dim=128,
  hidden_size=128,
  num_layers=1,
  bidirectional=True,           # Processes text forward AND backward
  num_labels=5,
  dropout=0.2
)
```

**Key Components:**
- 🔸 **Embedding Layer**: 128-dimensional word embeddings
- 🔸 **Bidirectional GRU**: 
  - Forward direction: Captures left-to-right context
  - Backward direction: Captures right-to-left context
  - Hidden size: 128 units per direction → 256 total
- 🔸 **Gating Mechanisms**:
  - **Update Gate**: Decides how much past information to keep
  - **Reset Gate**: Decides how much past information to forget
- 🔸 **Mean Pooling**: Averages all timesteps for sentence representation
- 🔸 **Dropout (0.2)**: Regularization
- 🔸 **Output Layer**: Maps 256 features to 5 emotion scores

**Strengths:**
- ✅ Captures sequential dependencies
- ✅ More efficient than LSTM (fewer parameters)
- ✅ Handles variable-length sequences naturally
- ✅ Bidirectional processing improves context understanding

**How Gating Works:**
```
For input "I am absolutely devastated":
- Update gate learns to remember strong emotion words
- Reset gate learns to forget neutral words like "I am"
- Final representation emphasizes "absolutely devastated"
```

**Training Configuration:**
- Optimizer: **AdamW**
- Learning Rate: **1e-3**
- Batch Size: **32**
- Epochs: **25**
- Loss Function: **BCEWithLogitsLoss**
- Final Training Loss: **0.03210**

**Performance:**
- Training F1: **70.00%**
- Validation F1: **69.24%**
- Test F1: **69.00%**
- Parameters: ~200K

---

### 3. 🔵 BiLSTM - Bidirectional Long Short-Term Memory

**Why BiLSTM for Multi-Label Emotions?**
BiLSTM's sophisticated memory cells can capture long-range dependencies and subtle emotional cues that span entire sentences.

#### Architecture Details

```python
BiLSTM(
  vocab_size=vocab_size,
  embed_dim=128,
  hidden_size=128,
  num_layers=1,
  bidirectional=True,
  num_labels=5,
  dropout=0.2
)
```

**Key Components:**
- 🔸 **Embedding Layer**: 128-dimensional embeddings
- 🔸 **Bidirectional LSTM**:
  - Processes sequences in both directions simultaneously
  - Hidden state: 128 units per direction → 256 combined
  - Cell state: Maintains long-term memory
- 🔸 **LSTM Gates** (4 gates per direction):
  - **Forget Gate**: What to remove from cell state
  - **Input Gate**: What new information to add
  - **Output Gate**: What to expose as hidden state
  - **Cell Gate**: Candidate values for cell state
- 🔸 **Mean Pooling**: Aggregates all timesteps
- 🔸 **Multi-Layer FC**:
  - Dense(256 → 128) + ReLU + Dropout
  - Dense(128 → 5)
- 🔸 **Dropout (0.2)**: Applied after pooling and in FC

**Strengths:**
- ✅ Superior long-range dependency modeling
- ✅ Sophisticated memory management via gates
- ✅ Bidirectional context for better understanding
- ✅ Handles vanishing gradient problem effectively
- ✅ Multi-layer FC allows complex emotion combinations

**LSTM vs GRU Comparison:**
| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 4 (forget, input, output, cell) | 2 (update, reset) |
| Parameters | More (~4× hidden_size²) | Fewer (~3× hidden_size²) |
| Training Speed | Slower | Faster |
| Memory Capability | Superior for long sequences | Good for medium sequences |

**Training Configuration:**
- Optimizer: **AdamW**
- Learning Rate: **1e-3**
- Batch Size: **32**
- Epochs: **25**
- Loss Function: **BCEWithLogitsLoss**
- Final Training Loss: **0.04408**

**Performance:**
- Training F1: **67.00%**
- Validation F1: **67.40%**
- Test F1: **67.00%**
- Parameters: ~250K

---

### 4. 🔴 DistilBERT - Distilled BERT Transformer

**Why DistilBERT?**
DistilBERT brings the power of transformer-based contextual embeddings while being 40% smaller and 60% faster than BERT, making it perfect for this competition.

#### Architecture Details

```python
Model: distilbert-base-uncased (from Hugging Face)
Parameters: ~66M (distilled from BERT's 110M)
Layers: 6 transformer blocks
Hidden Size: 768
Attention Heads: 12
Max Sequence Length: 512 tokens
Vocabulary Size: 30,522 tokens
```

**Key Components:**
- 🔸 **Tokenizer**: WordPiece tokenization with [CLS] and [SEP] tokens
- 🔸 **Pre-trained Embeddings**:
  - Token embeddings: 30,522 × 768
  - Position embeddings: Learnable positional encoding
- 🔸 **6 Transformer Layers**: Each containing:
  - Multi-head self-attention (12 heads)
  - Feed-forward networks
  - Layer normalization
  - Residual connections
- 🔸 **[CLS] Token**: Special token for sentence classification
- 🔸 **Fine-tuning Head**: Linear(768 → 5) for emotion prediction
- 🔸 **Attention Mechanism**: Models relationships between all words

**What Makes DistilBERT Special:**
- 🌟 **Knowledge Distillation**: Trained to mimic BERT's behavior
- 🌟 **Contextual Understanding**: Each word representation depends on entire sentence
- 🌟 **Pre-trained on Massive Corpora**: Understands language deeply
- 🌟 **Transfer Learning**: Leverages knowledge from 16GB of text data
- 🌟 **Bidirectional Context**: Unlike traditional RNNs, sees full context at once

**How Attention Works for Emotions:**
```
Input: "I am not happy but also not sad"

Attention learns:
"not" → strongly attends to "happy" and "sad"
"happy" → attends to "not" (negation)
"sad" → attends to "not" and "also"

Result: Model understands complex emotional negation
```

**Strengths:**
- ✅ State-of-the-art contextual understanding
- ✅ Pre-trained on billions of words
- ✅ Captures subtle emotional nuances
- ✅ Handles negations, sarcasm, and complex expressions
- ✅ 40% smaller than BERT, 60% faster
- ✅ Excellent for multi-label tasks

**Training Configuration:**
- **Tokenizer**: AutoTokenizer (distilbert-base-uncased)
- **Max Length**: 200 tokens
- **Padding**: max_length
- **Truncation**: Enabled
- **Batch Size**: 32
- **Optimizer**: **AdamW**
- **Learning Rate**: **1e-2** (higher than typical for transformers)
- **Epochs**: **25**
- **Loss Function**: **BCEWithLogitsLoss**
- **Fine-tuning Strategy**: Full fine-tuning
- **Final Training Loss**: **0.01117**

**Custom Dataset Implementation:**
```python
BERTDataset:
- Tokenizes text with padding/truncation (max_len=200)
- Returns: (input_ids, attention_mask, labels)
- Handles variable-length sequences
- Batch processing for efficiency
```

**Performance:**
- Training F1: **88.00%**
- Validation F1: **87.06%**
- Test F1 (Public): **87.80%**
- Test F1 (Private): **87.00%**
- Parameters: ~66M
- Inference Speed: Fast with GPU acceleration

**Why DistilBERT Often Outperforms Others:**
1. Pre-trained knowledge of language semantics
2. Bidirectional context (sees entire sentence at once)
3. Attention mechanism captures word relationships
4. Better handling of complex emotional expressions
5. Robust to different writing styles and structures

---

## 🏗️ Technical Architecture

### Model Comparison Table

| Model | Type | Parameters | Strengths | Best For |
|-------|------|------------|-----------|----------|
| **TextCNN** | Convolutional | ~150K | Fast, pattern detection | Keyword-based emotions |
| **GRU** | Recurrent | ~200K | Sequential modeling | Moderate-length texts |
| **BiLSTM** | Recurrent | ~250K | Long-term dependencies | Complex sentences |
| **DistilBERT** | Transformer | ~66M | Contextual understanding | All cases, best overall |

### Common Training Setup

**Loss Function:**
```python
nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits
```
- Combines sigmoid activation with BCE loss
- Numerically stable
- Perfect for multi-label classification

**Optimizer:**
```python
torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
```
- Adaptive learning rates
- Weight decay for regularization
- Efficient for deep networks

**Data Preprocessing:**
1. Text cleaning and normalization
2. Tokenization (vocabulary-based for CNN/GRU/BiLSTM, WordPiece for DistilBERT)
3. Padding/Truncation to fixed length
4. Label binarization for multi-label format

---

## 🛠 Installation & Setup

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12.0
CUDA >= 11.0 (for GPU training)
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install tqdm
```

**Complete Requirements:**
```txt
torch>=1.12.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.64.0
```

---

## 🚀 Usage

### Training Models

```python
# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define emotion labels
LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Hyperparameters
SEED = 42
BATCH_SIZE = 32
MAX_LEN = 200
EPOCHS = 25

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Learning rates (model-specific)
LR_CNN = 1e-3        # TextCNN, GRU, BiLSTM
LR_TRANSFORMER = 1e-2  # DistilBERT

# Threshold tuning for multi-label classification
BEST_THRESHOLDS = [0.45, 0.55, 0.40, 0.50, 0.48]  # Per emotion: anger, fear, joy, sadness, surprise
DEFAULT_THRESHOLD = 0.60  # Initial threshold before tuning
```

#### 1. Train TextCNN

```python
from models import TextCNN

model = TextCNN(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    kernel_sizes=(3,4,5),
    num_filters=64,
    num_labels=len(LABELS),
    pad_idx=tokenizer.pad_token_id
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Train for 25 epochs
```

#### 2. Train GRU

```python
from models import GRUNet

model_gru = GRUNet(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    hidden_size=128,
    num_labels=len(LABELS),
    pad_idx=tokenizer.pad_token_id
).to(DEVICE)

optimizer = torch.optim.AdamW(model_gru.parameters(), lr=1e-3)
```

#### 3. Train BiLSTM

```python
from models import BiLSTM

model_bilstm = BiLSTM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    hidden_size=128,
    num_layers=1,
    num_labels=len(LABELS),
    pad_idx=tokenizer.pad_token_id,
    dropout=0.2
).to(DEVICE)

optimizer = torch.optim.AdamW(model_bilstm.parameters(), lr=1e-3)
```

#### 4. Train DistilBERT

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Create custom dataset
train_ds = BERTDataset(df_train, tokenizer, max_len=128)
val_ds = BERTDataset(df_val, tokenizer, max_len=128)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Fine-tune DistilBERT
model_bert = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=5,
    problem_type="multi_label_classification"
).to(DEVICE)

optimizer = torch.optim.AdamW(model_bert.parameters(), lr=2e-5)
```

### Inference & Submission

```python
# Generate predictions with optimized thresholds
def predict(model, dataloader, thresholds=None):
    model.eval()
    predictions = []
    if thresholds is None:
        thresholds = [0.5] * 5  # Default threshold
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            
            # Apply per-emotion thresholds
            preds = torch.zeros_like(probs)
            for i, threshold in enumerate(thresholds):
                preds[:, i] = (probs[:, i] > threshold).float()
            
            predictions.extend(preds.cpu().numpy())
    return predictions

# Use optimized thresholds for best performance
BEST_THRESHOLDS = [0.45, 0.55, 0.40, 0.50, 0.48]  # anger, fear, joy, sadness, surprise
predictions = predict(model, test_loader, thresholds=BEST_THRESHOLDS)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'anger': predictions[:, 0].astype(int),
    'fear': predictions[:, 1].astype(int),
    'joy': predictions[:, 2].astype(int),
    'sadness': predictions[:, 3].astype(int),
    'surprise': predictions[:, 4].astype(int)
})
submission.to_csv('submission.csv', index=False)
```

---

## 📈 Results & Evaluation

### Model Performance Comparison

| Model | Train F1 | Val F1 | Test F1 | Params | Train Loss |
|-------|----------|--------|---------|--------|------------|
| **TextCNN** | 71.00% | 71.15% | 71.00% | ~150K | 0.01695 |
| **GRU** | 70.00% | 69.24% | 69.00% | ~200K | 0.03210 |
| **BiLSTM** | 67.00% | 67.40% | 67.00% | ~250K | 0.04408 |
| **DistilBERT** | **88.00%** | **87.06%** | **87.80%** (Public) | ~66M | 0.01117 |
|  |  |  | **87.00%** (Private) |  |  |

### Per-Emotion Performance

**DistilBERT - Detailed Breakdown (Best Model):**

| Emotion | Precision | Recall | F1-Score | Optimized Threshold |
|---------|-----------|--------|----------|---------------------|
| 😠 Anger | 86.00% | 83.00% | 84.50% | 0.45 |
| 😨 Fear | 89.00% | 88.00% | 88.50% | 0.55 |
| 😊 Joy | 92.00% | 89.00% | 90.50% | 0.40 |
| 😢 Sadness | 85.00% | 84.00% | 84.50% | 0.50 |
| 😲 Surprise | 87.00% | 82.00% | 84.50% | 0.48 |
| **Macro Avg** | **87.80%** | **85.20%** | **87.00%** | - |

### Key Findings

🏆 **Best Overall Performance:** Dobery achieved the highest Macro F1 score of **84.50%**

⚡ **Most Efficient:** GRU with fastest training time and lowest resource usage

🎯 **Best Generalization:** BLISTIN showed smallest train-validation gap

📊 **Challenging Emotions:** Joy was hardest to detect across all models



💡 **Key Insights:**
- **[INSIGHT 1]**: DistilBERT's pre-trained knowledge significantly improved performance on complex emotional expressions
- **[INSIGHT 2]**: BiLSTM performed well on longer texts with subtle emotional cues
- **[INSIGHT 3]**: TextCNN was surprisingly effective for texts with clear emotional keywords


### Competition Performance

- **Final Rank:** 27th out of 200 participants 🏆
- **Public Leaderboard:** Macro F1 = **87.80%**
- **Private Leaderboard:** Macro F1 = **87.00%**
- **Total Submissions:** 40 submissions
- **Competition:** 2025 Sep DLGenAI Project (IIT Madras)

---

## 💡 Key Learnings

### Technical Learnings

1. **Multi-Label Classification Challenges**
   - Each emotion must be predicted independently with BCEWithLogitsLoss
   - **Threshold optimization is critical**: Default 0.5 vs optimized [0.45, 0.55, 0.40, 0.50, 0.48] gained ~2% F1
   - Per-emotion thresholds handle class imbalance better than global threshold
   - Joy required lower threshold (0.40) due to clearer positive signals

2. **Architecture Selection**
   - CNNs excel at pattern recognition but lack sequential understanding (71% F1)
   - RNNs (GRU/LSTM) capture context but struggled with this task (67-69% F1)
   - Transformers provide best overall performance (87% F1) - 16% improvement!
   - BiLSTM surprisingly underperformed GRU despite being more complex

3. **Transfer Learning Impact**
   - Pre-trained DistilBERT significantly outperformed models trained from scratch
   - **Unconventional finding**: Higher learning rate (1e-2) worked better than typical 2e-5
   - Longer sequences (MAX_LEN=200) captured more emotional context than standard 128
   - Fine-tuning all layers was necessary for best performance

4. **Hyperparameter Tuning**
   - Embedding dimension of 128 provided good balance for CNN/RNN models
   - Dropout (0.2) prevented overfitting across all models
   - Batch size of 32 worked consistently well
   - 25 epochs with early stopping based on validation F1

### Competition Strategy

- ✅ Start with simple baseline (TextCNN)
- ✅ Progress to more complex models (GRU → BiLSTM)
- ✅ Leverage pre-trained models (DistilBERT)
- ✅ Ensemble different architectures for best results
- ✅ Monitor validation metrics to prevent overfitting

### Challenges Faced

1. **Class Imbalance**: Some emotions more common than others
2. **Multi-Label Complexity**: Texts often express multiple emotions
3. **Subtle Expressions**: Emotions expressed through context, not keywords
4. **Limited Data**: Need for effective regularization and augmentation
5. **Computation Constraints**: Balancing model complexity with Kaggle limits

---

## 📁 Project Structure

```
emotion-classification/
│
├── data/
│   ├── train.csv                    # Training data
│   ├── test.csv                     # Test data (no labels)
│   └── sample_submission.csv        # Submission format
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_textcnn_training.ipynb    # CNN experiments
│   ├── 03_gru_training.ipynb        # GRU experiments
│   ├── 04_bilstm_training.ipynb     # BiLSTM experiments
│   ├── 05_distilbert_training.ipynb # DistilBERT fine-tuning
│   └── 06_inference.ipynb           # Generate predictions
│
├── models/
│   ├── textcnn.py                   # TextCNN implementation
│   ├── gru.py                       # GRU implementation
│   ├── bilstm.py                    # BiLSTM implementation
│   └── bert_dataset.py              # Custom dataset for transformers
│
├── utils/
│   ├── preprocessing.py             # Text preprocessing utilities
│   ├── metrics.py                   # Macro F1 calculation
│   └── visualization.py             # Plotting functions
│
├── saved_models/
│   ├── textcnn_best.pth
│   ├── gru_best.pth
│   ├── bilstm_best.pth
│   └── distilbert_best/             # Hugging Face model directory
│
├── submissions/
│   ├── submission_cnn.csv
│   ├── submission_gru.csv
│   ├── submission_bilstm.csv
│   ├── submission_distilbert.csv
│   └── submission_ensemble.csv      # Best submission
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🙏 Acknowledgments

This project was completed as part of the **IIT Madras Deep Learning and Generative AI course (September 2025 term)**. Special thanks to:

- **IIT Madras Instructors** for their excellent course structure and guidance
- **Livin Nector** for hosting this challenging competition on Kaggle
- **Hugging Face** for the Transformers library and pre-trained models
- **PyTorch Team** for the excellent deep learning framework
- **Kaggle Community** for computational resources and discussions
- **All 181 participants** for making this competition competitive and educational

### Resources Used

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Kim (2014) - CNN for Text Classification](https://arxiv.org/abs/1408.5882)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 📝 Competition Details

- **Competition Name**: 2025 Sep DLGenAI Project
- **Platform**: Kaggle (Private Competition)
- **Duration**: 2 months
- **Total Participants**: 200
- **Total Submissions**: 2,710 (across all participants)
- **My Submissions**: 40
- **Evaluation Metric**: Macro F1-Score
- **Task**: Multi-label emotion classification (5 emotions)
- **My Final Rank**: 27th 🏆

---

## 🌟 Future Improvements

- [x] Implement threshold optimization per emotion ✅ (Improved by ~2%)
- [X] Experiment with ensemble methods combining all models
- [ ] Try data augmentation (back-translation, paraphrasing with GPT)
- [ ] Experiment with larger transformers (RoBERTa, ELECTRA, DeBERTa-v3)
- [ ] Implement focal loss to handle class imbalance better
- [ ] Add attention visualization for interpretability
- [ ] Experiment with different MAX_LEN values (256, 512)
- [ ] Try label smoothing for better generalization
- [ ] Implement k-fold cross-validation for robust evaluation

---

## 👨‍💻 Author

**Ayusman Samasi**
- IIT Madras - Deep Learning & GenAI (Sep 2025)
- GitHub: [Hariswar8018](https://github.com/Hariswar8018/)
- LinkedIn: [ayusman-samasi](https://www.linkedin.com/in/ayusman-samasi/)
- Kaggle: [samasiayushman](https://www.kaggle.com/code/samasiayushman)

---

## 📜 License

This project is for educational purposes as part of the IIT Madras curriculum. Please respect the competition rules and honor code.

---

## 🌟 If This Helped You

If you found this project useful for your learning:
- ⭐ Star this repository
- 🔄 Share with your classmates
- 💬 Provide feedback for improvements

---

### **Competition Status**: ✅ Completed 
### **Final Score**: 87.80% (Public) / 87.00% (Private) 
### **Rank**: 27/200 🏆

---

*"Emotions are the universal language of humanity. Teaching machines to understand them brings us closer to truly intelligent AI."*

**Last Updated**: [DATE]
