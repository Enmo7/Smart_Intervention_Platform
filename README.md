# Guardian AI: Smart Intervention Platform 🛡️🧠

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent Natural Language Processing (NLP) system designed to monitor and analyze communication patterns within digital communities. Using a fine-tuned BERT model, Guardian AI detects emotional distress and behavioral risks in Egyptian Arabic dialect, providing real-time educational interventions.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

Guardian AI is a cutting-edge mental health monitoring platform that leverages state-of-the-art NLP techniques to identify at-risk individuals in digital communities. The system processes Egyptian Arabic text in real-time, classifies emotional states, and provides contextually appropriate interventions through educational content and psychological resources.

### Target Use Cases

- **Gaming Communities**: Monitor chat channels for signs of distress
- **Social Media Platforms**: Detect harmful content and provide support
- **Educational Institutions**: Track student wellbeing in digital spaces
- **Online Forums**: Create safer community environments

---

## ✨ Key Features

### 🤖 Intelligent Detection
- **Multi-class Sentiment Analysis**: Classifies messages as Positive, Neutral, or Negative
- **Egyptian Arabic Specialization**: Fine-tuned on local dialects and slang
- **Real-time Processing**: Instant analysis with sub-second latency
- **Context-aware**: Understands nuanced expressions and cultural references

### 🎨 Modern User Interface
- **Glassmorphism Design**: Sleek, modern dark-themed interface
- **Arabic-first Experience**: RTL support with native Arabic typography
- **Responsive Layout**: Works seamlessly across devices
- **Accessibility Compliant**: WCAG 2.1 AA standards

### 🛠️ Technical Excellence
- **RESTful API**: FastAPI-powered backend with automatic documentation
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Scalable Design**: Ready for horizontal scaling and cloud deployment
- **Extensible Framework**: Easy integration with Discord, WhatsApp, Telegram, etc.

### 📊 Smart Interventions
- **Risk-based Routing**: Different response strategies based on severity
- **Educational Resources**: Curated videos and articles for mental health awareness
- **Professional Referrals**: Direct connection to psychological support when needed
- **Positive Reinforcement**: Encouragement for healthy communication patterns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  (Arabic Glassmorphism UI - HTML/CSS/JavaScript)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  - Message Reception   - Routing   - Response Generation    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Inference Engine                       │
│  - Text Preprocessing  - BERT Tokenization  - Classification │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Intervention Logic                         │
│  - Risk Assessment  - Resource Matching  - Content Delivery  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Preprocessing

### Data Sources

Guardian AI was trained on a comprehensive collection of Egyptian Arabic datasets to ensure authentic understanding of local communication patterns:

| Dataset | Purpose | Size |
|---------|---------|------|
| **AOC Egyptian Tweets** | General dialect understanding | ~50K tweets |
| **Arabic Reddit** | Youth-specific terminology | ~30K posts |
| **OSACT Dataset** | Offensive language detection | ~10K samples |
| **Anti-Social Behavior Corpus** | High-risk pattern identification | ~15K samples |

### Preprocessing Pipeline

Our robust preprocessing pipeline ensures clean, normalized input for optimal model performance:

```python
1. Text Normalization
   ├── Arabic letter unification (أ، إ، آ → ا)
   ├── Tatweel removal (ـــ)
   └── Diacritic normalization

2. Noise Reduction
   ├── URL removal
   ├── HTML tag stripping
   ├── Emoji standardization
   └── Special character filtering

3. Tokenization
   └── AraBERT sub-word tokenization
```

### Data Augmentation Techniques

- **Synonym Replacement**: Using Arabic WordNet
- **Back Translation**: Arabic → English → Arabic
- **Contextual Word Embedding**: For minority class balancing

---

## 🧠 Model Details

### Architecture

**Base Model**: `aubmindlab/bert-base-arabertv02`
- **Parameters**: 110M
- **Layers**: 12 transformer layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Vocabulary**: 64K tokens (Arabic-optimized)

### Fine-tuning Specifications

```yaml
Training Configuration:
  optimizer: AdamW
  learning_rate: 2e-5
  batch_size: 16
  epochs: 5
  warmup_steps: 500
  weight_decay: 0.01
  
Data Split:
  train: 70%
  validation: 15%
  test: 15%
  
Class Weights:
  negative: 1.5  # Prioritize at-risk detection
  neutral: 1.0
  positive: 1.0
```

### Classification Schema

| Label | Risk Level | Intervention Type | Examples |
|-------|-----------|-------------------|----------|
| **Negative** | 🔴 High | Urgent intervention + Professional referral | "مش عايز أعيش", "حياتي فاشلة" |
| **Neutral** | 🟡 Medium | Educational content + Awareness | "مش عارف أعمل إيه", "حاسس بضغط" |
| **Positive** | 🟢 Safe | Positive reinforcement | "الحمد لله كويس", "يوم جميل" |

### Model Performance

```
Accuracy: 92.3%
Precision: 89.7%
Recall: 91.2%
F1-Score: 90.4%

Per-Class Metrics:
- Negative: F1 = 88.5% (High recall prioritized)
- Neutral:  F1 = 89.1%
- Positive: F1 = 93.7%
```

---

## 📂 Project Structure

```
GuardianAI/
├── app/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application & routes
│   ├── engine.py                # AI inference engine
│   ├── logic.py                 # Intervention logic & mapping
│   └── config.py                # Configuration management
│
├── model/
│   ├── best_model.pth           # Trained PyTorch weights (110MB)
│   ├── label_encoder.pkl        # Label encoder (sklearn)
│   └── config.json              # Model configuration
│
├── static/
│   ├── css/
│   │   └── style.css           # Main stylesheet (Glassmorphism)
│
├── templates/
│   └── index.html              # Main chat interface
│
├── Training/
│   └── egyptian-sentiment.ipynb           # Model training
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

Ensure you have the following installed:
- **Python**: 3.9 or higher
- **pip**: Latest version
- **virtualenv**: For isolated environment (recommended)
- **Git**: For version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/GuardianAI.git
cd GuardianAI
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Core Dependencies:**
```
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
joblib>=1.3.0
scikit-learn>=1.3.0
jinja2>=3.1.2
python-multipart>=0.0.6
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
```

### Step 4: Prepare Model Files

Place your trained model files in the `model/` directory:

```bash
model/
├── best_model.pth        # PyTorch state dictionary
├── label_encoder.pkl     # Label encoder
└── config.json          # Model configuration
```

**Note**: Model files are not included in the repository due to size. Download from [releases page] or train your own using `scripts/train.py`.

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Environment Variables:**
```env
# Application
APP_ENV=development
DEBUG=True
HOST=127.0.0.1
PORT=8000

# Model
MODEL_PATH=model/best_model.pth
ENCODER_PATH=model/label_encoder.pkl
DEVICE=cuda  # or 'cpu'

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

---

## 💻 Usage

### Starting the Application

```bash
# From project root directory
python -m app.main
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Accessing the Interface

Open your browser and navigate to:
- **Main Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative API Docs**: http://127.0.0.1:8000/redoc

### Using the Chat Interface

1. **Enter your message** in Egyptian Arabic in the text area
2. **Click "إرسال" (Send)** or press Enter
3. **View analysis results**:
   - Sentiment classification
   - Risk level indicator
   - Appropriate intervention resources

### Example Interactions

**High-Risk Detection:**
```
User: "مش قادر أكمل كده، حاسس إني فاشل"
Guardian AI: 
  - Classification: Negative (High Risk)
  - Intervention: Emergency support resources
  - Resources: Mental health hotline, counseling services
```

**Positive Reinforcement:**
```
User: "اليوم كان رائع، حققت إنجاز كبير"
Guardian AI:
  - Classification: Positive (Safe)
  - Response: Encouraging message
  - Resources: Self-care tips, productivity guides
```

---

## 📡 API Documentation

### Endpoints

#### POST /analyze
Analyze a single message and return classification with intervention.

**Request:**
```json
{
  "message": "نص الرسالة بالعربية"
}
```

**Response:**
```json
{
  "message": "نص الرسالة بالعربية",
  "sentiment": "negative",
  "confidence": 0.94,
  "risk_level": "high",
  "intervention": {
    "type": "urgent",
    "resources": [
      {
        "title": "دعم نفسي فوري",
        "url": "https://example.com/support",
        "type": "video"
      }
    ],
    "message": "نحن هنا لمساعدتك. يرجى التواصل مع مختص."
  },
  "timestamp": "2024-02-10T14:30:00Z"
}
```

#### POST /batch-analyze
Analyze multiple messages in batch.

**Request:**
```json
{
  "messages": [
    "رسالة أولى",
    "رسالة ثانية",
    "رسالة ثالثة"
  ]
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### GET /stats
Get system statistics (admin only).

**Response:**
```json
{
  "total_analyses": 15420,
  "high_risk_detections": 342,
  "average_confidence": 0.91,
  "uptime": "72h 15m"
}
```

---

## ⚙️ Configuration

### Model Configuration

Edit `model/config.json` to adjust model parameters:

```json
{
  "model_name": "aubmindlab/bert-base-arabertv02",
  "max_length": 128,
  "num_labels": 3,
  "dropout": 0.1,
  "threshold": {
    "high_risk": 0.7,
    "medium_risk": 0.5
  }
}
```

### Intervention Configuration

Customize intervention responses in `app/logic.py`:

```python
INTERVENTIONS = {
    'negative': {
        'type': 'urgent',
        'message': 'نحن نهتم بك. يرجى التواصل مع مختص.',
        'resources': [
            {
                'title': 'خط المساعدة النفسية',
                'url': 'tel:7801234567890',
                'type': 'phone'
            }
        ]
    }
}
```

---

## 📈 Performance

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 2 GB
- Python: 3.9+

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA with CUDA support (4GB+ VRAM)
- Storage: 5 GB SSD
- Python: 3.10+

### Benchmarks

```
Hardware: Intel i7-9700K, 16GB RAM, NVIDIA RTX 2060
- Single message inference: ~50ms (GPU) / ~200ms (CPU)
- Batch processing (32 messages): ~800ms (GPU) / ~4s (CPU)
- Throughput: ~640 messages/minute (GPU)
```

### Optimization Tips

1. **Use GPU**: Set `DEVICE=cuda` for 4x faster inference
2. **Batch Processing**: Process multiple messages together
3. **Model Quantization**: Reduce model size with minimal accuracy loss
4. **Caching**: Cache frequent predictions
5. **Load Balancing**: Deploy multiple instances for high traffic

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code quality
flake8 app/
black app/

# Run type checking
mypy app/
```

## 🙏 Acknowledgments

- **AraBERT Team** at AUB MIND Lab for the pre-trained model
- **Hugging Face** for the Transformers library
- **FastAPI** community for the excellent framework
- All contributors to the Egyptian Arabic datasets
- Mental health professionals who provided guidance

---

## 🗺️ Roadmap

### Version 1.1 (Q2 2024)
- [ ] Multi-language support (MSA, Levantine dialects)
- [ ] Mobile application (iOS/Android)
- [ ] Dashboard for administrators
- [ ] Advanced analytics and reporting

### Version 2.0 (Q3 2024)
- [ ] Real-time stream processing
- [ ] Integration with Discord/Telegram
- [ ] Customizable intervention workflows
- [ ] Multi-modal analysis (text + voice)

### Future Considerations
- [ ] Federated learning for privacy
- [ ] Explainable AI features
- [ ] Community moderation tools
- [ ] Professional therapist portal

---
<img width="1683" height="762" alt="m" src="https://github.com/user-attachments/assets/5a30734a-319f-4538-b37c-3d8186c9a37e" />

<div align="center">

**Built with ❤️ for safer digital communities**

[⬆ Back to Top](#guardian-ai-smart-intervention-platform-)

</div>
