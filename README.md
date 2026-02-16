# Transformer Neural Machine Translation: English to Italian

A complete implementation of the Transformer architecture for neural machine translation, built from scratch using PyTorch. This project translates English sentences to Italian using a model trained on the OPUS Books dataset.


**Training**: 2 epochs on NVIDIA T4 GPU (Google Colab) in ~40 minutes 
**Dataset**: OPUS Books English-Italian (~127K sentence pairs) (90/10 train-test split)
**Performance**: Training Loss ~5.8, Validation Loss ~6.2 after 2 epochs

## Tech Stack

- **PyTorch** - Deep learning framework
- **Hugging Face** - Dataset loading
- **Tokenizers** - Word-level tokenization
- **TensorBoard** - Training visualization

### Key Features

- Multi-head self-attention with 8 attention heads
- Real-time training monitoring with TensorBoard
- Interactive CLI for translation testing
- Checkpoint for resumable training

## Architecture

- **Type**: Encoder-Decoder Transformer
- **Model Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Attention Heads**: 8
- **Layers**: 6 Encoder + 6 Decoder
- **Parameters**: ~60 Million
- **Sequence Length**: 320 tokens


## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Transformer-From-Scratch.git
cd Transformer-From-Scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm tensorboard
```

## Usage

### Training
Start training (automatically downloads dataset):
```bash
python train.py
```

**Note**: 
- Training requires a GPU for reasonable performance (T4 GPU recommended)
- First run will download the OPUS Books dataset (~500MB)
- Checkpoints are saved in `weights/` folder after each epoch
- Adjust `batch_size` in `config.py` if you encounter out-of-memory errors

### Inference
Translate sentences interactively:
```bash
python inference.py
```

### Monitoring
View training progress and sample translations:
```bash
tensorboard --logdir=runs
```


## Results

Sample translations after 2 epochs:

| English Input | Italian Output |
|---------------|----------------|
| Hello, how are you? | Ciao, come stai? |
| I love machine learning. | Amo imparare le macchine. |
| The book is on the table. | Il libro è sulla tavola. |

---

## Project Structure

```
├── model.py       # Transformer architecture (Attention, Encoder, Decoder)
├── dataset.py     # Data loading and preprocessing
├── train.py       # Training loop and validation
├── inference.py   # Interactive translation script
└── config.py      # Hyperparameters and configuration
```

---

## License

MIT License

---

<div align="center">
  <b>Built with PyTorch</b>
</div>
