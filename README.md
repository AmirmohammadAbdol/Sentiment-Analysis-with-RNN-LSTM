# Sentiment Analysis with RNN and LSTM

A deep learning project for binary sentiment classification of IMDb movie reviews using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. This project compares the performance of vanilla RNN vs LSTM on sentiment analysis tasks.

## 📊 Model Performance

| Model | Validation Accuracy |
|-------|---------------------|
| RNN   | 80%                 |
| LSTM  | 86%                 |
| **Improvement** | **+6% with LSTM** |

## 🗂️ Project Structure
- **data.py**  Data loading, preprocessing, vocabulary building, and DataLoaders
- **model.py**  RNN/LSTM model architecture with configurable cell types
- **trainer.py**  Training loop with progress bars, loss tracking, and model checkpointing
- **main.py**  Train both models, compare results, and generate visualizations
- **best_model.pt**  Saved best model checkpoint (generated after training)
- **rnn_vs_lstm_comparison.png**  Comparision between RNN and LSTM models (generated after training)
- **aclimdb**  Download aclImdb_v1.tar.gz from [here](https://ai.stanford.edu/~amaas/data/sentiment/) and extract the file in the same directory as the project .py files.


## ✨ Features

- Complete IMDb dataset preprocessing and tokenization
- Custom vocabulary builder with frequency filtering
- Configurable RNN/LSTM architecture (embedding size, hidden size, layers, dropout)
- GPU support with proper device management
- Training progress bars with tqdm
- Automatic best model saving based on validation accuracy
- Comprehensive loss/accuracy plotting for model comparison
- Gradient clipping for stable training

## 🛠️ Technologies

- **PyTorch** - Deep learning framework
- **Matplotlib** - Visualization and plotting
- **tqdm** - Progress bars

## 📈 Results Visualization

The project generates side-by-side comparison plots showing:
- Training Loss (RNN vs LSTM)
- Validation Loss (RNN vs LSTM)
- Training Accuracy (RNN vs LSTM)
- Validation Accuracy (RNN vs LSTM)

## 🚀 Installation and Usage

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/AmirmohammadAbdol/Sentiment-Analysis-RNN-LSTM.git
cd Sentiment-Analysis-RNN-LSTM

# Install dependencies
pip install torch matplotlib tqdm

Download the IMDb dataset from [here](https://ai.stanford.edu/~amaas/data/sentiment/)
Extract the dataset in the same folder as project

Run python main.py