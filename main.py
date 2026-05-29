import torch
from data import get_data_loaders_and_vocab
from model import SentimentRNN
from trainer import Trainer
import matplotlib.pyplot as plt


def train_and_evaluate(cell_type, data_dir, batch_size=64, embed_size=100, hidden_size=128, num_layers=2, dropout=0.5, num_epochs=10):
    print(f"\n{'='*50}")
    print(f"Training {cell_type.upper()} model")
    print(f"{'='*50}")
    train_loader, val_loader, vocab = get_data_loaders_and_vocab(data_dir, batch_size=batch_size, vocab_size=20000, min_freq=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SentimentRNN(
        vocab_size=len(vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=vocab.pad_idx,
        cell_type=cell_type).to(device)
    
    trainer = Trainer(model, train_loader, val_loader, device, lr = 0.001)
    trainer.train(num_epochs)
    
    return trainer

def plot_results(rnn_trainer, lstm_trainer):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(rnn_trainer.train_losses, label='RNN', marker='o', color='blue')
    axes[0, 0].plot(lstm_trainer.train_losses, label='LSTM', marker='s', color='green')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss: RNN vs LSTM')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # Plot Validation Loss
    axes[0, 1].plot(rnn_trainer.val_losses, label='RNN', marker='o', color='blue')
    axes[0, 1].plot(lstm_trainer.val_losses, label='LSTM', marker='s', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss: RNN vs LSTM')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # Plot Training Accuracy
    axes[1, 0].plot(rnn_trainer.train_accuracies, label='RNN', marker='o', color='blue')
    axes[1, 0].plot(lstm_trainer.train_accuracies, label='LSTM', marker='s', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy: RNN vs LSTM')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # Plot Validation Accuracy
    axes[1, 1].plot(rnn_trainer.val_accuracies, label='RNN', marker='o', color='blue')
    axes[1, 1].plot(lstm_trainer.val_accuracies, label='LSTM', marker='s', color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy: RNN vs LSTM')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig('rnn_vs_lstm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results(rnn_trainer, lstm_trainer):
    print("\n" + "="*60)
    print("FINAL RESULTS: RNN vs LSTM")
    print("="*60)
    print("\n--- RNN Results ---")
    print(f"Final Training Loss: {rnn_trainer.train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {rnn_trainer.train_accuracies[-1]:.4f}")
    print(f"Final Validation Loss: {rnn_trainer.val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {rnn_trainer.val_accuracies[-1]:.4f}")
    print(f"Best Validation Accuracy: {rnn_trainer.best_val_accuracy:.4f}")
    
    print("\n--- LSTM Results ---")
    print(f"Final Training Loss: {lstm_trainer.train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {lstm_trainer.train_accuracies[-1]:.4f}")
    print(f"Final Validation Loss: {lstm_trainer.val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {lstm_trainer.val_accuracies[-1]:.4f}")
    print(f"Best Validation Accuracy: {lstm_trainer.best_val_accuracy:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if rnn_trainer.best_val_accuracy > lstm_trainer.best_val_accuracy:
        print(f"🏆 WINNER: RNN with {rnn_trainer.best_val_accuracy:.4f} validation accuracy")
        print(f"   LSTM achieved: {lstm_trainer.best_val_accuracy:.4f}")
        print(f"   Difference: +{rnn_trainer.best_val_accuracy - lstm_trainer.best_val_accuracy:.4f}")
    elif lstm_trainer.best_val_accuracy > rnn_trainer.best_val_accuracy:
        print(f"🏆 WINNER: LSTM with {lstm_trainer.best_val_accuracy:.4f} validation accuracy")
        print(f"   RNN achieved: {rnn_trainer.best_val_accuracy:.4f}")
        print(f"   Difference: +{lstm_trainer.best_val_accuracy - rnn_trainer.best_val_accuracy:.4f}")
    else:
        print(f"🤝 TIE: Both models achieved {rnn_trainer.best_val_accuracy:.4f} validation accuracy")

def main():
    DATA_DIR = "aclImdb"
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    
    rnn_trainer = train_and_evaluate(
        cell_type='rnn',
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS)
    
    lstm_trainer = train_and_evaluate(
        cell_type='lstm',
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS)
    
    print_results(rnn_trainer, lstm_trainer)
    plot_results(rnn_trainer, lstm_trainer)

if __name__ == "__main__":
    main()