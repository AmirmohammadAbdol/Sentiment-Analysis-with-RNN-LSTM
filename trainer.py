import torch
from torch import nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train(True)
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for text, lengths, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
                text = text.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(text, lengths)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0            
            with torch.no_grad():
                for text, lengths, labels in tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                    text = text.to(self.device)
                    lengths = lengths.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(text, lengths)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = val_correct / val_total
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Step 8: Record performance and keep track of best validation accuracy
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f'  -> Saved best model with validation accuracy: {val_accuracy:.4f}')