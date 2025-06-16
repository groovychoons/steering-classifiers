import torch
import numpy as np

from torch import nn
from sklearn.model_selection import train_test_split
from dialz import Dataset


class LinearClassifier(nn.Module):
    def __init__(self, max_token_length):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(max_token_length*5, 100, dtype=torch.float64),
            #nn.ReLU(),
            nn.Linear(100, 2, dtype=torch.float64),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probs = torch.sigmoid(logits)
        return probs

class TransformerClassifier(nn.Module):
    def __init__(self, max_token_length):
        super().__init__()
        self.transformer_cls = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=max_token_length, nhead=2, dtype=torch.float64),
            #nn.ReLU(),
            nn.Linear(max_token_length, 2, dtype=torch.float64))

    def forward(self, x):
        x = x.mean(dim=1)  # Assuming x is of shape (batch_size, seq_length, features)
        logits = self.transformer_cls(x)
        probs = torch.sigmoid(logits)
        return probs
    

class CustomCAADataset(Dataset):
    def __init__(self, activation_score, labels, text):
        self.activation_score = activation_score
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        activation_score = self.activation_score[idx]
        labels = self.labels[idx]
        return activation_score, labels
    

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


def construct_dataset(df, padded_activation_scores, test_size=0.3, random_state=42):
    labels = df['label'].to_numpy(dtype=float)
    # Create the one-hot encoded matrix
    categories, inverse = np.unique(labels, return_inverse=True)
    one_hot_labels = np.zeros((labels.size, categories.size))
    one_hot_labels[np.arange(labels.size), inverse] = 1

    # split data
    X_train, X_test, text_train, text_test, y_train, y_test = train_test_split(
        padded_activation_scores, df.text.to_list(), one_hot_labels, test_size=test_size, random_state=random_state)
    
    # construct datasets with steering vector features
    train_dataset = CustomCAADataset(X_train, y_train, text_train)
    test_dataset = CustomCAADataset(X_test, y_test, text_test)

    return train_dataset, test_dataset


def optimize(train_dataset, test_dataset, max_token_length, learning_rate=1e-3, batch_size=64, epochs=50, is_transformer=False):
    if not is_transformer:
        model = LinearClassifier(max_token_length)
    else:
        model = TransformerClassifier(max_token_length)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    accs = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        acc = test_loop(test_dataloader, model, loss_fn)
        accs.append(acc)
    print("Done!")
    return max(accs)
