# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from neuralnetworks.convlstm import ConvLSTM



# Training function
def train_model(args):
    # Load data
    X = torch.load(os.path.join(args.data_dir, 'X.pt'))
    y = torch.load(os.path.join(args.data_dir, 'y.pt'))

    # Split data into train and test
    train_size = int(0.9 * len(X))
    test_size = len(X) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        TensorDataset(X, y), [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, loss function, optimizer
    model = ConvLSTM(input_dim=3, hidden_dim=64, kernel_size=(3, 3), num_layers=1).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            output, _ = model(X_batch)
            loss = criterion(output[:, -1, :, :, :], y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)

    # SageMaker specific parameters
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train_model(args)
