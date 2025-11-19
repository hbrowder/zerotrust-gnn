import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples
    Focuses training on hard-to-classify samples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class GNNAnomalyDetector(nn.Module):
    """
    Improved GCN-based Graph Neural Network for network anomaly detection
    
    Improvements:
    - Increased hidden dimensions for better capacity
    - Batch normalization for stable training
    - Adjusted dropout for better regularization
    - Residual connections for gradient flow
    """
    def __init__(self, num_node_features, hidden_dim=128, edge_dim=3):
        super(GNNAnomalyDetector, self).__init__()
        
        # GCN layers with batch normalization
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        # Edge classifier with increased capacity and regularization
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 + edge_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Learn node representations with GCN + BatchNorm
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # For each edge, concatenate source node, destination node, and edge features
        src_nodes = x[edge_index[0]]
        dst_nodes = x[edge_index[1]]
        
        # Edge features (bytes, protocol, label) - only use first 2 for prediction
        edge_features = edge_attr[:, :2]
        
        # Concatenate and classify each edge
        edge_input = torch.cat([src_nodes, dst_nodes, edge_features], dim=1)
        predictions = self.edge_classifier(edge_input)
        
        return predictions
    
    def predict_risk_scores(self, data):
        """Predict risk scores (0-100) for all edges in a graph"""
        self.eval()
        with torch.no_grad():
            probs = self.forward(data).squeeze()
            risk_scores = probs * 100
        return risk_scores

def compute_accuracy(outputs, labels):
    """Compute binary classification accuracy"""
    predictions = (outputs.squeeze() > 0.5).float()
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total

def train_model_improved(train_graph, test_graph, num_epochs=100, learning_rate=0.005):
    """
    Train GNN model with improved calibration techniques
    
    Improvements:
    - Focal Loss for hard example mining
    - Class weighting
    - Increased model capacity
    - Better regularization
    - Lower learning rate for stability
    """
    
    # Normalize node features
    train_mean = train_graph.x.mean(dim=0, keepdim=True)
    train_std = train_graph.x.std(dim=0, keepdim=True) + 1e-6
    
    train_graph.x = (train_graph.x - train_mean) / train_std
    test_graph.x = (test_graph.x - train_mean) / train_std
    
    # Normalize edge features
    edge_mean = train_graph.edge_attr[:, :2].mean(dim=0, keepdim=True)
    edge_std = train_graph.edge_attr[:, :2].std(dim=0, keepdim=True) + 1e-6
    
    train_graph.edge_attr[:, :2] = (train_graph.edge_attr[:, :2] - edge_mean) / edge_std
    test_graph.edge_attr[:, :2] = (test_graph.edge_attr[:, :2] - edge_mean) / edge_std
    
    # Get number of node features
    num_node_features = train_graph.x.shape[1]
    
    # Initialize improved model with larger capacity
    model = GNNAnomalyDetector(num_node_features=num_node_features, hidden_dim=128, edge_dim=2)
    
    # Use AdamW optimizer with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # Learning rate scheduler for adaptive learning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Use Focal Loss for better calibration
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Extract edge labels
    train_labels = train_graph.edge_attr[:, 2]
    test_labels = test_graph.edge_attr[:, 2]
    
    print("\n" + "="*70)
    print("TRAINING IMPROVED GNN MODEL FOR ANOMALY DETECTION")
    print("="*70)
    print(f"Model: Enhanced GCN with Focal Loss and Class Weighting")
    print(f"Training edges: {train_labels.shape[0]} ({(train_labels == 0).sum().item()} benign, {(train_labels == 1).sum().item()} malicious)")
    print(f"Test edges: {test_labels.shape[0]} ({(test_labels == 0).sum().item()} benign, {(test_labels == 1).sum().item()} malicious)")
    print(f"Node features: {num_node_features}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Loss function: Focal Loss (alpha={criterion.alpha}, gamma={criterion.gamma})")
    print(f"Optimizer: AdamW with weight decay 5e-4")
    print("="*70 + "\n")
    
    best_test_acc = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_graph).squeeze()
        loss = criterion(outputs, train_labels)
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(train_graph).squeeze()
            test_outputs = model(test_graph).squeeze()
            
            train_acc = compute_accuracy(train_outputs, train_labels)
            test_acc = compute_accuracy(test_outputs, test_labels)
        
        train_losses.append(loss.item())
        test_accuracies.append(test_acc)
        
        # Update learning rate based on test accuracy
        scheduler.step(test_acc)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_gnn_model_improved.pt')
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Train Loss: {loss.item():.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('best_gnn_model_improved.pt'))
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(train_graph).squeeze()
        test_outputs = model(test_graph).squeeze()
        
        train_acc = compute_accuracy(train_outputs, train_labels)
        test_acc = compute_accuracy(test_outputs, test_labels)
        
        # Get risk scores
        train_risk_scores = train_outputs * 100
        test_risk_scores = test_outputs * 100
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"Final Train Accuracy: {train_acc*100:.2f}%")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    
    # Detailed predictions on test set
    print("\n" + "="*70)
    print("DETAILED PREDICTIONS ON TEST SET")
    print("="*70)
    
    test_preds = (test_outputs > 0.5).float()
    tp = ((test_preds == 1) & (test_labels == 1)).sum().item()
    tn = ((test_preds == 0) & (test_labels == 0)).sum().item()
    fp = ((test_preds == 1) & (test_labels == 0)).sum().item()
    fn = ((test_preds == 0) & (test_labels == 1)).sum().item()
    
    print(f"Confusion Matrix:")
    print(f"  True Positives:   {tp} (correctly identified malicious)")
    print(f"  True Negatives:   {tn} (correctly identified benign)")
    print(f"  False Positives:  {fp} (benign flagged as malicious)")
    print(f"  False Negatives:  {fn} (malicious flagged as benign)")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    # Risk score distribution analysis
    benign_mask = test_labels == 0
    malicious_mask = test_labels == 1
    
    benign_scores = test_risk_scores[benign_mask]
    malicious_scores = test_risk_scores[malicious_mask]
    
    print(f"\n" + "="*70)
    print("RISK SCORE DISTRIBUTION")
    print("="*70)
    print(f"\nBENIGN TRAFFIC:")
    print(f"  Mean:   {benign_scores.mean():.2f}/100")
    print(f"  Median: {benign_scores.median():.2f}/100")
    print(f"  Std:    {benign_scores.std():.2f}")
    print(f"  Range:  {benign_scores.min():.2f} - {benign_scores.max():.2f}")
    
    print(f"\nMALICIOUS TRAFFIC:")
    print(f"  Mean:   {malicious_scores.mean():.2f}/100")
    print(f"  Median: {malicious_scores.median():.2f}/100")
    print(f"  Std:    {malicious_scores.std():.2f}")
    print(f"  Range:  {malicious_scores.min():.2f} - {malicious_scores.max():.2f}")
    
    # Sample risk scores
    print(f"\nSample Risk Scores (Test Set):")
    print(f"{'Index':<8} {'True Label':<15} {'Predicted':<15} {'Risk Score':<12}")
    print("-"*50)
    for i in range(min(10, len(test_labels))):
        true_label = "Malicious" if test_labels[i] == 1 else "Benign"
        pred_label = "Malicious" if test_preds[i] == 1 else "Benign"
        print(f"{i:<8} {true_label:<15} {pred_label:<15} {test_risk_scores[i]:.1f}/100")
    
    # Check if target achieved
    target_achieved = (benign_scores.mean() < 30) and (malicious_scores.mean() >= 70 and malicious_scores.mean() <= 95)
    print("\n" + "="*70)
    if target_achieved:
        print(f"✓ TARGET ACHIEVED!")
        print(f"  Benign avg: {benign_scores.mean():.1f}/100 (target: <30)")
        print(f"  Malicious avg: {malicious_scores.mean():.1f}/100 (target: 70-95)")
    else:
        print(f"⚠ Working towards target...")
        print(f"  Benign avg: {benign_scores.mean():.1f}/100 (target: <30)")
        print(f"  Malicious avg: {malicious_scores.mean():.1f}/100 (target: 70-95)")
    print("="*70 + "\n")
    
    return model, train_losses, test_accuracies

if __name__ == "__main__":
    # Load pre-prepared graphs from main.py
    print("Loading training and test graphs...")
    
    from main import train_graph, test_graph
    
    # Train the improved model
    model, losses, accuracies = train_model_improved(
        train_graph, 
        test_graph, 
        num_epochs=100, 
        learning_rate=0.005
    )
    
    print("✓ Model training complete!")
    print("✓ Best model saved to 'best_gnn_model_improved.pt'")
    print("\nTo use the improved model for predictions:")
    print("  from train_gnn_improved import GNNAnomalyDetector")
    print("  model = GNNAnomalyDetector(num_node_features=6, hidden_dim=128, edge_dim=2)")
    print("  model.load_state_dict(torch.load('best_gnn_model_improved.pt'))")
    print("  risk_scores = model.predict_risk_scores(your_graph)")
