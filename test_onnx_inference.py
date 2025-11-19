import onnxruntime as ort
import numpy as np
import torch
from train_gnn import GNNAnomalyDetector
from torch_geometric.data import Data

def test_onnx_inference():
    """
    Test ONNX model inference and compare with PyTorch model
    """
    print("="*70)
    print("TESTING ONNX MODEL INFERENCE")
    print("="*70)
    
    # Create dummy test graph
    num_nodes = 30
    num_edges = 50
    
    print(f"\n1. Creating test graph ({num_nodes} nodes, {num_edges} edges)...")
    
    # Node features (normalized)
    node_features = np.random.randn(num_nodes, 6).astype(np.float32)
    
    # Edge indices (random connections)
    src_nodes = np.random.randint(0, num_nodes, num_edges, dtype=np.int64)
    dst_nodes = np.random.randint(0, num_nodes, num_edges, dtype=np.int64)
    edge_index = np.stack([src_nodes, dst_nodes], axis=0)
    
    # Edge attributes (bytes and protocol only)
    edge_attr = np.random.randn(num_edges, 2).astype(np.float32)
    
    print(f"   ✓ Test graph created")
    print(f"   - Node features: {node_features.shape}")
    print(f"   - Edge index: {edge_index.shape}")
    print(f"   - Edge attributes: {edge_attr.shape}")
    
    # Test with ONNX Runtime
    print("\n2. Running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession('gnn_model.onnx')
    
    onnx_inputs = {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attributes': edge_attr
    }
    
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_probs = onnx_outputs[0]
    onnx_risk_scores = onnx_probs * 100  # Convert to 0-100 scale
    
    print(f"   ✓ ONNX inference successful")
    print(f"   - Output shape: {onnx_probs.shape}")
    print(f"   - Anomaly probabilities range: [{onnx_probs.min():.4f}, {onnx_probs.max():.4f}]")
    print(f"   - Risk scores range: [{onnx_risk_scores.min():.2f}, {onnx_risk_scores.max():.2f}]")
    
    # Test with PyTorch model for comparison
    print("\n3. Running inference with PyTorch model (for comparison)...")
    pytorch_model = GNNAnomalyDetector(num_node_features=6, hidden_dim=128, edge_dim=2)
    pytorch_model.load_state_dict(torch.load('best_gnn_model.pt'))
    pytorch_model.eval()
    
    # Create PyG Data object
    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr)
    )
    
    with torch.no_grad():
        pytorch_probs = pytorch_model(data).numpy()
    
    pytorch_risk_scores = pytorch_probs * 100
    
    print(f"   ✓ PyTorch inference successful")
    print(f"   - Output shape: {pytorch_probs.shape}")
    print(f"   - Anomaly probabilities range: [{pytorch_probs.min():.4f}, {pytorch_probs.max():.4f}]")
    print(f"   - Risk scores range: [{pytorch_risk_scores.min():.2f}, {pytorch_risk_scores.max():.2f}]")
    
    # Compare outputs
    print("\n4. Comparing ONNX vs PyTorch outputs...")
    diff = np.abs(onnx_probs - pytorch_probs)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   - Maximum difference: {max_diff:.6f}")
    print(f"   - Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print("   ✓ ONNX and PyTorch models match closely (difference < 0.0001)")
    else:
        print(f"   ⚠ Some differences detected (max diff: {max_diff:.6f})")
    
    # Display sample predictions
    print("\n5. Sample predictions (first 10 edges):")
    print("-" * 70)
    print(f"{'Edge':<8} {'ONNX Risk':<15} {'PyTorch Risk':<15} {'Difference':<15}")
    print("-" * 70)
    for i in range(min(10, num_edges)):
        print(f"{i:<8} {onnx_risk_scores[i][0]:<15.2f} {pytorch_risk_scores[i][0]:<15.2f} {abs(onnx_risk_scores[i][0] - pytorch_risk_scores[i][0]):<15.4f}")
    print("-" * 70)
    
    print("\n" + "="*70)
    print("✓ ONNX MODEL TEST COMPLETE")
    print("="*70)
    print("\nThe ONNX model is ready for deployment!")
    print("Use 'gnn_model.onnx' for production inference with ONNX Runtime.")
    print("="*70)

if __name__ == "__main__":
    test_onnx_inference()
