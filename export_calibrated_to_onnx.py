import torch
import torch.onnx
from torch_geometric.data import Data
from train_gnn_calibrated import GNNAnomalyDetector
import numpy as np

class GNNONNXWrapper(torch.nn.Module):
    """Wrapper for calibrated GNN model to make it ONNX-compatible"""
    def __init__(self, gnn_model):
        super(GNNONNXWrapper, self).__init__()
        self.gnn_model = gnn_model
    
    def forward(self, x, edge_index, edge_attr):
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        predictions = self.gnn_model(data)
        return predictions

def export_calibrated_gnn_to_onnx(
    model_path='best_gnn_model_calibrated.pt',
    onnx_path='gnn_model_calibrated.onnx',
    num_nodes=50,
    num_edges=100
):
    """Export calibrated GNN model to ONNX format"""
    
    print("="*70)
    print("EXPORTING CALIBRATED GNN MODEL TO ONNX FORMAT")
    print("="*70)
    
    # Load trained model with temperature parameter
    print(f"\n1. Loading calibrated model from '{model_path}'...")
    model = GNNAnomalyDetector(num_node_features=6, hidden_dim=128, edge_dim=2, temperature=1.5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("   ✓ Calibrated model loaded successfully")
    
    # Wrap model for ONNX compatibility
    print("\n2. Wrapping model for ONNX compatibility...")
    wrapped_model = GNNONNXWrapper(model)
    wrapped_model.eval()
    print("   ✓ Model wrapped")
    
    # Create dummy inputs
    print(f"\n3. Creating dummy inputs...")
    print(f"   - Node features: [{num_nodes}, 6]")
    print(f"   - Edge indices: [2, {num_edges}]")
    print(f"   - Edge attributes: [{num_edges}, 2]")
    
    dummy_x = torch.randn(num_nodes, 6, dtype=torch.float32)
    src_nodes = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    dst_nodes = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    dummy_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
    dummy_edge_attr = torch.randn(num_edges, 2, dtype=torch.float32)
    
    print("   ✓ Dummy inputs created")
    
    # Test forward pass
    print("\n4. Testing forward pass with dummy inputs...")
    with torch.no_grad():
        test_output = wrapped_model(dummy_x, dummy_edge_index, dummy_edge_attr)
    print(f"   ✓ Forward pass successful")
    print(f"   - Output shape: {test_output.shape}")
    print(f"   - Sample predictions: {test_output[:5].squeeze().tolist()}")
    
    # Export to ONNX
    print(f"\n5. Exporting to ONNX format...")
    torch.onnx.export(
        wrapped_model,
        (dummy_x, dummy_edge_index, dummy_edge_attr),
        onnx_path,
        input_names=['node_features', 'edge_index', 'edge_attributes'],
        output_names=['anomaly_probabilities'],
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'edge_attributes': {0: 'num_edges'},
            'anomaly_probabilities': {0: 'num_edges'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    print(f"   ✓ ONNX export successful: '{onnx_path}'")
    
    # Verify ONNX model
    print("\n6. Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ✓ ONNX model is valid")
    
    # Get file sizes
    import os
    pt_size = os.path.getsize(model_path) / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024
    reduction = (1 - onnx_size / pt_size) * 100
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nModel Information:")
    print(f"  - PyTorch model: {model_path} ({pt_size:.1f} KB)")
    print(f"  - ONNX model: {onnx_path} ({onnx_size:.1f} KB)")
    print(f"  - Size reduction: {reduction:.1f}%")
    print(f"\nONNX Model Details:")
    print(f"  - Inputs:")
    print(f"      1. node_features: [num_nodes, 6] (float32)")
    print(f"      2. edge_index: [2, num_edges] (int64)")
    print(f"      3. edge_attributes: [num_edges, 2] (float32)")
    print(f"  - Output:")
    print(f"      1. anomaly_probabilities: [num_edges, 1] (float32)")
    print(f"\nCalibration Parameters:")
    print(f"  - Temperature scaling: {model.temperature}")
    print(f"  - Target score range (benign): <30/100")
    print(f"  - Target score range (malicious): 70-95/100")
    print(f"\nUsage with ONNX Runtime:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{onnx_path}')")
    print(f"  outputs = session.run(None, {{")
    print(f"      'node_features': node_features_array,")
    print(f"      'edge_index': edge_index_array,")
    print(f"      'edge_attributes': edge_attr_array")
    print(f"  }})")
    print(f"  risk_scores = outputs[0] * 100  # Convert to 0-100 scale")
    print("\n" + "="*70)
    
    return onnx_path

if __name__ == "__main__":
    export_calibrated_gnn_to_onnx(
        model_path='best_gnn_model_calibrated.pt',
        onnx_path='gnn_model_calibrated.onnx',
        num_nodes=50,
        num_edges=100
    )
