import torch
import torch.onnx
from torch_geometric.data import Data
from train_gnn import GNNAnomalyDetector
import numpy as np

class GNNONNXWrapper(torch.nn.Module):
    """
    Wrapper for GNN model to make it ONNX-compatible
    Converts separate tensor inputs into PyTorch Geometric Data format
    """
    def __init__(self, gnn_model):
        super(GNNONNXWrapper, self).__init__()
        self.gnn_model = gnn_model
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, 6]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, 2] (bytes, protocol only)
        
        Returns:
            predictions: Anomaly probabilities [num_edges, 1]
        """
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Forward pass through GNN
        predictions = self.gnn_model(data)
        
        return predictions

def export_gnn_to_onnx(
    model_path='best_gnn_model.pt',
    onnx_path='gnn_model.onnx',
    num_nodes=50,
    num_edges=100
):
    """
    Export trained GNN model to ONNX format
    
    Args:
        model_path: Path to trained PyTorch model (.pt file)
        onnx_path: Path to save ONNX model (.onnx file)
        num_nodes: Number of nodes in dummy graph (for inference shape)
        num_edges: Number of edges in dummy graph (for inference shape)
    """
    
    print("="*70)
    print("EXPORTING GNN MODEL TO ONNX FORMAT")
    print("="*70)
    
    # Load trained model
    print(f"\n1. Loading trained model from '{model_path}'...")
    model = GNNAnomalyDetector(num_node_features=6, hidden_dim=128, edge_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Wrap model for ONNX compatibility
    print("\n2. Wrapping model for ONNX compatibility...")
    wrapped_model = GNNONNXWrapper(model)
    wrapped_model.eval()
    print("   ✓ Model wrapped")
    
    # Create dummy inputs with realistic shapes
    print(f"\n3. Creating dummy inputs...")
    print(f"   - Node features: [{num_nodes}, 6]")
    print(f"   - Edge indices: [2, {num_edges}]")
    print(f"   - Edge attributes: [{num_edges}, 2]")
    
    # Dummy node features (normalized, matching training distribution)
    dummy_x = torch.randn(num_nodes, 6, dtype=torch.float32)
    
    # Dummy edge indices (random connections between nodes)
    src_nodes = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    dst_nodes = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
    dummy_edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
    
    # Dummy edge attributes (bytes and protocol only - normalized)
    dummy_edge_attr = torch.randn(num_edges, 2, dtype=torch.float32)
    
    print("   ✓ Dummy inputs created")
    
    # Test forward pass with dummy inputs
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
    
    # Print model info
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nONNX Model Information:")
    print(f"  - File: {onnx_path}")
    print(f"  - Inputs:")
    print(f"      1. node_features: [num_nodes, 6] (float32)")
    print(f"      2. edge_index: [2, num_edges] (int64)")
    print(f"      3. edge_attributes: [num_edges, 2] (float32)")
    print(f"  - Output:")
    print(f"      1. anomaly_probabilities: [num_edges, 1] (float32)")
    print(f"\nDynamic Axes (flexible input size):")
    print(f"  - num_nodes: Can vary for different graphs")
    print(f"  - num_edges: Can vary for different graphs")
    print(f"\nUsage with ONNX Runtime:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{onnx_path}')")
    print(f"  outputs = session.run(None, {{")
    print(f"      'node_features': node_features_array,")
    print(f"      'edge_index': edge_index_array,")
    print(f"      'edge_attributes': edge_attr_array")
    print(f"  }})")
    print(f"  anomaly_probs = outputs[0]")
    print(f"  risk_scores = anomaly_probs * 100  # Convert to 0-100 scale")
    print("\n" + "="*70)
    
    return onnx_path

if __name__ == "__main__":
    # Export with default dummy input shapes
    # These shapes are flexible thanks to dynamic_axes
    export_gnn_to_onnx(
        model_path='best_gnn_model.pt',
        onnx_path='gnn_model.onnx',
        num_nodes=50,   # Dummy: can be any number during inference
        num_edges=100   # Dummy: can be any number during inference
    )
