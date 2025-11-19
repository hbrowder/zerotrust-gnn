import onnxruntime as ort
import numpy as np
import pandas as pd

def load_normalization_stats(train_csv_path):
    """
    Compute normalization statistics from training data
    """
    df = pd.read_csv(train_csv_path)
    
    # Get unique IPs for node mapping
    unique_ips = sorted(list(set(df['src_ip'].unique()) | set(df['dst_ip'].unique())))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    
    num_nodes = len(unique_ips)
    node_features = np.zeros((num_nodes, 6), dtype=np.float32)
    
    # Compute node features
    for ip_idx, ip in enumerate(unique_ips):
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        
        avg_src_port = src_flows['src_port'].mean() if len(src_flows) > 0 else 0
        avg_dst_port = dst_flows['dst_port'].mean() if len(dst_flows) > 0 else 0
        protocol_diversity = len(set(src_flows['protocol'].unique()) | set(dst_flows['protocol'].unique()))
        total_bytes_sent = src_flows['bytes'].sum()
        num_src_flows = len(src_flows)
        num_dst_flows = len(dst_flows)
        
        node_features[ip_idx] = [
            avg_src_port, avg_dst_port, protocol_diversity,
            total_bytes_sent, num_src_flows, num_dst_flows
        ]
    
    # Compute normalization stats
    node_mean = node_features.mean(axis=0, keepdims=True)
    node_std = node_features.std(axis=0, keepdims=True) + 1e-6
    
    # Edge features
    edge_mean = df[['bytes', 'protocol']].values.mean(axis=0, keepdims=True)
    edge_std = df[['bytes', 'protocol']].values.std(axis=0, keepdims=True) + 1e-6
    
    return node_mean, node_std, edge_mean, edge_std

def load_sample_graph_from_csv(csv_path, node_mean, node_std, edge_mean, edge_std):
    """
    Load a sample graph from CSV file and convert to ONNX input format
    """
    df = pd.read_csv(csv_path)
    
    # Get unique IPs for node mapping
    unique_ips = sorted(list(set(df['src_ip'].unique()) | set(df['dst_ip'].unique())))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    
    num_nodes = len(unique_ips)
    num_edges = len(df)
    
    # Initialize node features
    node_features = np.zeros((num_nodes, 6), dtype=np.float32)
    
    # Compute node features
    for ip_idx, ip in enumerate(unique_ips):
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        
        avg_src_port = src_flows['src_port'].mean() if len(src_flows) > 0 else 0
        avg_dst_port = dst_flows['dst_port'].mean() if len(dst_flows) > 0 else 0
        protocol_diversity = len(set(src_flows['protocol'].unique()) | set(dst_flows['protocol'].unique()))
        total_bytes_sent = src_flows['bytes'].sum()
        num_src_flows = len(src_flows)
        num_dst_flows = len(dst_flows)
        
        node_features[ip_idx] = [
            avg_src_port, avg_dst_port, protocol_diversity,
            total_bytes_sent, num_src_flows, num_dst_flows
        ]
    
    # Build edge index
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_attr = np.zeros((num_edges, 2), dtype=np.float32)
    edge_labels = np.zeros(num_edges, dtype=np.int32)
    
    for i, row in df.iterrows():
        src_idx = ip_to_idx[row['src_ip']]
        dst_idx = ip_to_idx[row['dst_ip']]
        
        edge_index[0, i] = src_idx
        edge_index[1, i] = dst_idx
        edge_attr[i] = [row['bytes'], row['protocol']]
        edge_labels[i] = row['label']
    
    # Normalize features using TRAINING statistics
    node_features = (node_features - node_mean) / node_std
    edge_attr = (edge_attr - edge_mean) / edge_std
    
    return node_features, edge_index, edge_attr, edge_labels, df

def run_onnx_inference(onnx_model_path, node_features, edge_index, edge_attr):
    """
    Run inference using ONNX Runtime
    
    Args:
        onnx_model_path: Path to ONNX model file
        node_features: Node features array [num_nodes, 6]
        edge_index: Edge connectivity array [2, num_edges]
        edge_attr: Edge attributes array [num_edges, 2]
    
    Returns:
        risk_scores: Risk scores for each edge [num_edges]
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    
    # Prepare inputs
    onnx_inputs = {
        'node_features': node_features.astype(np.float32),
        'edge_index': edge_index.astype(np.int64),
        'edge_attributes': edge_attr.astype(np.float32)
    }
    
    # Run inference
    outputs = session.run(None, onnx_inputs)
    
    # Convert probabilities to risk scores (0-100)
    anomaly_probs = outputs[0]
    risk_scores = anomaly_probs.squeeze() * 100
    
    return risk_scores

def main():
    print("="*70)
    print("ONNX RUNTIME INFERENCE - NETWORK ANOMALY DETECTION")
    print("="*70)
    
    # Load normalization statistics from training data
    print("\n1. Loading normalization statistics from training data...")
    node_mean, node_std, edge_mean, edge_std = load_normalization_stats('train_traffic.csv')
    print("   ✓ Normalization statistics loaded")
    
    # Load ONNX model
    onnx_model_path = 'gnn_model.onnx'
    print(f"\n2. Loading ONNX model: {onnx_model_path}")
    
    # Load sample graph from test data
    csv_path = 'test_traffic.csv'
    print(f"\n3. Loading sample graph from: {csv_path}")
    
    node_features, edge_index, edge_attr, edge_labels, df = load_sample_graph_from_csv(
        csv_path, node_mean, node_std, edge_mean, edge_std
    )
    
    print(f"   ✓ Graph loaded:")
    print(f"     - Nodes: {node_features.shape[0]}")
    print(f"     - Edges: {edge_index.shape[1]}")
    print(f"     - Benign flows: {(edge_labels == 0).sum()}")
    print(f"     - Malicious flows: {(edge_labels == 1).sum()}")
    
    # Run inference
    print(f"\n4. Running ONNX inference...")
    risk_scores = run_onnx_inference(onnx_model_path, node_features, edge_index, edge_attr)
    print(f"   ✓ Inference complete - {len(risk_scores)} edges scored")
    
    # Analyze results
    print("\n" + "="*70)
    print("RISK SCORE ANALYSIS")
    print("="*70)
    
    benign_scores = risk_scores[edge_labels == 0]
    malicious_scores = risk_scores[edge_labels == 1]
    
    print(f"\nBENIGN TRAFFIC ({len(benign_scores)} flows):")
    print(f"  Average Risk:  {benign_scores.mean():.2f}/100")
    print(f"  Risk Range:    {benign_scores.min():.2f} - {benign_scores.max():.2f}")
    print(f"  Std Dev:       {benign_scores.std():.2f}")
    print(f"  → Low risk scores indicate SAFE network activity")
    
    print(f"\nMALICIOUS TRAFFIC ({len(malicious_scores)} flows):")
    print(f"  Average Risk:  {malicious_scores.mean():.2f}/100")
    print(f"  Risk Range:    {malicious_scores.min():.2f} - {malicious_scores.max():.2f}")
    print(f"  Std Dev:       {malicious_scores.std():.2f}")
    print(f"  → High risk scores indicate DANGEROUS attack traffic")
    
    # Sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 20 flows)")
    print("="*70)
    print(f"{'Index':<8} {'Source IP':<18} {'Dest IP':<18} {'True Label':<12} {'Risk Score':<12} {'Prediction':<12}")
    print("-"*70)
    
    for i in range(min(20, len(risk_scores))):
        true_label = "Malicious" if edge_labels[i] == 1 else "Benign"
        predicted_label = "Malicious" if risk_scores[i] >= 50 else "Benign"
        correct = "✓" if (risk_scores[i] >= 50) == edge_labels[i] else "✗"
        
        print(f"{i:<8} {df.iloc[i]['src_ip']:<18} {df.iloc[i]['dst_ip']:<18} {true_label:<12} {risk_scores[i]:<12.2f} {predicted_label:<12} {correct}")
    
    # Compute accuracy
    predictions = (risk_scores >= 50).astype(int)
    accuracy = (predictions == edge_labels).mean() * 100
    
    print("\n" + "="*70)
    print(f"MODEL ACCURACY: {accuracy:.2f}%")
    print("="*70)
    
    # Risk distribution
    print(f"\nRISK DISTRIBUTION:")
    print(f"  Low Risk (0-25):      {(risk_scores < 25).sum()} flows")
    print(f"  Medium Risk (25-50):  {((risk_scores >= 25) & (risk_scores < 50)).sum()} flows")
    print(f"  High Risk (50-75):    {((risk_scores >= 50) & (risk_scores < 75)).sum()} flows")
    print(f"  Critical (75-100):    {(risk_scores >= 75).sum()} flows")
    
    # Show contrast between benign and malicious
    print(f"\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"✓ Benign traffic averages {benign_scores.mean():.1f}/100 risk")
    print(f"✓ Malicious traffic averages {malicious_scores.mean():.1f}/100 risk")
    print(f"✓ Risk difference: {abs(malicious_scores.mean() - benign_scores.mean()):.1f} points")
    print(f"✓ Model correctly identifies {accuracy:.1f}% of all traffic")
    
    print("\n" + "="*70)
    print("✓ ONNX INFERENCE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
