import numpy as np
import pandas as pd
import onnxruntime as ort

def load_normalization_stats(train_csv_path):
    """Compute normalization statistics from training data"""
    df = pd.read_csv(train_csv_path)
    unique_ips = sorted(list(set(df['src_ip'].unique()) | set(df['dst_ip'].unique())))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    num_nodes = len(unique_ips)
    node_features = np.zeros((num_nodes, 6), dtype=np.float32)
    
    for ip_idx, ip in enumerate(unique_ips):
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        avg_src_port = src_flows['src_port'].mean() if len(src_flows) > 0 else 0
        avg_dst_port = dst_flows['dst_port'].mean() if len(dst_flows) > 0 else 0
        protocol_diversity = len(set(src_flows['protocol'].unique()) | set(dst_flows['protocol'].unique()))
        total_bytes_sent = src_flows['bytes'].sum()
        num_src_flows = len(src_flows)
        num_dst_flows = len(dst_flows)
        node_features[ip_idx] = [avg_src_port, avg_dst_port, protocol_diversity, total_bytes_sent, num_src_flows, num_dst_flows]
    
    node_mean = node_features.mean(axis=0, keepdims=True)
    node_std = node_features.std(axis=0, keepdims=True) + 1e-6
    edge_mean = df[['bytes', 'protocol']].values.mean(axis=0, keepdims=True)
    edge_std = df[['bytes', 'protocol']].values.std(axis=0, keepdims=True) + 1e-6
    return node_mean, node_std, edge_mean, edge_std

def load_graph(csv_path, node_mean, node_std, edge_mean, edge_std):
    """Load graph from CSV"""
    df = pd.read_csv(csv_path)
    unique_ips = sorted(list(set(df['src_ip'].unique()) | set(df['dst_ip'].unique())))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    num_nodes = len(unique_ips)
    num_edges = len(df)
    node_features = np.zeros((num_nodes, 6), dtype=np.float32)
    
    for ip_idx, ip in enumerate(unique_ips):
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        avg_src_port = src_flows['src_port'].mean() if len(src_flows) > 0 else 0
        avg_dst_port = dst_flows['dst_port'].mean() if len(dst_flows) > 0 else 0
        protocol_diversity = len(set(src_flows['protocol'].unique()) | set(dst_flows['protocol'].unique()))
        total_bytes_sent = src_flows['bytes'].sum()
        num_src_flows = len(src_flows)
        num_dst_flows = len(dst_flows)
        node_features[ip_idx] = [avg_src_port, avg_dst_port, protocol_diversity, total_bytes_sent, num_src_flows, num_dst_flows]
    
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_attr = np.zeros((num_edges, 2), dtype=np.float32)
    edge_labels = np.zeros(num_edges, dtype=np.int32)
    
    for i, row in df.iterrows():
        edge_index[0, i] = ip_to_idx[row['src_ip']]
        edge_index[1, i] = ip_to_idx[row['dst_ip']]
        edge_attr[i] = [row['bytes'], row['protocol']]
        edge_labels[i] = row['label']
    
    node_features = (node_features - node_mean) / node_std
    edge_attr = (edge_attr - edge_mean) / edge_std
    return node_features, edge_index, edge_attr, edge_labels

def run_inference(onnx_model_path, node_features, edge_index, edge_attr):
    """Run ONNX inference"""
    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(None, {
        'node_features': node_features.astype(np.float32),
        'edge_index': edge_index.astype(np.int64),
        'edge_attributes': edge_attr.astype(np.float32)
    })
    return outputs[0].squeeze() * 100

print("="*70)
print("DETAILED RISK SCORE DISTRIBUTION ANALYSIS")
print("="*70)

# Load data
node_mean, node_std, edge_mean, edge_std = load_normalization_stats('train_traffic.csv')
node_features, edge_index, edge_attr, labels = load_graph('test_traffic.csv', node_mean, node_std, edge_mean, edge_std)
risk_scores = run_inference('gnn_model.onnx', node_features, edge_index, edge_attr)

benign_scores = risk_scores[labels == 0]
malicious_scores = risk_scores[labels == 1]

print(f"\n📊 SCORE DISTRIBUTION BREAKDOWN")
print("="*70)

print(f"\nBENIGN TRAFFIC ({len(benign_scores)} flows):")
print(f"  Mean:      {benign_scores.mean():.2f}/100")
print(f"  Median:    {np.median(benign_scores):.2f}/100")
print(f"  Std Dev:   {benign_scores.std():.2f}")
print(f"  Min:       {benign_scores.min():.2f}/100")
print(f"  Max:       {benign_scores.max():.2f}/100")
print(f"\n  Score Ranges:")
print(f"    0-25:    {(benign_scores < 25).sum()} flows ({(benign_scores < 25).sum()/len(benign_scores)*100:.1f}%)")
print(f"    25-50:   {((benign_scores >= 25) & (benign_scores < 50)).sum()} flows ({((benign_scores >= 25) & (benign_scores < 50)).sum()/len(benign_scores)*100:.1f}%)")
print(f"    50-75:   {((benign_scores >= 50) & (benign_scores < 75)).sum()} flows ({((benign_scores >= 50) & (benign_scores < 75)).sum()/len(benign_scores)*100:.1f}%)")
print(f"    75-100:  {(benign_scores >= 75).sum()} flows ({(benign_scores >= 75).sum()/len(benign_scores)*100:.1f}%)  ⚠️ FALSE POSITIVES")

print(f"\nMALICIOUS TRAFFIC ({len(malicious_scores)} flows):")
print(f"  Mean:      {malicious_scores.mean():.2f}/100")
print(f"  Median:    {np.median(malicious_scores):.2f}/100")
print(f"  Std Dev:   {malicious_scores.std():.2f}")
print(f"  Min:       {malicious_scores.min():.2f}/100")
print(f"  Max:       {malicious_scores.max():.2f}/100")
print(f"\n  Score Ranges:")
print(f"    0-25:    {(malicious_scores < 25).sum()} flows ({(malicious_scores < 25).sum()/len(malicious_scores)*100:.1f}%)  ⚠️ FALSE NEGATIVES")
print(f"    25-50:   {((malicious_scores >= 25) & (malicious_scores < 50)).sum()} flows ({((malicious_scores >= 25) & (malicious_scores < 50)).sum()/len(malicious_scores)*100:.1f}%)  ⚠️ FALSE NEGATIVES")
print(f"    50-75:   {((malicious_scores >= 50) & (malicious_scores < 75)).sum()} flows ({((malicious_scores >= 50) & (malicious_scores < 75)).sum()/len(malicious_scores)*100:.1f}%)")
print(f"    75-100:  {(malicious_scores >= 75).sum()} flows ({(malicious_scores >= 75).sum()/len(malicious_scores)*100:.1f}%)")

print(f"\n⚠️ PROBLEM AREAS:")
print("="*70)
print(f"False Positives (Benign scored ≥50): {(benign_scores >= 50).sum()}/{len(benign_scores)} ({(benign_scores >= 50).sum()/len(benign_scores)*100:.1f}%)")
print(f"False Negatives (Malicious scored <50): {(malicious_scores < 50).sum()}/{len(malicious_scores)} ({(malicious_scores < 50).sum()/len(malicious_scores)*100:.1f}%)")

print(f"\n📈 IDEAL vs CURRENT:")
print("="*70)
print(f"CURRENT:")
print(f"  Benign:    {benign_scores.mean():.1f}/100 (range: {benign_scores.min():.1f}-{benign_scores.max():.1f})")
print(f"  Malicious: {malicious_scores.mean():.1f}/100 (range: {malicious_scores.min():.1f}-{malicious_scores.max():.1f})")
print(f"\nIDEAL TARGET:")
print(f"  Benign:    <30/100 (tight range: 0-30)")
print(f"  Malicious: 70-95/100 (tight range: 70-100)")

print(f"\n💡 RECOMMENDATION:")
print("="*70)
print("The model needs RETRAINING with better calibration, not just sigmoid adjustment.")
print("Current issues:")
print(f"  1. Wide overlap: Some benign scores up to {benign_scores.max():.1f}/100")
print(f"  2. Poor separation: Some malicious scores down to {malicious_scores.min():.1f}/100")
print(f"  3. High variance in both classes")
print("\nSolutions:")
print("  ✓ Retrain with focal loss or class weighting")
print("  ✓ Add more diverse training data")
print("  ✓ Use calibration techniques (Platt/Isotonic)")
print("  ✓ Increase model capacity or adjust architecture")
