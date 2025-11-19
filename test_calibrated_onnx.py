import onnxruntime as ort
import numpy as np
import pandas as pd

def load_normalization_stats(train_csv_path):
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

print("="*70)
print("TESTING CALIBRATED ONNX MODEL")
print("="*70)

# Load normalization stats and test data
node_mean, node_std, edge_mean, edge_std = load_normalization_stats('train_traffic.csv')
node_features, edge_index, edge_attr, labels = load_graph('test_traffic.csv', node_mean, node_std, edge_mean, edge_std)

# Run inference with calibrated ONNX model
session = ort.InferenceSession('gnn_model_calibrated.onnx')
outputs = session.run(None, {
    'node_features': node_features.astype(np.float32),
    'edge_index': edge_index.astype(np.int64),
    'edge_attributes': edge_attr.astype(np.float32)
})
risk_scores = outputs[0].squeeze() * 100

benign_scores = risk_scores[labels == 0]
malicious_scores = risk_scores[labels == 1]

print(f"\n✅ CALIBRATED MODEL RESULTS")
print("="*70)
print(f"\nBENIGN TRAFFIC ({len(benign_scores)} flows):")
print(f"  Average Risk:  {benign_scores.mean():.2f}/100  (target: <30) {'✓' if benign_scores.mean() < 30 else '✗'}")
print(f"  Median:        {np.median(benign_scores):.2f}/100")
print(f"  Range:         {benign_scores.min():.2f} - {benign_scores.max():.2f}")

print(f"\nMALICIOUS TRAFFIC ({len(malicious_scores)} flows):")
print(f"  Average Risk:  {malicious_scores.mean():.2f}/100  (target: 70-95) {'✓' if 70 <= malicious_scores.mean() <= 95 else '✗'}")
print(f"  Median:        {np.median(malicious_scores):.2f}/100")
print(f"  Range:         {malicious_scores.min():.2f} - {malicious_scores.max():.2f}")

print(f"\n⚡ SEPARATION: {abs(malicious_scores.mean() - benign_scores.mean()):.1f} points")

accuracy = ((risk_scores >= 50) == labels).mean() * 100
print(f"\n🎯 ACCURACY: {accuracy:.2f}%")

print("\n" + "="*70)
if benign_scores.mean() < 30 and 70 <= malicious_scores.mean() <= 95:
    print("✅ TARGET ACHIEVED - Model is ready for production!")
else:
    print("⚠️ Target not fully met")
print("="*70)
