import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # Simple GNN layer (GCN = Graph Conv Net)
from torch_geometric.loader import DataLoader
import pandas as pd
from scapy.all import rdpcap  # For PCAP parsing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os

# Your code will go here
print("Setup ready!")

def pcap_to_csv(pcap_file, csv_output, label='benign'):  # label: 'benign' or 'malicious'
  packets = rdpcap(pcap_file)
  flows = []
  for pkt in packets:
      if pkt.haslayer('IP'):
          flow = {
              'src_ip': pkt['IP'].src,
              'dst_ip': pkt['IP'].dst,
              'src_port': pkt['TCP'].sport if pkt.haslayer('TCP') else pkt['UDP'].sport if pkt.haslayer('UDP') else 0,
              'dst_port': pkt['TCP'].dport if pkt.haslayer('TCP') else pkt['UDP'].dport if pkt.haslayer('UDP') else 0,
              'bytes': len(pkt),
              'protocol': pkt['IP'].proto,  # 1=ICMP, 6=TCP, 17=UDP
              'label': 0 if label == 'benign' else 1  # 0=normal, 1=anomaly
          }
          flows.append(flow)
  df = pd.DataFrame(flows)
  df.to_csv(csv_output, index=False)
  print(f"Saved {len(df)} flows to {csv_output}")

# Check if CIC-IDS2017 dataset is available, otherwise use PCAP files
if os.path.exists('cicids_traffic.csv'):
    print("\n=== Using CIC-IDS2017 Dataset ===")
    merged_df = pd.read_csv('cicids_traffic.csv')
    merged_df.to_csv('all_traffic.csv', index=False)
    print(f"Loaded {len(merged_df)} flows from cicids_traffic.csv")
    print(f"  - Benign flows: {len(merged_df[merged_df['label'] == 0])}")
    print(f"  - Malicious flows: {len(merged_df[merged_df['label'] == 1])}")
else:
    print("\n=== Using PCAP Files ===")
    # Run it
    pcap_to_csv('http.cap', 'benign.csv', 'benign')  # Normal traffic
    pcap_to_csv('ctf-icmp.pcap', 'malicious.csv', 'malicious')  # Scan traffic

    # Merge the CSV files
    print("\nMerging CSV files...")
    merged_df = pd.concat([pd.read_csv('benign.csv'), pd.read_csv('malicious.csv')])
    merged_df.to_csv('all_traffic.csv', index=False)
    print(f"Merged {len(merged_df)} total flows into all_traffic.csv")
    print(f"  - Benign flows: {len(merged_df[merged_df['label'] == 0])}")
    print(f"  - Malicious flows: {len(merged_df[merged_df['label'] == 1])}")

# Convert CSV to PyTorch Geometric graph
def csv_to_graph(csv_file):
    df = pd.read_csv(csv_file)
    
    # Get unique IPs (nodes)
    unique_ips = list(set(df['src_ip'].unique().tolist() + df['dst_ip'].unique().tolist()))
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
    num_nodes = len(unique_ips)
    
    print(f"\nBuilding graph from {csv_file}...")
    print(f"  Nodes (unique IPs): {num_nodes}")
    
    # Build node features: aggregated stats per IP
    node_features = torch.zeros((num_nodes, 6))  # [avg_src_port_when_src, avg_dst_port_when_dst, protocol_diversity, total_bytes, num_outgoing, num_incoming]
    
    for idx, ip in enumerate(unique_ips):
        src_flows = df[df['src_ip'] == ip]
        dst_flows = df[df['dst_ip'] == ip]
        all_flows = pd.concat([src_flows, dst_flows])
        
        # Avg source port when IP is the source
        if len(src_flows) > 0:
            node_features[idx, 0] = src_flows['src_port'].mean()
        
        # Avg destination port when IP is the destination
        if len(dst_flows) > 0:
            node_features[idx, 1] = dst_flows['dst_port'].mean()
        
        # Protocol diversity and total bytes across all flows
        if len(all_flows) > 0:
            node_features[idx, 2] = all_flows['protocol'].nunique()
            node_features[idx, 3] = all_flows['bytes'].sum()
        
        # Flow counts
        node_features[idx, 4] = len(src_flows)  # Outgoing flows
        node_features[idx, 5] = len(dst_flows)  # Incoming flows
    
    # Build edges from flows (src_ip -> dst_ip)
    edge_index = []
    edge_attr = []
    
    for _, row in df.iterrows():
        src_idx = ip_to_idx[row['src_ip']]
        dst_idx = ip_to_idx[row['dst_ip']]
        
        edge_index.append([src_idx, dst_idx])
        edge_attr.append([row['bytes'], row['protocol'], row['label']])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    print(f"  Edges (flows): {data.edge_index.shape[1]}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Graph created successfully!")
    
    return data

# Split and balance the dataset for training
def split_and_balance_data(csv_file, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file)
    
    print(f"\n=== Splitting and Balancing Dataset ===")
    print(f"Original dataset: {len(df)} flows")
    print(f"  Benign: {len(df[df['label'] == 0])}")
    print(f"  Malicious: {len(df[df['label'] == 1])}")
    
    # Remove duplicate flows to prevent data leakage
    df_before = len(df)
    df = df.drop_duplicates(subset=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'bytes', 'protocol'], keep='first')
    df = df.reset_index(drop=True)
    print(f"\nAfter deduplication: {len(df)} flows (removed {df_before - len(df)} duplicates)")
    print(f"  Benign: {len(df[df['label'] == 0])}")
    print(f"  Malicious: {len(df[df['label'] == 1])}")
    
    # Separate by class
    benign_df = df[df['label'] == 0]
    malicious_df = df[df['label'] == 1]
    
    # Balance by undersampling the majority class WITHOUT replacement
    min_class_size = min(len(benign_df), len(malicious_df))
    
    # Undersample only the majority class, keep minority class intact
    if len(benign_df) < len(malicious_df):
        # Benign is minority, keep all benign and undersample malicious
        benign_balanced = benign_df
        malicious_balanced = malicious_df.sample(n=min_class_size, random_state=random_state, replace=False)
    else:
        # Malicious is minority, keep all malicious and undersample benign
        benign_balanced = benign_df.sample(n=min_class_size, random_state=random_state, replace=False)
        malicious_balanced = malicious_df
    
    # Combine balanced data
    balanced_df = pd.concat([benign_balanced, malicious_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\nBalanced dataset: {len(balanced_df)} flows")
    print(f"  Benign: {len(balanced_df[balanced_df['label'] == 0])}")
    print(f"  Malicious: {len(balanced_df[balanced_df['label'] == 1])}")
    
    # Split into train/test with stratification
    # For very small datasets, ensure both splits have samples from both classes
    if len(balanced_df) < 10:
        print(f"\nWarning: Small dataset ({len(balanced_df)} samples). Using manual split to ensure class representation.")
        benign_split = balanced_df[balanced_df['label'] == 0]
        malicious_split = balanced_df[balanced_df['label'] == 1]
        
        # Check if we have enough samples to create meaningful splits
        if len(benign_split) < 2 or len(malicious_split) < 2:
            raise ValueError(
                f"Insufficient unique samples after deduplication and balancing: "
                f"{len(benign_split)} benign, {len(malicious_split)} malicious. "
                f"Need at least 2 samples per class to create train/test splits with class representation. "
                f"Consider using more diverse PCAP files with unique network flows."
            )
        
        # Allocate at least 1 sample from each class to each split
        # Use floor division to split evenly, ensuring at least 1 for test
        n_benign_test = max(1, len(benign_split) // 2)
        n_malicious_test = max(1, len(malicious_split) // 2)
        
        benign_test = benign_split.sample(n=n_benign_test, random_state=random_state)
        malicious_test = malicious_split.sample(n=n_malicious_test, random_state=random_state)
        
        test_df = pd.concat([benign_test, malicious_test])
        train_df = balanced_df.drop(test_df.index)
        
        # Shuffle both sets
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(
            balanced_df, 
            test_size=test_size, 
            stratify=balanced_df['label'],
            random_state=random_state
        )
    
    # Validate that both splits contain both classes
    train_classes = set(train_df['label'].unique())
    test_classes = set(test_df['label'].unique())
    expected_classes = {0, 1}
    
    if train_classes != expected_classes or test_classes != expected_classes:
        raise ValueError(
            f"Split validation failed! Train has classes {train_classes}, "
            f"Test has classes {test_classes}, but both should have {expected_classes}. "
            f"This indicates insufficient data diversity."
        )
    
    print(f"\nTrain set: {len(train_df)} flows")
    print(f"  Benign: {len(train_df[train_df['label'] == 0])}")
    print(f"  Malicious: {len(train_df[train_df['label'] == 1])}")
    
    print(f"\nTest set: {len(test_df)} flows")
    print(f"  Benign: {len(test_df[test_df['label'] == 0])}")
    print(f"  Malicious: {len(test_df[test_df['label'] == 1])}")
    
    # Save to CSV
    train_df.to_csv('train_traffic.csv', index=False)
    test_df.to_csv('test_traffic.csv', index=False)
    
    return train_df, test_df

# Split and balance data
train_df, test_df = split_and_balance_data('all_traffic.csv')

# Create graphs for train and test sets
print("\n=== Creating Training Graph ===")
train_graph = csv_to_graph('train_traffic.csv')

print("\n=== Creating Test Graph ===")
test_graph = csv_to_graph('test_traffic.csv')

# Create DataLoader for batching (for GNN training)
print("\n=== Setting up DataLoaders ===")
# For graph-level tasks, we typically use batch_size=1 since each graph represents the entire network
# But you can adjust this if you have multiple graph samples
train_loader = DataLoader([train_graph], batch_size=1, shuffle=True)
test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)

print(f"Train loader: {len(train_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

print("\n=== Summary ===")
print(f"Training graph: {train_graph}")
print(f"Test graph: {test_graph}")
print("\nDatasets ready for GNN training!")
