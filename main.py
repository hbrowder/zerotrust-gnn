import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # Simple GNN layer (GCN = Graph Conv Net)
import pandas as pd
from scapy.all import rdpcap  # For PCAP parsing
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

# Create graph from merged traffic data
graph_data = csv_to_graph('all_traffic.csv')
print(f"\n{graph_data}")
