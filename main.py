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
