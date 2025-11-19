"""
Create a sample CIC-IDS2017-style dataset with diverse attack types
for testing the GNN pipeline
"""

import pandas as pd
import random

def generate_sample_cicids_dataset(output_file='sample_cicids2017.csv', num_flows=500):
    """Generate a diverse sample dataset mimicking CIC-IDS2017 format"""
    
    random.seed(42)
    
    # Define diverse IP pools
    benign_ips = [f'192.168.1.{i}' for i in range(10, 50)]
    attacker_ips = [f'10.0.0.{i}' for i in range(100, 120)]
    server_ips = ['172.16.0.10', '172.16.0.20', '172.16.0.30', '8.8.8.8', '1.1.1.1']
    web_servers = ['93.184.216.34', '151.101.1.69', '104.244.42.1']
    
    # Attack types with their characteristics
    attack_types = {
        'BENIGN': {
            'label': 'BENIGN',
            'protocols': [6],  # TCP
            'ports': [80, 443, 22, 21, 25, 53],
            'bytes_range': (100, 5000)
        },
        'PortScan': {
            'label': 'PortScan',
            'protocols': [6],  # TCP
            'ports': list(range(1, 65535, 100)),  # Scanning many ports
            'bytes_range': (40, 100)
        },
        'DDoS': {
            'label': 'DDoS',
            'protocols': [6, 17],  # TCP, UDP
            'ports': [80, 443],
            'bytes_range': (10, 500)
        },
        'DoS Hulk': {
            'label': 'DoS Hulk',
            'protocols': [6],  # TCP
            'ports': [80, 8080],
            'bytes_range': (500, 2000)
        },
        'Bot': {
            'label': 'Bot',
            'protocols': [6],  # TCP
            'ports': [6667, 6697, 8080],  # IRC, HTTP
            'bytes_range': (100, 1000)
        },
        'Web Attack': {
            'label': 'Web Attack',
            'protocols': [6],  # TCP
            'ports': [80, 443, 8080],
            'bytes_range': (200, 3000)
        },
        'FTP-Patator': {
            'label': 'FTP-Patator',
            'protocols': [6],  # TCP
            'ports': [21],
            'bytes_range': (50, 200)
        },
        'SSH-Patator': {
            'label': 'SSH-Patator',
            'protocols': [6],  # TCP
            'ports': [22],
            'bytes_range': (80, 300)
        }
    }
    
    # Generate flows
    flows = []
    
    # Distribute attack types
    attack_distribution = {
        'BENIGN': int(num_flows * 0.4),
        'PortScan': int(num_flows * 0.15),
        'DDoS': int(num_flows * 0.15),
        'DoS Hulk': int(num_flows * 0.1),
        'Bot': int(num_flows * 0.08),
        'Web Attack': int(num_flows * 0.07),
        'FTP-Patator': int(num_flows * 0.03),
        'SSH-Patator': int(num_flows * 0.02)
    }
    
    for attack_type, count in attack_distribution.items():
        attack_info = attack_types[attack_type]
        
        for _ in range(count):
            # Select IPs based on attack type
            if attack_type == 'BENIGN':
                src_ip = random.choice(benign_ips)
                dst_ip = random.choice(server_ips + web_servers)
            elif attack_type == 'PortScan':
                src_ip = random.choice(attacker_ips)
                dst_ip = random.choice(server_ips)
            elif attack_type in ['Bot', 'DDoS', 'DoS Hulk']:
                src_ip = random.choice(attacker_ips + benign_ips)
                dst_ip = random.choice(server_ips)
            else:  # Web attacks, Patator
                src_ip = random.choice(attacker_ips)
                dst_ip = random.choice(web_servers + server_ips)
            
            # Generate flow characteristics
            protocol = random.choice(attack_info['protocols'])
            
            # For PortScan, vary destination port significantly
            if attack_type == 'PortScan':
                dst_port = random.choice(attack_info['ports'])
                src_port = random.randint(49152, 65535)
            else:
                dst_port = random.choice(attack_info['ports'])
                src_port = random.randint(1024, 65535)
            
            bytes_min, bytes_max = attack_info['bytes_range']
            flow_bytes = random.randint(bytes_min, bytes_max)
            
            # Create flow
            num_packets = random.randint(1, 50)
            flow = {
                ' Source IP': src_ip,
                ' Destination IP': dst_ip,
                ' Source Port': src_port,
                ' Destination Port': dst_port,
                ' Protocol': protocol,
                ' Flow Duration': random.randint(1000, 100000),
                ' Total Fwd Packets': num_packets,
                ' Total Backward Packets': random.randint(0, 40),
                ' Total Length of Fwd Packets': flow_bytes,  # Actual bytes (will be used)
                ' Total Length of Bwd Packets': random.randint(0, flow_bytes),
                ' Fwd Packet Length Max': random.randint(40, 1500),
                ' Fwd Packet Length Min': 40,
                ' Fwd Packet Length Mean': flow_bytes / max(1, num_packets),  # Average per packet
                ' Fwd Packet Length Std': random.uniform(0, 100),
                ' Flow Bytes/s': flow_bytes / max(1, random.uniform(0.1, 10)),
                ' Flow Packets/s': random.uniform(1, 1000),
                ' Label': attack_info['label']
            }
            
            flows.append(flow)
    
    # Create DataFrame
    df = pd.DataFrame(flows)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"=== Generated Sample CIC-IDS2017 Dataset ===")
    print(f"Output file: {output_file}")
    print(f"Total flows: {len(df)}")
    print(f"\nLabel distribution:")
    print(df[' Label'].value_counts())
    print(f"\nUnique IPs:")
    print(f"  Sources: {df[' Source IP'].nunique()}")
    print(f"  Destinations: {df[' Destination IP'].nunique()}")
    print(f"\nProtocols:")
    print(df[' Protocol'].value_counts())
    print(f"\nSample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generate_sample_cicids_dataset('sample_cicids2017.csv', num_flows=500)
    print("\n✓ Sample dataset created!")
    print("\nNext steps:")
    print("  1. Process it: python integrate_cicids2017.py sample_cicids2017.csv --sample 500")
    print("  2. Use in pipeline: Replace all_traffic.csv with cicids_pipeline_format.csv in main.py")
