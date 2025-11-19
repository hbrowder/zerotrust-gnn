"""
CIC-IDS2017 Dataset Integration Script

This script helps you integrate the CIC-IDS2017 dataset into your GNN pipeline.

DOWNLOAD INSTRUCTIONS:
=======================
1. Visit one of these sources:
   - Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017
   - Official: https://www.unb.ca/cic/datasets/ids-2017.html  
   - IMPACT: https://www.impactcybertrust.org/dataset_view?idDataset=917

2. Download one or more CSV files (e.g., Monday-WorkingHours.pcap_ISCX.csv)

3. Place the CSV file(s) in this directory

4. Run: python integrate_cicids2017.py <filename.csv>

USAGE:
======
python integrate_cicids2017.py Monday-WorkingHours.pcap_ISCX.csv --sample 1000
"""

import pandas as pd
import sys
import argparse

def inspect_cicids_csv(filename):
    """Inspect the structure of a CIC-IDS2017 CSV file"""
    print(f"\n=== Inspecting {filename} ===")
    df = pd.read_csv(filename, nrows=5)
    
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. '{col}'")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nLabel distribution (if available):")
    if ' Label' in df.columns:
        full_df = pd.read_csv(filename)
        print(full_df[' Label'].value_counts())
    elif 'Label' in df.columns:
        full_df = pd.read_csv(filename)
        print(full_df['Label'].value_counts())
    
    return df

def process_cicids2017(input_csv, output_csv='cicids_pipeline_format.csv', sample_size=None):
    """Convert CIC-IDS2017 CSV to pipeline format (src_ip, dst_ip, src_port, dst_port, bytes, protocol, label)"""
    
    print(f"\n=== Processing {input_csv} ===")
    
    # Read the dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} flows")
    
    # Standard CIC-IDS2017 column names (with leading spaces)
    # Try to find the right columns regardless of spacing
    
    def find_column(df, possible_names):
        """Find column that matches any of the possible names (case/space insensitive)"""
        # First pass: try exact matches (prefer these)
        for col in df.columns:
            col_clean = col.strip().lower().replace('_', ' ')
            for name in possible_names:
                name_clean = name.strip().lower().replace('_', ' ')
                if col_clean == name_clean:
                    return col
        
        # Second pass: try substring matches
        for name in possible_names:
            name_clean = name.strip().lower().replace('_', ' ')
            for col in df.columns:
                col_clean = col.strip().lower().replace('_', ' ')
                if name_clean in col_clean:
                    return col
        
        return None
    
    # Map columns
    src_ip_col = find_column(df, ['Source IP', 'Src IP', 'source ip'])
    dst_ip_col = find_column(df, ['Destination IP', 'Dst IP', 'destination ip'])
    src_port_col = find_column(df, ['Source Port', 'Src Port', 'source port'])
    dst_port_col = find_column(df, ['Destination Port', 'Dst Port', 'destination port'])
    protocol_col = find_column(df, ['Protocol', 'protocol'])
    label_col = find_column(df, ['Label', 'label'])
    
    # Find bytes column - prioritize actual byte-length fields, NOT duration or rates
    # Order matters: try actual byte counts first, then fallbacks
    bytes_col = find_column(df, [
        'Total Length of Fwd Packets',  # Actual forward bytes
        'Total Length of Bwd Packets',  # Actual backward bytes
        'Fwd Packet Length Mean',       # Average packet size
        'Total Fwd Packet',              # Total forward packet count (less ideal)
        'Packet Length Mean',            # Average packet length
    ])
    
    # Explicitly reject Flow Duration and Flow Bytes/s (time and rate, not bytes)
    if bytes_col and ('duration' in bytes_col.lower() or 'bytes/s' in bytes_col.lower()):
        print(f"Warning: Rejecting '{bytes_col}' as bytes column (duration/rate, not bytes)")
        bytes_col = None
    
    print(f"\nColumn mapping:")
    print(f"  Source IP: {src_ip_col}")
    print(f"  Dest IP: {dst_ip_col}")
    print(f"  Source Port: {src_port_col}")
    print(f"  Dest Port: {dst_port_col}")
    print(f"  Protocol: {protocol_col}")
    print(f"  Bytes: {bytes_col}")
    print(f"  Label: {label_col}")
    
    # Validate required columns
    missing_cols = []
    if not src_ip_col:
        missing_cols.append('Source IP')
    if not dst_ip_col:
        missing_cols.append('Destination IP')
    if not label_col:
        missing_cols.append('Label')
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Run with --inspect to see available columns.")
    
    # Create pipeline format DataFrame
    pipeline_df = pd.DataFrame()
    
    if src_ip_col:
        pipeline_df['src_ip'] = df[src_ip_col]
    if dst_ip_col:
        pipeline_df['dst_ip'] = df[dst_ip_col]
    if src_port_col:
        # Safe conversion: convert to numeric, coerce errors, fill NaN with 0
        pipeline_df['src_port'] = pd.to_numeric(df[src_port_col], errors='coerce').fillna(0).astype(int)
    if dst_port_col:
        pipeline_df['dst_port'] = pd.to_numeric(df[dst_port_col], errors='coerce').fillna(0).astype(int)
    if bytes_col:
        # Safe conversion: handle Infinity and non-numeric values
        bytes_series = pd.to_numeric(df[bytes_col], errors='coerce')
        # Replace infinity with NaN, then fill with default
        bytes_series = bytes_series.replace([float('inf'), float('-inf')], float('nan'))
        pipeline_df['bytes'] = bytes_series.fillna(100).abs().astype(int)
        print(f"  Using '{bytes_col}' for bytes (found {(~bytes_series.isna()).sum()} valid values)")
    else:
        print("  Warning: No suitable bytes column found, using default value of 100")
        pipeline_df['bytes'] = 100  # Default value
    
    # Handle protocol
    if protocol_col:
        if df[protocol_col].dtype == 'object':
            # Convert string protocol to numeric
            protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1, 'tcp': 6, 'udp': 17, 'icmp': 1}
            pipeline_df['protocol'] = df[protocol_col].map(protocol_map).fillna(6).astype(int)
        else:
            pipeline_df['protocol'] = df[protocol_col].fillna(6).astype(int)
    else:
        pipeline_df['protocol'] = 6  # Default to TCP
    
    # Handle label (convert to binary: 0=benign, 1=malicious)
    if label_col:
        # CIC-IDS2017 labels: BENIGN, Bot, DDoS, DoS, Heartbleed, Infiltration, PortScan, Web Attack, etc.
        pipeline_df['label'] = (df[label_col].astype(str).str.strip().str.upper() != 'BENIGN').astype(int)
        
        print(f"\nOriginal label distribution:")
        print(df[label_col].value_counts())
    else:
        print("Warning: No label column found, defaulting all to benign (0)")
        pipeline_df['label'] = 0
    
    # Remove rows with missing critical data
    pipeline_df = pipeline_df.dropna(subset=['src_ip', 'dst_ip'])
    
    # Sample if requested
    if sample_size and len(pipeline_df) > sample_size:
        print(f"\nSampling {sample_size} flows from {len(pipeline_df)} total")
        
        # Stratified sampling to maintain class balance
        if 'label' in pipeline_df.columns:
            benign = pipeline_df[pipeline_df['label'] == 0]
            malicious = pipeline_df[pipeline_df['label'] == 1]
            
            n_benign = min(len(benign), sample_size // 2)
            n_malicious = min(len(malicious), sample_size // 2)
            
            sampled_benign = benign.sample(n=n_benign, random_state=42) if len(benign) > 0 else benign
            sampled_malicious = malicious.sample(n=n_malicious, random_state=42) if len(malicious) > 0 else malicious
            
            pipeline_df = pd.concat([sampled_benign, sampled_malicious])
        else:
            pipeline_df = pipeline_df.sample(n=sample_size, random_state=42)
    
    pipeline_df = pipeline_df.reset_index(drop=True)
    
    # Save to CSV
    pipeline_df.to_csv(output_csv, index=False)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Output file: {output_csv}")
    print(f"Total flows: {len(pipeline_df)}")
    if 'label' in pipeline_df.columns:
        print(f"  Benign: {len(pipeline_df[pipeline_df['label'] == 0])}")
        print(f"  Malicious: {len(pipeline_df[pipeline_df['label'] == 1])}")
    
    print(f"\nSample data:")
    print(pipeline_df.head(10))
    
    return pipeline_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CIC-IDS2017 dataset for GNN pipeline')
    parser.add_argument('input_file', nargs='?', help='Input CIC-IDS2017 CSV file')
    parser.add_argument('--inspect', action='store_true', help='Inspect the CSV structure without processing')
    parser.add_argument('--sample', type=int, default=1000, help='Number of flows to sample (default: 1000)')
    parser.add_argument('--output', default='cicids_pipeline_format.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    if not args.input_file:
        print(__doc__)
        print("\n" + "="*70)
        print("ERROR: Please provide an input CSV file")
        print("="*70)
        print("\nEXAMPLE:")
        print("  python integrate_cicids2017.py Monday-WorkingHours.pcap_ISCX.csv --sample 1000")
        sys.exit(1)
    
    if args.inspect:
        inspect_cicids_csv(args.input_file)
    else:
        process_cicids2017(args.input_file, args.output, args.sample)
        print(f"\n✓ Ready to use in your pipeline!")
        print(f"  Replace 'all_traffic.csv' with '{args.output}' in main.py")
