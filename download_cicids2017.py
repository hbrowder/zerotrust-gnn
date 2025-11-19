import requests
import pandas as pd
import os
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print(f"✓ Downloaded: {filename}")

def download_cicids2017_sample():
    """Download CIC-IDS2017 dataset from public sources"""
    print("=== Downloading CIC-IDS2017 Dataset ===\n")
    
    # Using direct links to sample files from the dataset
    # Note: For full dataset, users should visit https://www.unb.ca/cic/datasets/ids-2017.html
    
    datasets = {
        "Monday-WorkingHours.pcap_ISCX.csv": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv",
    }
    
    # Try downloading from GitHub mirror (smaller sample)
    print("Note: This downloads a sample. For the full dataset (2.8M+ flows),")
    print("visit: https://www.unb.ca/cic/datasets/ids-2017.html\n")
    
    for filename, url in datasets.items():
        try:
            print(f"Downloading {filename}...")
            download_file(url, filename)
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print("\nAlternative: Download manually from:")
            print("  - Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017")
            print("  - Official: https://www.unb.ca/cic/datasets/ids-2017.html")
            print("  - IMPACT: https://www.impactcybertrust.org/dataset_view?idDataset=917")
            return False
    
    return True

def process_cicids2017_to_pipeline_format(input_csv, output_csv, sample_size=1000):
    """Convert CIC-IDS2017 CSV to pipeline format"""
    print(f"\n=== Processing {input_csv} ===")
    
    # Read the CIC-IDS2017 CSV
    df = pd.read_csv(input_csv)
    print(f"Original dataset: {len(df)} flows")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # CIC-IDS2017 has columns like: Source IP, Destination IP, Source Port, 
    # Destination Port, Protocol, Flow Bytes/s, Label, etc.
    
    # Map to our pipeline format
    column_mapping = {
        ' Source IP': 'src_ip',
        ' Destination IP': 'dst_ip',
        ' Source Port': 'src_port',
        ' Destination Port': 'dst_port',
        ' Protocol': 'protocol',
        ' Label': 'label'
    }
    
    # Check which columns exist
    available_cols = {}
    for orig, new in column_mapping.items():
        for col in df.columns:
            if orig.strip().lower() in col.strip().lower():
                available_cols[col] = new
                break
    
    print(f"Mapped columns: {available_cols}")
    
    # Select and rename columns
    df_pipeline = df[list(available_cols.keys())].copy()
    df_pipeline.columns = [available_cols[col] for col in df_pipeline.columns]
    
    # Add bytes column (use Flow Bytes/s or Total Length of Fwd Packets if available)
    bytes_candidates = [' Flow Bytes/s', 'Flow Bytes/s', ' Total Length of Fwd Packets',
                       'Total Fwd Packet', ' Total Fwd Packets']
    for candidate in bytes_candidates:
        matching_cols = [col for col in df.columns if candidate.strip().lower() in col.strip().lower()]
        if matching_cols:
            df_pipeline['bytes'] = df[matching_cols[0]].fillna(0).astype(int).abs()
            print(f"Using '{matching_cols[0]}' for bytes")
            break
    
    if 'bytes' not in df_pipeline.columns:
        print("Warning: Could not find bytes column, using default value of 100")
        df_pipeline['bytes'] = 100
    
    # Convert label to binary (0=benign, 1=malicious)
    if 'label' in df_pipeline.columns:
        df_pipeline['label'] = (df_pipeline['label'].str.strip().str.upper() != 'BENIGN').astype(int)
    
    # Convert protocol to numeric if needed (TCP=6, UDP=17, ICMP=1)
    if 'protocol' in df_pipeline.columns:
        if df_pipeline['protocol'].dtype == 'object':
            protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
            df_pipeline['protocol'] = df_pipeline['protocol'].map(protocol_map).fillna(6).astype(int)
    
    # Sample if dataset is too large
    if len(df_pipeline) > sample_size:
        print(f"Sampling {sample_size} flows from {len(df_pipeline)} total flows")
        df_pipeline = df_pipeline.sample(n=sample_size, random_state=42)
    
    # Clean data
    df_pipeline = df_pipeline.dropna()
    df_pipeline = df_pipeline.reset_index(drop=True)
    
    # Save to CSV
    df_pipeline.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df_pipeline)} flows to {output_csv}")
    print(f"  Benign: {len(df_pipeline[df_pipeline['label'] == 0])}")
    print(f"  Malicious: {len(df_pipeline[df_pipeline['label'] == 1])}")
    
    return df_pipeline

if __name__ == "__main__":
    # Download dataset
    success = download_cicids2017_sample()
    
    if success:
        # Process to pipeline format
        process_cicids2017_to_pipeline_format(
            'Monday-WorkingHours.pcap_ISCX.csv',
            'cicids2017_processed.csv',
            sample_size=1000
        )
        print("\n✓ Dataset ready! Use 'cicids2017_processed.csv' in your pipeline.")
    else:
        print("\n✗ Download failed. Please download manually and run:")
        print("  python download_cicids2017.py")
