# CIC-IDS2017 Dataset Integration Guide

## Overview
This guide helps you integrate the CIC-IDS2017 dataset into your ZeroTrustGNN pipeline for improved training with diverse attack types.

## Quick Start (Using Sample Data)

The project includes scripts to generate and use a sample CIC-IDS2017-style dataset:

```bash
# 1. Generate sample dataset (500 flows, 8 attack types)
python create_sample_cicids.py

# 2. Convert to pipeline format
python integrate_cicids2017.py sample_cicids2017.csv --sample 500

# 3. Run the pipeline (automatically uses cicids_traffic.csv if available)
python main.py
```

### Current Results with Sample Data
- **Total flows**: 500 (no duplicates)
- **After balancing**: 400 flows (200 benign, 200 malicious)
- **Training graph**: 68 nodes, 320 edges
- **Test graph**: 55 nodes, 80 edges
- **Attack diversity**: 8 different attack types (PortScan, DDoS, DoS Hulk, Bot, Web Attack, FTP-Patator, SSH-Patator)

**This is a 160x improvement over the original PCAP files!**

---

## Getting the Full CIC-IDS2017 Dataset (2.8M+ Flows)

### Option 1: Kaggle (Easiest)

1. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Place the downloaded `kaggle.json` in `~/.kaggle/`

3. **Download the dataset**:
   ```bash
   # Option A: Full dataset
   kaggle datasets download -d cicdataset/cicids2017
   unzip cicids2017.zip
   
   # Option B: Specific day (smaller)
   kaggle datasets download -d cicdataset/cicids2017 -f Monday-WorkingHours.pcap_ISCX.csv
   ```

4. **Process for pipeline**:
   ```bash
   python integrate_cicids2017.py Monday-WorkingHours.pcap_ISCX.csv --sample 5000
   python main.py
   ```

### Option 2: Official Source

1. **Visit**: https://www.unb.ca/cic/datasets/ids-2017.html
2. **Download**:
   - `MachineLearningCSV.zip` (Pre-processed CSV files)
   - OR `GeneratedLabelledFlows.zip` (Labeled network flows)
3. **Extract** and process:
   ```bash
   unzip MachineLearningCSV.zip
   python integrate_cicids2017.py Monday-WorkingHours.pcap_ISCX.csv --sample 5000
   ```

### Option 3: IMPACT Cyber Trust

1. **Visit**: https://www.impactcybertrust.org/dataset_view?idDataset=917
2. Register for access (free for research)
3. Download CSV files
4. Process as above

---

## Dataset Details

### Included Attack Types
| Attack Type | Description | Flows in Sample |
|------------|-------------|-----------------|
| **BENIGN** | Normal network traffic | 200 |
| **PortScan** | Port scanning attacks | 75 |
| **DDoS** | Distributed Denial of Service | 75 |
| **DoS Hulk** | HTTP Denial of Service | 50 |
| **Bot** | Botnet traffic (IRC-based) | 40 |
| **Web Attack** | SQL injection, XSS | 35 |
| **FTP-Patator** | FTP brute force | 15 |
| **SSH-Patator** | SSH brute force | 10 |

### Full Dataset (5 Days)
- **Monday**: Benign traffic (normal network activity)
- **Tuesday**: FTP-Patator, SSH-Patator attacks
- **Wednesday**: DoS, Heartbleed attacks
- **Thursday**: Web Attack, Infiltration
- **Friday**: Botnet, DDoS, PortScan

**Total**: 2,830,540 flows with 80+ features

---

## Processing Custom Datasets

### Inspect Dataset Structure
```bash
python integrate_cicids2017.py your_dataset.csv --inspect
```

### Adjust Sample Size
```bash
# Small test (500 flows)
python integrate_cicids2017.py dataset.csv --sample 500

# Medium dataset (5000 flows)
python integrate_cicids2017.py dataset.csv --sample 5000

# Large dataset (50000 flows)
python integrate_cicids2017.py dataset.csv --sample 50000

# Full dataset (no sampling)
python integrate_cicids2017.py dataset.csv
```

### Output Format
The integration script converts CIC-IDS2017 format to pipeline format:

**Input (CIC-IDS2017)**:
- 80+ features including: Source/Dest IP, Ports, Protocol, Flow bytes, Packet stats, etc.

**Output (Pipeline Format)**:
```
src_ip, dst_ip, src_port, dst_port, bytes, protocol, label
```

---

## Comparison: PCAP vs CIC-IDS2017

| Metric | PCAP Files | CIC-IDS2017 Sample | Improvement |
|--------|-----------|-------------------|-------------|
| Total flows | 139 | 500 | 3.6x |
| Unique flows | 16 | 500 | 31x |
| After balancing | 4 | 400 | 100x |
| Training edges | 2 | 320 | 160x |
| Graph nodes | 4 | 68 | 17x |
| Attack types | 1 | 8 | 8x |
| Unique IPs | 6 | 68 | 11x |

---

## Troubleshooting

### "No column found" errors
- Run with `--inspect` flag to see actual column names
- The script auto-detects common variations (spaces, case)

### Out of memory
- Reduce `--sample` size
- Process one day at a time for full dataset

### Label issues
- Ensure the CSV has a "Label" column
- Check that labels include "BENIGN" for normal traffic

---

## Citation

When using CIC-IDS2017, please cite:

```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani,
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",
4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018
```

---

## Next Steps

1. ✅ Sample dataset integrated
2. 📥 Download full CIC-IDS2017 for production training
3. 🧠 Train GNN model on balanced dataset
4. 📊 Evaluate with diverse attack types
5. 🔄 Consider adding CIC-IDS2018 or UNSW-NB15 for more diversity
