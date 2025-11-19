# Zeek Integration Guide for ZeroTrustGNN

## Overview

**Zeek** (formerly Bro) is a powerful network analysis framework that provides richer protocol parsing and metadata extraction compared to Scapy. While Zeek cannot be installed directly in the Replit environment (requires root access and large dependencies), it can be used as an **offline preprocessing tool** to generate enhanced flow data for the GNN model.

## Why Use Zeek?

**Advantages over Scapy:**
- 📊 **Richer Protocol Analysis**: Deep inspection of HTTP, DNS, SSL/TLS, SSH, and more
- 🔍 **Behavioral Metadata**: Connection states, duration, byte counts, packet counts
- 🚨 **Built-in Anomaly Detection**: Suspicious activity notices and alerts
- 📝 **Structured Logs**: Well-formatted TSV/JSON output with consistent schema
- 🎯 **Application Layer Visibility**: Extract URLs, user agents, file hashes, certificates

**When to Use Zeek:**
- You have access to a Linux server or workstation where you can install Zeek
- You need deeper protocol analysis beyond basic IP/TCP/UDP parsing
- You want to preprocess large PCAP datasets offline before uploading to the API
- You're analyzing corporate network traffic that requires application-layer insights

## Installation (Offline Server Only)

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y zeek
```

### macOS
```bash
brew install zeek
```

### From Source
```bash
git clone --recursive https://github.com/zeek/zeek
cd zeek
./configure
make -j$(nproc)
sudo make install
```

## Processing PCAP Files with Zeek

### Basic Usage

```bash
# Process a PCAP file
zeek -r /path/to/capture.pcap

# This creates multiple log files in the current directory:
# - conn.log (connection summaries)
# - dns.log (DNS queries)
# - http.log (HTTP requests)
# - ssl.log (TLS/SSL handshakes)
# - notice.log (detected anomalies)
# ... and more
```

### Extract Connection Data

The most important file for GNN analysis is `conn.log`, which contains flow-level information:

```bash
# Process PCAP and extract connection log
zeek -r capture.pcap

# View conn.log fields
head -1 conn.log
# Fields: ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, 
#         service, duration, orig_bytes, resp_bytes, conn_state, ...
```

### Convert Zeek Logs to CSV

Zeek logs are TSV (tab-separated) by default. Convert to CSV for easier processing:

```bash
# Extract relevant fields for GNN
zeek-cut ts id.orig_h id.orig_p id.resp_h id.resp_p proto orig_bytes resp_bytes duration < conn.log > flows.csv

# Or use Python to convert
python3 << EOF
import pandas as pd

# Read Zeek conn.log (skip comments starting with #)
df = pd.read_csv('conn.log', sep='\t', comment='#', 
                 names=['ts', 'uid', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 
                        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 
                        'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 
                        'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 
                        'resp_ip_bytes', 'tunnel_parents'])

# Select fields needed for GNN
gnn_df = df[['src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'orig_bytes']]

# Map protocol names to numbers (tcp=6, udp=17, icmp=1)
protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
gnn_df['protocol'] = gnn_df['proto'].map(protocol_map).fillna(0).astype(int)
gnn_df['bytes'] = gnn_df['orig_bytes'].fillna(0).astype(int)

# Save to CSV
gnn_df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'bytes', 'protocol']].to_csv('flows_for_gnn.csv', index=False)
print(f"Processed {len(gnn_df)} flows from Zeek conn.log")
EOF
```

## Integration with ZeroTrustGNN API

### Option 1: Preprocess Locally, Upload PCAP to API

```bash
# 1. Process large PCAP with Zeek locally
zeek -r large_capture.pcap

# 2. Use Zeek logs to identify suspicious IPs
grep -i "notice" notice.log

# 3. Filter PCAP to suspicious traffic only
tcpdump -r large_capture.pcap 'host 192.168.1.100' -w suspicious_only.pcap

# 4. Upload smaller PCAP to ZeroTrustGNN API
base64 suspicious_only.pcap > suspicious_base64.txt

# 5. Send to API
curl -X POST https://your-api.repl.co/scan \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"file\": \"$(cat suspicious_base64.txt)\"}"
```

### Option 2: Future Enhancement - Zeek CSV Upload

**Currently not implemented**, but could be added to the API:

```python
# Hypothetical future endpoint: POST /scan-csv
# Accept Zeek conn.log or CSV flows directly

@app.route('/scan-csv', methods=['POST'])
@require_api_key
def scan_zeek_csv():
    """
    Accept Zeek conn.log or CSV flows for analysis
    Bypasses Scapy parsing for preprocessed data
    """
    # Parse CSV
    # Convert to graph format
    # Run GNN inference
    # Return alerts
```

## Zeek Features for Enhanced Analysis

### 1. Protocol-Specific Anomalies

```bash
# HTTP anomalies
zeek -r capture.pcap http-suspicious.zeek

# DNS tunneling detection
zeek -r capture.pcap dns-tunnel.zeek

# SSH brute force attempts
zeek -r capture.pcap ssh-detect.zeek
```

### 2. File Extraction

```bash
# Extract files transferred over HTTP, FTP, etc.
zeek -r capture.pcap extract-all-files.zeek

# Extracted files appear in ./extract_files/
# Can compute file hashes and check against threat intel
```

### 3. TLS Certificate Analysis

```bash
# Extract SSL/TLS certificates
zeek -r capture.pcap ssl-log-certs.zeek

# Check for:
# - Self-signed certificates
# - Expired certificates
# - Certificate mismatches
# - Weak cipher suites
```

### 4. Geolocation and Intelligence

```bash
# Add GeoIP database
zeek -r capture.pcap geoip-enrichment.zeek

# Enrich conn.log with country codes
# Detect connections to unusual geographic locations
```

## Comparison: Scapy vs Zeek

| Feature | Scapy (Current) | Zeek (Offline) |
|---------|----------------|----------------|
| **Installation** | Python package | Requires root access |
| **PCAP Parsing** | ✅ Basic IP/TCP/UDP | ✅ Deep protocol analysis |
| **Replit Compatible** | ✅ Yes | ❌ No (too large, needs root) |
| **Protocol Depth** | Layer 3-4 | Layer 3-7 (application layer) |
| **Output Format** | Python objects | Structured TSV/JSON logs |
| **Performance** | Fast for small files | Optimized for large captures |
| **Anomaly Detection** | ❌ None | ✅ Built-in notices |
| **File Extraction** | ❌ Manual | ✅ Automatic |
| **Use Case** | Real-time API processing | Offline batch analysis |

## Recommended Workflow

### For Production Use with Zeek

```
┌─────────────────┐
│ Network Traffic │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Zeek Analysis  │ ← Offline server
│  (conn.log)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Filter/Enrich   │ ← Select suspicious flows
│ (Python script) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Generate PCAP  │ ← Create filtered PCAP
│  (tcpdump)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ ZeroTrustGNN    │ ← Upload to Replit API
│  API (/scan)    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Risk Scores    │ ← GNN predictions
│  & Alerts       │
└─────────────────┘
```

### For Current Scapy-Based API

```
┌─────────────────┐
│   PCAP File     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ ZeroTrustGNN    │ ← Direct upload (Base64)
│  API (/scan)    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Scapy Parsing   │ ← Extract IP flows
└────────┬────────┘
         │
         v
┌─────────────────┐
│  GNN Analysis   │ ← Graph construction
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Risk Scores    │ ← Calibrated predictions
│  & Alerts       │
└─────────────────┘
```

## Conclusion

**Current Solution**: Scapy provides sufficient packet parsing for the ZeroTrustGNN API to analyze network flows in real-time within the Replit environment.

**Future Enhancement**: Users with access to Zeek on external servers can preprocess large PCAP datasets offline, filter to suspicious traffic, and upload smaller PCAPs to the API for GNN-based risk scoring.

**Best of Both Worlds**: Use Zeek's rich analysis for filtering and preprocessing, then leverage ZeroTrustGNN's calibrated GNN model for final risk assessment.

---

**Note**: Zeek installation requires root access and significant disk space (~500MB+), which is why it cannot be used directly in the Replit environment. This guide documents it as a complementary tool for users with their own infrastructure.
