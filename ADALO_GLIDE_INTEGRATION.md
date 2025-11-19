# ZeroTrustGNN API - Adalo/Glide Integration Guide

## Overview

This API provides a `/scan` endpoint for network anomaly detection that works with no-code platforms like **Adalo** and **Glide**.

## API Endpoint

```
POST https://your-replit-app.repl.co/scan
```

### Request Format

```json
{
  "file": "BASE64_ENCODED_PCAP_FILE",
  "risk_threshold": 50
}
```

### Response Format

```json
{
  "success": true,
  "timestamp": "2025-11-19T18:30:00.000Z",
  "summary": {
    "total_flows": 25,
    "high_risk_flows": 5,
    "medium_risk_flows": 3,
    "low_risk_flows": 17,
    "average_risk_score": 35.4,
    "max_risk_score": 98.2,
    "alerts_triggered": 5
  },
  "alerts": [
    {
      "flow_id": 1,
      "risk_score": 98.2,
      "risk_level": "critical",
      "source_ip": "10.0.0.115",
      "source_port": 54321,
      "destination_ip": "172.16.0.30",
      "destination_port": 80,
      "protocol": "TCP",
      "bytes_transferred": 1500,
      "message": "CRITICAL risk flow detected: 10.0.0.115:54321 → 172.16.0.30:80"
    }
  ]
}
```

---

## Adalo Integration (Recommended)

Adalo supports file uploads with Base64 encoding, making it the best choice for PCAP scanning.

### Setup Steps:

#### 1. Add File Uploader Component
- Add a **File Uploader** component to your Adalo screen
- Configure to accept PCAP files (or all files)

#### 2. Create Custom Action
- Click **Add Action** → **Custom Actions** → **+ New Action**
- Configure:
  - **Name**: "Scan PCAP for Threats"
  - **Method**: POST
  - **URL**: `https://your-replit-app.repl.co/scan`
  - **Headers**: 
    ```
    Content-Type: application/json
    X-API-Key: YOUR_API_KEY_HERE
    ```
  
**Important**: Replace `YOUR_API_KEY_HERE` with your actual API key from Replit Secrets.

#### 3. Configure Request Body
```json
{
  "file": "[Magic Text: File Uploader's File]",
  "risk_threshold": 50
}
```

**Important**: Use Magic Text to select the File Uploader component's output. Adalo automatically converts it to Base64.

#### 4. Add Button to Trigger
- Add a **Button** component
- Set action to your Custom Action "Scan PCAP for Threats"
- Configure loading states and success/error handling

#### 5. Display Results
Create a list component to display alerts:

- **Title**: `{alert.message}`
- **Risk Score**: `{alert.risk_score}/100`
- **Risk Level**: `{alert.risk_level}`
- **Source**: `{alert.source_ip}:{alert.source_port}`
- **Destination**: `{alert.destination_ip}:{alert.destination_port}`

Add conditional visibility:
- Show red background if `risk_level = "critical"`
- Show orange if `risk_level = "high"`
- Show yellow if `risk_level = "medium"`

---

## Glide Integration (Limited)

**Note**: Glide has limitations with file uploads. Files uploaded in Glide stay as URLs, not Base64 content. You'll need a workaround.

### Option A: Use Middleware (Recommended)

1. **Set up Make/Zapier automation**:
   - Trigger: Glide webhook when file uploaded
   - Action 1: Download file from Glide URL
   - Action 2: Convert to Base64
   - Action 3: POST to `/scan` endpoint with X-API-Key header
   - Action 4: Send results back to Glide via webhook

2. **Configure Glide Workflow**:
   - Add **Trigger Webhook** action when user uploads file
   - Send file URL to Make/Zapier
   - Receive results via Glide's Webhook Trigger

### Option B: Direct API Call (Limited)

If PCAP files are already Base64-encoded in Glide:

1. **Add Call API Action**:
   - Method: POST
   - URL: `https://your-replit-app.repl.co/scan`
   - Headers: 
     ```
     Content-Type: application/json
     X-API-Key: YOUR_API_KEY_HERE
     ```
   - Body:
     ```json
     {
       "file": "$base64_column",
       "risk_threshold": 50
     }
     ```

2. **Parse Response**:
   - Use **Query JSON** column to extract:
     - `summary.total_flows`
     - `summary.high_risk_flows`
     - `alerts[0].risk_score`
   - Display in Glide components

---

## Testing the API

### Method 1: Using cURL (Terminal)

```bash
# Convert PCAP to Base64
base64 -i your_file.pcap > pcap_base64.txt

# Test API
curl -X POST https://your-replit-app.repl.co/scan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{"file": "'"$(cat pcap_base64.txt)"'", "risk_threshold": 50}'
```

### Method 2: Using Python

```python
import requests
import base64

# Read and encode PCAP file
with open('traffic.pcap', 'rb') as f:
    pcap_base64 = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post('https://your-replit-app.repl.co/scan', 
    headers={'X-API-Key': 'YOUR_API_KEY_HERE'},
    json={
        'file': pcap_base64,
        'risk_threshold': 50
    }
)

# Print results
result = response.json()
print(f"Total flows: {result['summary']['total_flows']}")
print(f"High risk flows: {result['summary']['high_risk_flows']}")
print(f"Alerts triggered: {result['summary']['alerts_triggered']}")

for alert in result['alerts']:
    print(f"\n{alert['risk_level'].upper()}: {alert['message']}")
    print(f"  Risk Score: {alert['risk_score']}/100")
```

---

## Risk Score Interpretation

| Score Range | Risk Level | Color | Action |
|-------------|-----------|-------|--------|
| 0-24 | **Low** | 🟢 Green | Monitor only |
| 25-49 | **Medium** | 🟡 Yellow | Review activity |
| 50-74 | **High** | 🟠 Orange | Investigate immediately |
| 75-100 | **Critical** | 🔴 Red | Block and investigate |

### Calibrated Ranges:
- **Benign traffic**: Typically scores 17.8-23.0/100
- **Malicious traffic**: Typically scores 80.7-93.4/100
- **Default threshold**: 50/100 (configurable via `risk_threshold`)

---

## Error Handling

### Common Errors:

**400 Bad Request - Missing file**
```json
{
  "success": false,
  "error": "Missing 'file' field in request body. Expected Base64-encoded PCAP file."
}
```
**Solution**: Ensure `file` field is present in JSON body

**400 Bad Request - Invalid Base64**
```json
{
  "success": false,
  "error": "Invalid Base64 encoding: Incorrect padding"
}
```
**Solution**: Verify file is properly Base64-encoded

**400 Bad Request - Invalid PCAP**
```json
{
  "success": false,
  "error": "Invalid PCAP file: No valid IP flows found in PCAP file"
}
```
**Solution**: Ensure file is a valid PCAP with IP traffic

---

## Example Adalo User Flow

1. **User uploads PCAP file** → File Uploader component
2. **User clicks "Scan" button** → Triggers Custom Action
3. **Adalo sends Base64 PCAP** → POST to `/scan` endpoint
4. **API processes PCAP** → GNN model analyzes network flows
5. **API returns alerts** → JSON response with risk scores
6. **Adalo displays results** → List of high-risk flows with colors

---

## Production Deployment

### 1. Deploy Replit App
- Click **Deployments** in Replit
- Configure production settings
- Deploy to get permanent URL

### 2. Update Adalo/Glide URLs
- Replace `your-replit-app.repl.co` with production URL
- Test integration end-to-end

### 3. Verify Authentication
- API key authentication is **required** (already configured in setup)
- Ensure your Adalo/Glide Custom Action includes:
  ```
  X-API-Key: YOUR_API_KEY_HERE
  ```
- Without this header, requests will return 401 Unauthorized

---

## Performance Notes

- **File size limit**: Recommended <10MB PCAP files
- **Processing time**: ~2-5 seconds for typical PCAP (100-500 flows)
- **Concurrent requests**: Supports multiple simultaneous scans
- **Rate limiting**: Configure as needed for production

---

## Support

For issues or questions:
1. Check `/health` endpoint for API status
2. Review error messages in response
3. Verify PCAP file format and encoding
4. Test with sample PCAP files first

**Health Check**: `GET https://your-replit-app.repl.co/health`

---

## Sample PCAP Files for Testing

Use the included test files:
- `test_traffic.csv` → Convert flows to PCAP for testing
- Small PCAP with benign traffic for baseline
- Mixed traffic PCAP for alert testing

Happy scanning! 🛡️
