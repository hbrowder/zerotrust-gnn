from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import tempfile
import onnxruntime as ort
import numpy as np
import pandas as pd
from scapy.all import rdpcap
from datetime import datetime

app = Flask(__name__)
CORS(app)

def process_pcap_to_graph(pcap_path):
    """
    Process PCAP file and convert to graph format for GNN inference
    Returns node_features, edge_index, edge_attr ready for ONNX model
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        raise ValueError(f"Invalid PCAP file: {str(e)}")
    
    flows = []
    for pkt in packets:
        if pkt.haslayer('IP'):
            ip_layer = pkt['IP']
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            bytes_transferred = len(pkt)
            
            src_port = 0
            dst_port = 0
            
            if pkt.haslayer('TCP'):
                src_port = pkt['TCP'].sport
                dst_port = pkt['TCP'].dport
            elif pkt.haslayer('UDP'):
                src_port = pkt['UDP'].sport
                dst_port = pkt['UDP'].dport
            
            flows.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'bytes': bytes_transferred,
                'protocol': protocol
            })
    
    if not flows:
        raise ValueError("No valid IP flows found in PCAP file")
    
    df = pd.DataFrame(flows)
    
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
        
        node_features[ip_idx] = [
            avg_src_port, avg_dst_port, protocol_diversity,
            total_bytes_sent, num_src_flows, num_dst_flows
        ]
    
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_attr = np.zeros((num_edges, 2), dtype=np.float32)
    
    for i, row in df.iterrows():
        src_idx = ip_to_idx[row['src_ip']]
        dst_idx = ip_to_idx[row['dst_ip']]
        
        edge_index[0, i] = src_idx
        edge_index[1, i] = dst_idx
        edge_attr[i] = [row['bytes'], row['protocol']]
    
    node_mean = node_features.mean(axis=0, keepdims=True)
    node_std = node_features.std(axis=0, keepdims=True) + 1e-6
    node_features = (node_features - node_mean) / node_std
    
    edge_mean = edge_attr.mean(axis=0, keepdims=True)
    edge_std = edge_attr.std(axis=0, keepdims=True) + 1e-6
    edge_attr = (edge_attr - edge_mean) / edge_std
    
    flow_details = []
    for i, row in df.iterrows():
        flow_details.append({
            'src_ip': row['src_ip'],
            'dst_ip': row['dst_ip'],
            'src_port': int(row['src_port']),
            'dst_port': int(row['dst_port']),
            'bytes': int(row['bytes']),
            'protocol': int(row['protocol'])
        })
    
    return node_features, edge_index, edge_attr, flow_details

def run_onnx_inference(node_features, edge_index, edge_attr, model_path='gnn_model_calibrated.onnx'):
    """Run ONNX model inference and return risk scores"""
    session = ort.InferenceSession(model_path)
    
    outputs = session.run(None, {
        'node_features': node_features.astype(np.float32),
        'edge_index': edge_index.astype(np.int64),
        'edge_attributes': edge_attr.astype(np.float32)
    })
    
    risk_scores = outputs[0].squeeze() * 100
    return risk_scores

def generate_alerts(risk_scores, flow_details, threshold=50):
    """Generate structured alerts based on risk scores"""
    alerts = []
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    
    for i, (score, flow) in enumerate(zip(risk_scores, flow_details)):
        risk_level = "low"
        if score >= 75:
            risk_level = "critical"
            high_risk_count += 1
        elif score >= 50:
            risk_level = "high"
            high_risk_count += 1
        elif score >= 25:
            risk_level = "medium"
            medium_risk_count += 1
        else:
            low_risk_count += 1
        
        if score >= threshold:
            protocol_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(flow['protocol'], f"Protocol-{flow['protocol']}")
            
            alerts.append({
                'flow_id': i + 1,
                'risk_score': round(float(score), 2),
                'risk_level': risk_level,
                'source_ip': flow['src_ip'],
                'source_port': flow['src_port'],
                'destination_ip': flow['dst_ip'],
                'destination_port': flow['dst_port'],
                'protocol': protocol_name,
                'bytes_transferred': flow['bytes'],
                'message': f"{risk_level.upper()} risk flow detected: {flow['src_ip']}:{flow['src_port']} → {flow['dst_ip']}:{flow['dst_port']}"
            })
    
    summary = {
        'total_flows': len(risk_scores),
        'high_risk_flows': high_risk_count,
        'medium_risk_flows': medium_risk_count,
        'low_risk_flows': low_risk_count,
        'average_risk_score': round(float(np.mean(risk_scores)), 2),
        'max_risk_score': round(float(np.max(risk_scores)), 2),
        'alerts_triggered': len(alerts)
    }
    
    return alerts, summary

@app.route('/scan', methods=['POST'])
def scan_pcap():
    """
    POST /scan endpoint for Adalo/Glide integration
    Accepts Base64-encoded PCAP file and returns JSON alerts
    """
    try:
        data = request.get_json()
        
        if not data or 'file' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "file" field in request body. Expected Base64-encoded PCAP file.'
            }), 400
        
        file_data = data['file']
        
        if file_data.startswith('data:'):
            file_data = file_data.split(',', 1)[1]
        
        try:
            pcap_bytes = base64.b64decode(file_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid Base64 encoding: {str(e)}'
            }), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
            tmp_file.write(pcap_bytes)
            tmp_path = tmp_file.name
        
        try:
            node_features, edge_index, edge_attr, flow_details = process_pcap_to_graph(tmp_path)
            
            risk_scores = run_onnx_inference(node_features, edge_index, edge_attr)
            
            threshold = data.get('risk_threshold', 50)
            alerts, summary = generate_alerts(risk_scores, flow_details, threshold)
            
            response = {
                'success': True,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'summary': summary,
                'alerts': alerts[:100]
            }
            
            return jsonify(response), 200
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'gnn_model_calibrated.onnx',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'name': 'ZeroTrustGNN API',
        'version': '1.0',
        'description': 'Network anomaly detection API using Graph Neural Networks',
        'endpoints': {
            'POST /scan': {
                'description': 'Scan PCAP file for network anomalies',
                'input': {
                    'file': 'Base64-encoded PCAP file (required)',
                    'risk_threshold': 'Alert threshold 0-100 (optional, default: 50)'
                },
                'output': {
                    'success': 'Boolean',
                    'timestamp': 'ISO 8601 timestamp',
                    'summary': 'Overall statistics',
                    'alerts': 'Array of high-risk flows'
                }
            },
            'GET /health': {
                'description': 'Health check endpoint',
                'output': {
                    'status': 'API status',
                    'model': 'ONNX model filename',
                    'timestamp': 'ISO 8601 timestamp'
                }
            }
        },
        'integration': {
            'adalo': 'Use Custom Action with POST method, send file as Base64 in JSON body',
            'glide': 'Use Call API action with POST method, trigger via webhook for file uploads'
        }
    }), 200

if __name__ == '__main__':
    print("="*70)
    print("🚀 ZeroTrustGNN API Server")
    print("="*70)
    print("\nEndpoints:")
    print("  POST /scan   - Upload PCAP file for anomaly detection")
    print("  GET  /health - Health check")
    print("  GET  /       - API documentation")
    print("\nIntegration:")
    print("  Adalo: Use Custom Action with Base64 file upload")
    print("  Glide: Use Call API action or webhook trigger")
    print("\nModel: gnn_model_calibrated.onnx")
    print("Calibration: Benign <30/100, Malicious 70-95/100")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
