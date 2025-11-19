"""
Data Anonymization Module for GDPR Compliance
Handles IP address pseudonymization with consistent hashing
"""
import hashlib
import os
from typing import Optional

ANONYMIZATION_SALT = os.environ.get('ANONYMIZATION_SALT', 'ztgnn-default-salt-change-in-production')

def anonymize_ip(ip_address: str, salt: Optional[str] = None) -> str:
    """
    Anonymize IP address using SHA-256 hashing with salt
    
    Args:
        ip_address: Original IP address (e.g., "192.168.1.1")
        salt: Optional salt for hashing (uses environment variable if not provided)
    
    Returns:
        Anonymized IP as hex string (e.g., "anon_a3f2c1...")
    
    GDPR Compliance:
        - Pseudonymization technique per GDPR Article 4(5)
        - Consistent hashing allows correlation without revealing original IP
        - Salt stored separately from data for additional security
    """
    hash_salt = salt or ANONYMIZATION_SALT
    
    combined = f"{ip_address}:{hash_salt}"
    
    hash_object = hashlib.sha256(combined.encode('utf-8'))
    anonymized = hash_object.hexdigest()[:16]
    
    return f"anon_{anonymized}"

def anonymize_flow_data(flow_dict: dict) -> dict:
    """
    Anonymize all IP addresses in a network flow dictionary
    
    Args:
        flow_dict: Dictionary containing src_ip, dst_ip, and other flow data
    
    Returns:
        New dictionary with anonymized IP addresses
    """
    anonymized = flow_dict.copy()
    
    if 'src_ip' in anonymized:
        anonymized['src_ip'] = anonymize_ip(anonymized['src_ip'])
    
    if 'dst_ip' in anonymized:
        anonymized['dst_ip'] = anonymize_ip(anonymized['dst_ip'])
    
    if 'original_src_ip' not in anonymized and 'src_ip' in flow_dict:
        anonymized['_note'] = 'IPs anonymized for GDPR compliance'
    
    return anonymized

def should_anonymize() -> bool:
    """
    Check if data anonymization is enabled
    
    Returns:
        True if ANONYMIZE_DATA environment variable is set to 'true' (default: True)
    """
    env_value = os.environ.get('ANONYMIZE_DATA', 'true').lower().strip()
    return env_value in ('true', '1', 'yes', 'on')
