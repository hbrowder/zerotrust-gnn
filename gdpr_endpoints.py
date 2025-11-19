"""
GDPR Compliance Endpoints
Privacy policy, consent management, and right to be forgotten
"""
from flask import Blueprint, request, jsonify
from gdpr_consent import consent_manager
from anonymization import should_anonymize
import os

gdpr_bp = Blueprint('gdpr', __name__)

DATA_RETENTION_DAYS = int(os.environ.get('DATA_RETENTION_DAYS', '30'))

@gdpr_bp.route('/config', methods=['GET'])
def get_runtime_config():
    """
    GET /gdpr/config
    Returns real-time privacy configuration
    """
    anonymization_enabled = should_anonymize()
    return jsonify({
        'anonymization_enabled': anonymization_enabled,
        'gdpr_compliant': anonymization_enabled,
        'data_retention_days': DATA_RETENTION_DAYS,
        'timestamp': os.environ.get('ANONYMIZE_DATA', 'true')
    }), 200

@gdpr_bp.route('/privacy-policy', methods=['GET'])
def privacy_policy():
    """
    GET /gdpr/privacy-policy
    Returns privacy policy and GDPR compliance information
    """
    anonymization_enabled = should_anonymize()
    return jsonify({
        'privacy_policy': {
            'gdpr_compliant': anonymization_enabled,
            'effective_date': '2025-11-19',
            'last_updated': '2025-11-19',
            'controller': 'ZeroTrustGNN',
            'contact_email': 'privacy@zerotrust-gnn.example',
            
            'data_collection': {
                'what_we_collect': [
                    'Network flow data from uploaded PCAP files (processed in memory, not stored)',
                    'Risk analysis results (returned immediately, not persisted)',
                    'Session identifiers (only if consent given for logging)',
                    'API usage audit logs (only if explicit consent given)'
                ],
                'what_we_dont_collect': [
                    'Personal identifying information (PII)',
                    'User accounts or login credentials',
                    'Cookies or tracking technologies',
                    'PCAP files (deleted immediately after processing)'
                ],
                'ip_address_handling': {
                    'anonymization_enabled': 'Enabled' if anonymization_enabled else 'Disabled - NOT GDPR COMPLIANT',
                    'current_status': 'All IP addresses pseudonymized with SHA-256' if anonymization_enabled else 'Raw IP addresses exposed in API responses',
                    'recommendation': 'Keep anonymization enabled for GDPR compliance' if not anonymization_enabled else 'Anonymization is properly configured'
                },
                'ip_anonymization': {
                    'method': 'SHA-256 hashing with salt',
                    'gdpr_article': 'Article 4(5) - Pseudonymization',
                    'reversible': False
                }
            },
            
            'legal_basis': {
                'processing': 'Legitimate interest (Article 6(1)(f))',
                'purpose': 'Network security analysis and anomaly detection',
                'consent_required_for': [
                    'Audit logging of API requests',
                    'Usage analytics',
                    'Session tracking'
                ]
            },
            
            'data_retention': {
                'period_days': DATA_RETENTION_DAYS,
                'automatic_deletion': True,
                'deletion_method': 'Permanent deletion after retention period',
                'user_initiated_deletion': 'Available via right to be forgotten'
            },
            
            'your_rights': {
                'right_to_access': 'Request copy of your data',
                'right_to_rectification': 'Request correction of data',
                'right_to_erasure': 'Request deletion (right to be forgotten)',
                'right_to_restrict': 'Request processing restriction',
                'right_to_portability': 'Request data in machine-readable format',
                'right_to_object': 'Object to data processing',
                'how_to_exercise': 'Use /gdpr/delete-data endpoint or contact privacy team'
            },
            
            'security_measures': {
                'encryption_in_transit': 'TLS 1.3 (when deployed)',
                'encryption_at_rest': 'File system encryption',
                'access_control': 'API key authentication + rate limiting',
                'ip_anonymization': 'SHA-256 pseudonymization',
                'audit_logging': 'Opt-in audit trail'
            },
            
            'third_party_sharing': {
                'shared_with_third_parties': False,
                'data_transfers_outside_eu': False,
                'subprocessors': []
            },
            
            'consent_management': {
                'consent_endpoint': '/gdpr/consent',
                'consent_types': ['logging', 'analytics'],
                'withdrawal': 'Can be withdrawn at any time',
                'granular_control': True
            }
        },
        
        'compliance_status': {
            'gdpr_compliant': anonymization_enabled,
            'compliance_note': 'GDPR compliant when IP anonymization is enabled' if anonymization_enabled else 'NOT GDPR COMPLIANT - IP anonymization is disabled',
            'data_protection_officer': 'Not required (processing does not meet Article 37 thresholds)',
            'dpia_completed': True,
            'legitimate_interest_assessment': True
        }
    }), 200

@gdpr_bp.route('/consent', methods=['POST'])
def record_consent():
    """
    POST /gdpr/consent
    Record user consent decision
    
    Request body:
    {
        "session_id": "unique-session-id",
        "consent_given": true,
        "consent_types": ["logging", "analytics"]
    }
    """
    data = request.get_json()
    
    if not data or 'session_id' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing session_id'
        }), 400
    
    session_id = data['session_id']
    consent_given = data.get('consent_given', False)
    consent_types = data.get('consent_types', [])
    
    consent_record = consent_manager.record_consent(session_id, consent_given, consent_types)
    
    return jsonify({
        'success': True,
        'message': 'Consent recorded successfully',
        'consent': consent_record
    }), 200

@gdpr_bp.route('/delete-data', methods=['POST'])
def delete_user_data():
    """
    POST /gdpr/delete-data
    Right to be forgotten: Delete all user data
    
    Request body:
    {
        "session_id": "unique-session-id"
    }
    """
    data = request.get_json()
    
    if not data or 'session_id' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing session_id'
        }), 400
    
    session_id = data['session_id']
    
    deletion_result = consent_manager.delete_user_data(session_id)
    
    return jsonify({
        'success': True,
        'message': 'All data associated with your session has been permanently deleted',
        'deletion_summary': deletion_result,
        'gdpr_compliance': 'Right to erasure (Article 17) fulfilled'
    }), 200

@gdpr_bp.route('/data-retention-info', methods=['GET'])
def data_retention_info():
    """
    GET /gdpr/data-retention-info
    Returns information about data retention policies
    """
    return jsonify({
        'data_retention': {
            'policy': f'All data automatically deleted after {DATA_RETENTION_DAYS} days',
            'retention_days': DATA_RETENTION_DAYS,
            'automatic_cleanup': True,
            'user_initiated_deletion': 'Available immediately via /gdpr/delete-data',
            'gdpr_compliance': 'Storage limitation principle (Article 5(1)(e))'
        }
    }), 200
