"""
GDPR Consent Management System
Handles user consent, audit logging, data retention, and right to be forgotten
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

CONSENT_DATA_DIR = os.environ.get('CONSENT_DATA_DIR', './gdpr_data')
DATA_RETENTION_DAYS = int(os.environ.get('DATA_RETENTION_DAYS', '30'))

Path(CONSENT_DATA_DIR).mkdir(parents=True, exist_ok=True)

class GDPRConsentManager:
    """Manages GDPR consent and audit logging"""
    
    def __init__(self):
        self.consent_file = os.path.join(CONSENT_DATA_DIR, 'consents.json')
        self.audit_log_file = os.path.join(CONSENT_DATA_DIR, 'audit_log.json')
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Create consent and audit log files if they don't exist"""
        for file_path in [self.consent_file, self.audit_log_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def record_consent(self, session_id: str, consent_given: bool, consent_types: List[str]) -> Dict:
        """
        Record user consent decision
        
        Args:
            session_id: Unique session identifier
            consent_given: Whether user gave consent
            consent_types: Types of consent (e.g., ['analytics', 'logging'])
        
        Returns:
            Consent record dictionary
        """
        consent_record = {
            'session_id': session_id,
            'consent_given': consent_given,
            'consent_types': consent_types,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'ip_anonymized': True,
            'data_retention_days': DATA_RETENTION_DAYS
        }
        
        consents = self._read_consents()
        consents.append(consent_record)
        self._write_consents(consents)
        
        return consent_record
    
    def check_consent(self, session_id: str, consent_type: str = 'logging') -> bool:
        """
        Check if user has given specific consent
        
        Args:
            session_id: Unique session identifier
            consent_type: Type of consent to check
        
        Returns:
            True if consent was given
        """
        consents = self._read_consents()
        
        for consent in reversed(consents):
            if consent['session_id'] == session_id:
                return consent.get('consent_given', False) and consent_type in consent.get('consent_types', [])
        
        return False
    
    def log_audit_event(self, session_id: str, event_type: str, event_data: Dict) -> None:
        """
        Log audit event if user has given consent
        
        Args:
            session_id: Unique session identifier
            event_type: Type of event (e.g., 'scan_request', 'api_call')
            event_data: Event-specific data
        """
        if not self.check_consent(session_id, 'logging'):
            return
        
        audit_event = {
            'session_id': session_id,
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_data': event_data,
            'retention_expires': (datetime.utcnow() + timedelta(days=DATA_RETENTION_DAYS)).isoformat() + 'Z'
        }
        
        audit_logs = self._read_audit_logs()
        audit_logs.append(audit_event)
        self._write_audit_logs(audit_logs)
    
    def delete_user_data(self, session_id: str) -> Dict:
        """
        Right to be forgotten: Delete all data for a session ID
        
        Args:
            session_id: Unique session identifier to delete
        
        Returns:
            Deletion summary
        """
        consents = self._read_consents()
        audit_logs = self._read_audit_logs()
        
        consents_before = len(consents)
        audit_logs_before = len(audit_logs)
        
        consents = [c for c in consents if c['session_id'] != session_id]
        audit_logs = [a for a in audit_logs if a['session_id'] != session_id]
        
        self._write_consents(consents)
        self._write_audit_logs(audit_logs)
        
        deletion_log = {
            'session_id': session_id,
            'deleted_at': datetime.utcnow().isoformat() + 'Z',
            'consents_deleted': consents_before - len(consents),
            'audit_logs_deleted': audit_logs_before - len(audit_logs)
        }
        
        deletion_log_file = os.path.join(CONSENT_DATA_DIR, 'deletions.json')
        if os.path.exists(deletion_log_file):
            with open(deletion_log_file, 'r') as f:
                deletions = json.load(f)
        else:
            deletions = []
        
        deletions.append(deletion_log)
        with open(deletion_log_file, 'w') as f:
            json.dump(deletions, f, indent=2)
        
        return deletion_log
    
    def cleanup_expired_data(self) -> Dict:
        """
        Clean up data older than retention period
        
        Returns:
            Cleanup summary
        """
        cutoff_date = datetime.utcnow() - timedelta(days=DATA_RETENTION_DAYS)
        cutoff_str = cutoff_date.isoformat() + 'Z'
        
        audit_logs = self._read_audit_logs()
        consents = self._read_consents()
        
        logs_before = len(audit_logs)
        consents_before = len(consents)
        
        audit_logs = [a for a in audit_logs if a.get('retention_expires', '9999') > cutoff_str]
        consents = [c for c in consents if c.get('timestamp', '9999') > cutoff_str]
        
        self._write_audit_logs(audit_logs)
        self._write_consents(consents)
        
        return {
            'cleaned_at': datetime.utcnow().isoformat() + 'Z',
            'retention_days': DATA_RETENTION_DAYS,
            'audit_logs_deleted': logs_before - len(audit_logs),
            'consents_deleted': consents_before - len(consents)
        }
    
    def _read_consents(self) -> List[Dict]:
        """Read all consent records"""
        with open(self.consent_file, 'r') as f:
            return json.load(f)
    
    def _write_consents(self, consents: List[Dict]) -> None:
        """Write consent records"""
        with open(self.consent_file, 'w') as f:
            json.dump(consents, f, indent=2)
    
    def _read_audit_logs(self) -> List[Dict]:
        """Read all audit logs"""
        with open(self.audit_log_file, 'r') as f:
            return json.load(f)
    
    def _write_audit_logs(self, logs: List[Dict]) -> None:
        """Write audit logs"""
        with open(self.audit_log_file, 'w') as f:
            json.dump(logs, f, indent=2)

consent_manager = GDPRConsentManager()
