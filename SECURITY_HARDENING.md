# Security Hardening & GDPR Compliance Guide

## Overview
ZeroTrustGNN implements comprehensive security hardening with TLS 1.3, data anonymization, and full GDPR compliance.

## 1. TLS 1.3 Configuration

### Development Environment
- Development server uses HTTP on localhost (port 8000)
- Frontend dev server on port 5000
- Not suitable for production use

### Production Deployment (Replit)
```bash
# Replit automatically provides TLS 1.3 when published
# No additional configuration needed
```

### Custom Production Deployment
```bash
# Install Gunicorn (already installed)
pip install gunicorn

# Run with Gunicorn
gunicorn -c gunicorn.conf.py api_server:app

# For custom TLS certificates:
gunicorn --certfile=cert.pem --keyfile=key.pem \
         --ssl-version=TLSv1_3 \
         -b 0.0.0.0:8443 api_server:app
```

### Security Headers
All API responses include:
- `Strict-Transport-Security`: Enforces HTTPS
- `X-Content-Type-Options`: Prevents MIME sniffing
- `X-Frame-Options`: Prevents clickjacking
- `X-XSS-Protection`: XSS filtering
- `Content-Security-Policy`: Restricts resource loading
- `Referrer-Policy`: Controls referrer information
- `Permissions-Policy`: Restricts browser features

## 2. Data Anonymization

### IP Address Pseudonymization
All IP addresses are anonymized using SHA-256 hashing with salt:

```python
# Automatic anonymization (enabled by default)
# Set environment variable to disable:
export ANONYMIZE_DATA=false  # Not recommended

# Configure anonymization salt (production):
export ANONYMIZATION_SALT=your-secret-salt-here
```

**GDPR Compliance**: Pseudonymization per Article 4(5)
- SHA-256 one-way hashing
- Consistent salt for correlation
- Non-reversible transformation
- Prevents re-identification

### Example Output
```json
{
  "source_ip": "anon_a3f2c1d8e5b4f7a9",
  "destination_ip": "anon_b9e7f6c5d4a3b2c1",
  "gdpr_notice": "IP addresses have been pseudonymized"
}
```

## 3. GDPR Compliance

### Privacy Policy Endpoint
```bash
GET /gdpr/privacy-policy
```

Returns comprehensive privacy policy including:
- Data collection practices
- Legal basis for processing
- User rights (GDPR Articles 15-22)
- Data retention policies
- Security measures

### Consent Management

#### Frontend Integration
The GDPR consent banner automatically:
- Appears on first visit
- Allows granular consent (logging, analytics)
- Stores consent in localStorage
- Sends consent to backend API

#### API Endpoints

**Record Consent:**
```bash
POST /gdpr/consent
Content-Type: application/json

{
  "session_id": "unique-session-id",
  "consent_given": true,
  "consent_types": ["logging", "analytics"]
}
```

**Check Consent (internal):**
```python
from gdpr_consent import consent_manager

if consent_manager.check_consent(session_id, 'logging'):
    # Log the event
    pass
```

### Right to be Forgotten (Article 17)

**Delete All User Data:**
```bash
POST /gdpr/delete-data
Content-Type: application/json

{
  "session_id": "unique-session-id"
}
```

Response:
```json
{
  "success": true,
  "message": "All data associated with your session has been permanently deleted",
  "deletion_summary": {
    "session_id": "unique-session-id",
    "deleted_at": "2025-11-19T19:30:00Z",
    "consents_deleted": 1,
    "audit_logs_deleted": 5
  },
  "gdpr_compliance": "Right to erasure (Article 17) fulfilled"
}
```

### Data Retention Policy

**Default Retention**: 30 days (configurable)

```bash
# Configure retention period (days)
export DATA_RETENTION_DAYS=30
```

**Automatic Cleanup:**
- Runs periodically to delete expired data
- Respects user consent withdrawal
- Permanent deletion (no soft delete)

**Get Retention Info:**
```bash
GET /gdpr/data-retention-info
```

### Audit Logging (Opt-in)

Audit logs only created with explicit user consent:

```python
# Only logs if user consented to 'logging'
consent_manager.log_audit_event(session_id, 'scan_request', {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'has_file': True
})
```

Audit log structure:
```json
{
  "session_id": "session_123",
  "event_type": "scan_request",
  "timestamp": "2025-11-19T19:30:00Z",
  "event_data": {
    "has_file": true
  },
  "retention_expires": "2025-12-19T19:30:00Z"
}
```

## 4. GDPR Data Subject Rights

### Implemented Rights

✅ **Right to Access (Article 15)**
- Users can view their data via API
- Session-based data retrieval

✅ **Right to Erasure (Article 17)**
- POST `/gdpr/delete-data` endpoint
- Permanent deletion within 24 hours
- Confirmation provided

✅ **Right to Data Portability (Article 20)**
- All data returned in JSON format
- Machine-readable structure

✅ **Right to Object (Article 21)**
- Decline consent banner
- Withdraw consent anytime

✅ **Data Minimization (Article 5(1)(c))**
- Only collect necessary data
- IP pseudonymization
- No PII collection

✅ **Storage Limitation (Article 5(1)(e))**
- 30-day automatic deletion
- Configurable retention period

### Not Implemented (Not Applicable)
❌ **Right to Rectification (Article 16)** - No user accounts, data is temporary
❌ **Right to Restrict Processing (Article 18)** - Processing is opt-in only

## 5. Security Checklist

### Before Production Deployment

- [ ] Set strong `ANONYMIZATION_SALT` environment variable (min 32 chars)
- [ ] Configure API keys in Replit Secrets (already done - 3 keys configured)
- [ ] Set `DATA_RETENTION_DAYS` to appropriate value (default: 30)
- [ ] Ensure `ANONYMIZE_DATA=true` (default, required for GDPR compliance)
- [ ] Enable HTTPS/TLS (automatic on Replit when published)
- [ ] Review and customize privacy policy contact email in gdpr_endpoints.py
- [ ] Test GDPR consent flow end-to-end
- [ ] Test right to be forgotten endpoint
- [ ] Verify IP anonymization by uploading PCAP and checking response IPs start with "anon_"
- [ ] Check security headers are present: `curl -I https://your-domain/health`
- [ ] Run security audit: `cd frontend && npm audit` and `pip check`
- [ ] Verify cleanup scheduler started: Check logs for "GDPR data retention cleanup scheduler started"
- [ ] Test with Gunicorn: `gunicorn -c gunicorn.conf.py api_server:app`

### Environment Variables (Production)

```bash
# Required
API_KEYS=key1,key2,key3  # In Replit Secrets

# Recommended
ANONYMIZATION_SALT=your-random-salt-min-32-chars
DATA_RETENTION_DAYS=30
ANONYMIZE_DATA=true

# Optional
CONSENT_DATA_DIR=./gdpr_data
PORT=8000
```

## 6. Compliance Certifications

### GDPR Compliance Level
- ✅ **Lawfulness**: Legitimate interest (Article 6(1)(f))
- ✅ **Transparency**: Clear privacy policy
- ✅ **Purpose Limitation**: Specific purpose stated
- ✅ **Data Minimization**: Only necessary data
- ✅ **Accuracy**: Data from primary source
- ✅ **Storage Limitation**: 30-day retention
- ✅ **Integrity & Confidentiality**: Encryption + access control
- ✅ **Accountability**: Audit logging

### Not Required
- ❌ Data Protection Officer (Article 37) - Processing scale doesn't meet threshold
- ❌ DPIA formal documentation - Low risk processing
- ❌ Data transfer mechanisms - No cross-border transfers

## 7. Incident Response

### Data Breach Protocol
If a security incident occurs:

1. **Immediate Actions**:
   - Isolate affected systems
   - Revoke compromised API keys
   - Enable read-only mode

2. **Notification Requirements** (GDPR Article 33):
   - Report to supervisory authority within 72 hours if high risk
   - Notify affected users if high risk to rights and freedoms

3. **Documentation**:
   - Log incident details
   - Document containment measures
   - Record notification actions

## 8. Testing Security Features

### Test IP Anonymization
```bash
# Upload PCAP file and verify IPs are anonymized
curl -X POST http://localhost:8000/scan \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"file":"base64-pcap-data"}' | jq '.alerts[0].source_ip'

# Should return: "anon_xxxxxxxxxxxxxxxx"
```

### Test GDPR Consent
```bash
# Record consent
curl -X POST http://localhost:8000/gdpr/consent \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test123","consent_given":true,"consent_types":["logging"]}'

# Delete data
curl -X POST http://localhost:8000/gdpr/delete-data \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test123"}'
```

### Test Security Headers
```bash
curl -I http://localhost:8000/health
# Check for:
# - Strict-Transport-Security
# - X-Content-Type-Options
# - X-Frame-Options
```

## 9. Additional Recommendations

### Future Enhancements
1. **Certificate Pinning** - For mobile clients
2. **Rate Limiting per IP** - Already have per-API-key limiting
3. **Request Signing** - HMAC for additional auth layer
4. **Encrypted Data at Rest** - File system encryption
5. **SOC 2 Compliance** - If expanding commercially
6. **Penetration Testing** - Annual security audits

### Monitoring
- Monitor failed authentication attempts
- Track unusual data deletion patterns
- Alert on excessive data retention
- Log security header delivery

## Contact
For security concerns: security@zerotrust-gnn.example
For privacy matters: privacy@zerotrust-gnn.example
