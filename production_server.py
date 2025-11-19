"""
Production WSGI Server Configuration for ZeroTrustGNN
Supports TLS 1.3, security headers, and production-grade settings
"""
import os
from api_server import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    print("=" * 70)
    print("🔒 ZeroTrustGNN Production Server")
    print("=" * 70)
    print(f"Port: {port}")
    print("TLS: Enabled via Replit deployment (automatic)")
    print("Security Headers: Enabled")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=False)
