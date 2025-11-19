"""
Vercel Cron Job for GDPR Data Cleanup
Note: Vercel Cron requires Pro plan ($20/month)
For free tier, use external cron service like cron-job.org
"""
from flask import Flask, jsonify
from gdpr_consent import consent_manager
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/api/cleanup', methods=['GET', 'POST'])
def cleanup():
    """
    Cleanup endpoint for GDPR data retention
    Called daily by external cron service
    """
    try:
        result = consent_manager.cleanup_expired_data()
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': os.environ.get('VERCEL_TIMESTAMP', 'unknown')
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

handler = app
