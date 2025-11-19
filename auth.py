"""
Zero-Trust Authentication and Rate Limiting Module
Provides API key validation and request rate limiting
"""

import os
from functools import wraps
from flask import request, jsonify
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window
    Tracks requests per API key with configurable limits
    """
    def __init__(self, max_requests=10, window_minutes=1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.request_history = defaultdict(deque)
    
    def is_allowed(self, api_key):
        """
        Check if request is allowed for given API key
        Returns (allowed: bool, retry_after: int)
        """
        now = time.time()
        history = self.request_history[api_key]
        
        while history and history[0] < now - self.window_seconds:
            history.popleft()
        
        if len(history) < self.max_requests:
            history.append(now)
            return True, 0
        else:
            oldest = history[0]
            retry_after = int(oldest + self.window_seconds - now) + 1
            return False, retry_after
    
    def get_remaining(self, api_key):
        """Get remaining requests for API key"""
        now = time.time()
        history = self.request_history[api_key]
        
        while history and history[0] < now - self.window_seconds:
            history.popleft()
        
        return self.max_requests - len(history)

class APIKeyAuth:
    """
    API Key authentication manager
    Loads keys from environment and validates requests
    """
    def __init__(self):
        self.load_api_keys()
        self.rate_limiter = RateLimiter(
            max_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '10')),
            window_minutes=int(os.getenv('RATE_LIMIT_WINDOW', '1'))
        )
    
    def load_api_keys(self):
        """Load API keys from environment variable"""
        api_keys_str = os.getenv('API_KEYS', '')
        if api_keys_str:
            self.valid_keys = set(key.strip() for key in api_keys_str.split(',') if key.strip())
        else:
            self.valid_keys = set()
        
        if not self.valid_keys:
            print("⚠️  WARNING: No API keys configured. API is unprotected!")
            print("   Set API_KEYS environment variable with comma-separated keys")
    
    def validate_key(self, api_key):
        """Validate API key"""
        if not self.valid_keys:
            return True
        
        return api_key in self.valid_keys
    
    def extract_api_key(self, request_obj):
        """Extract API key from request headers"""
        return request_obj.headers.get('X-API-Key', '')

def require_api_key(f):
    """
    Decorator to require API key authentication on endpoints
    Usage: @require_api_key
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_manager = getattr(require_api_key, 'auth_manager', None)
        if not auth_manager:
            return jsonify({
                'success': False,
                'error': 'Authentication system not initialized',
                'error_code': 'AUTH_NOT_INITIALIZED'
            }), 500
        
        api_key = auth_manager.extract_api_key(request)
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Missing API key. Include X-API-Key header in your request.',
                'error_code': 'MISSING_API_KEY'
            }), 401
        
        if not auth_manager.validate_key(api_key):
            return jsonify({
                'success': False,
                'error': 'Invalid API key',
                'error_code': 'INVALID_API_KEY'
            }), 403
        
        allowed, retry_after = auth_manager.rate_limiter.is_allowed(api_key)
        
        if not allowed:
            return jsonify({
                'success': False,
                'error': f'Rate limit exceeded. Try again in {retry_after} seconds.',
                'error_code': 'RATE_LIMIT_EXCEEDED',
                'retry_after': retry_after
            }), 429
        
        remaining = auth_manager.rate_limiter.get_remaining(api_key)
        request.rate_limit_remaining = remaining
        
        return f(*args, **kwargs)
    
    return decorated_function

def init_auth(app):
    """
    Initialize authentication system for Flask app
    Call this in your app setup
    """
    auth_manager = APIKeyAuth()
    require_api_key.auth_manager = auth_manager
    
    @app.after_request
    def add_rate_limit_headers(response):
        """Add rate limit info to response headers"""
        if hasattr(request, 'rate_limit_remaining'):
            response.headers['X-RateLimit-Remaining'] = str(request.rate_limit_remaining)
            response.headers['X-RateLimit-Limit'] = str(auth_manager.rate_limiter.max_requests)
        return response
    
    print("="*70)
    print("🔒 AUTHENTICATION & RATE LIMITING")
    print("="*70)
    if auth_manager.valid_keys:
        print(f"  ✓ API Keys: {len(auth_manager.valid_keys)} key(s) configured")
    else:
        print("  ⚠ API Keys: NONE (API is open - not recommended for production)")
    print(f"  ✓ Rate Limit: {auth_manager.rate_limiter.max_requests} requests per {auth_manager.rate_limiter.window_seconds//60} minute(s)")
    print("="*70 + "\n")
    
    return auth_manager
