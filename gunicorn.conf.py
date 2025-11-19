"""
Gunicorn Configuration for Production Deployment
TLS 1.3 Ready - Replit handles TLS termination automatically
"""
import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"

workers = multiprocessing.cpu_count() * 2 + 1

worker_class = 'sync'

timeout = 120

keepalive = 5

max_requests = 1000
max_requests_jitter = 50

accesslog = '-'
errorlog = '-'
loglevel = 'info'

preload_app = True

forwarded_allow_ips = '*'
proxy_protocol = True
proxy_allow_ips = '*'
