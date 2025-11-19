"""
Vercel Serverless Function Entry Point
WSGI handler for Flask app
"""
from api_server import app

handler = app
