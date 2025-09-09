from flask import Flask, request, jsonify
import os
import json
import psutil
import subprocess
import base64
import zipfile
import io
from pathlib import Path

app = Flask(__name__)

# Your existing handler functions (keep all the same)
def get_system_info():
    # ... (same code as before)
    pass

def execute_python_code(input_data):
    # ... (same code as before)
    pass

# ... (all your other handler functions)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint for Load Balancer"""
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['POST'])
def handle_request():
    """Main request handler for Load Balancer"""
    try:
        data = request.get_json()
        input_data = data.get("input", {})
        
        # Call your existing handler logic
        result = handler({"input": input_data})
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handler(event):
    """Your existing handler logic (keep the same)"""
    # ... (all your existing handler code)
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    health_port = int(os.environ.get("PORT_HEALTH", 8081))
    
    # Start health check server in separate thread
    from threading import Thread
    
    def health_server():
        health_app = Flask(__name__)
        
        @health_app.route('/ping')
        def health_ping():
            return jsonify({"status": "healthy"}), 200
            
        health_app.run(host='0.0.0.0', port=health_port)
    
    health_thread = Thread(target=health_server)
    health_thread.daemon = True
    health_thread.start()
    
    # Start main server
    app.run(host='0.0.0.0', port=port)
