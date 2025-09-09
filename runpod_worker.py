import runpod
import json
import time
import subprocess
import os
import base64
import zipfile
import io
from pathlib import Path

def handler(event):
    """
    General purpose compute handler for offloading Windows 11 laptop tasks
    """
    try:
        input_data = event.get("input", {})
        task_type = input_data.get("task_type")
        
        print(f"Processing task: {task_type}")
        
        if task_type == "system_info":
            return get_system_info()
            
        elif task_type == "python_execution":
            return execute_python_code(input_data)
            
        elif task_type == "file_processing":
            return process_files(input_data)
            
        elif task_type == "data_analysis":
            return analyze_data(input_data)
            
        elif task_type == "compression":
            return compress_data(input_data)
            
        elif task_type == "text_processing":
            return process_text(input_data)
            
        elif task_type == "batch_operations":
            return batch_operations(input_data)
            
        else:
            return {"error": f"Unknown task type: {task_type}"}
            
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def get_system_info():
    """Return system information about the RunPod worker"""
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        "status": "success"
    }
    
    # Check for GPU if available
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0].split(', ')
            info["gpu_name"] = gpu_info[0]
            info["gpu_memory_mb"] = int(gpu_info[1])
    except:
        info["gpu_name"] = "No GPU or CUDA not available"
    
    return info

def execute_python_code(input_data):
    """Execute Python code safely"""
    code = input_data.get("code", "")
    timeout = input_data.get("timeout", 60)
    
    if not code:
        return {"error": "No code provided"}
    
    try:
        # Create a temporary file
        with open("/tmp/exec_code.py", "w") as f:
            f.write(code)
        
        # Execute with timeout
        result = subprocess.run(
            ["python", "/tmp/exec_code.py"], 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "status": "success"
        }
        
    except subprocess.TimeoutExpired:
        return {"error": f"Code execution timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}

def process_files(input_data):
    """Process files sent as base64 encoded data"""
    files = input_data.get("files", [])
    operation = input_data.get("operation", "info")
    
    results = []
    
    for file_info in files:
        filename = file_info.get("filename")
        file_data = file_info.get("data")  # base64 encoded
        
        try:
            # Decode file data
            decoded_data = base64.b64decode(file_data)
            
            if operation == "info":
                results.append({
                    "filename": filename,
                    "size_bytes": len(decoded_data),
                    "size_mb": round(len(decoded_data) / (1024**2), 2)
                })
                
            elif operation == "word_count" and filename.endswith('.txt'):
                content = decoded_data.decode('utf-8')
                word_count = len(content.split())
                line_count = len(content.split('\n'))
                char_count = len(content)
                
                results.append({
                    "filename": filename,
                    "word_count": word_count,
                    "line_count": line_count,
                    "char_count": char_count
                })
                
        except Exception as e:
            results.append({
                "filename": filename,
                "error": str(e)
            })
    
    return {"results": results, "status": "success"}

def analyze_data(input_data):
    """Analyze CSV/JSON data"""
    data = input_data.get("data")
    data_type = input_data.get("data_type", "json")
    analysis_type = input_data.get("analysis", "summary")
    
    try:
        if data_type == "json":
            import json
            parsed_data = json.loads(data) if isinstance(data, str) else data
            
        elif data_type == "csv":
            import pandas as pd
            import io
            df = pd.read_csv(io.StringIO(data))
            
            if analysis_type == "summary":
                return {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "status": "success"
                }
            
            elif analysis_type == "statistics":
                numeric_cols = df.select_dtypes(include=['number']).columns
                stats = df[numeric_cols].describe().to_dict()
                return {"statistics": stats, "status": "success"}
        
        return {"analysis": "completed", "status": "success"}
        
    except Exception as e:
        return {"error": str(e)}

def compress_data(input_data):
    """Compress files or data"""
    data = input_data.get("data")
    compression_type = input_data.get("type", "zip")
    
    if compression_type == "zip":
        # Create zip in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if isinstance(data, dict):
                for filename, file_content in data.items():
                    if isinstance(file_content, str):
                        zip_file.writestr(filename, file_content.encode())
                    else:
                        zip_file.writestr(filename, file_content)
        
        zip_buffer.seek(0)
        compressed_data = base64.b64encode(zip_buffer.getvalue()).decode()
        
        return {
            "compressed_data": compressed_data,
            "original_size": len(str(data)),
            "compressed_size": len(zip_buffer.getvalue()),
            "status": "success"
        }
    
    return {"error": "Unsupported compression type"}

def process_text(input_data):
    """Process text data"""
    text = input_data.get("text", "")
    operation = input_data.get("operation", "analyze")
    
    if operation == "analyze":
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "char_count": len(text),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "status": "success"
        }
        
    elif operation == "transform":
        transform_type = input_data.get("transform", "uppercase")
        
        if transform_type == "uppercase":
            result = text.upper()
        elif transform_type == "lowercase":
            result = text.lower()
        elif transform_type == "reverse":
            result = text[::-1]
        elif transform_type == "word_reverse":
            result = ' '.join(word[::-1] for word in text.split())
        else:
            result = text
            
        return {"transformed_text": result, "status": "success"}
    
    return {"error": "Unknown text operation"}

def batch_operations(input_data):
    """Handle batch operations"""
    operations = input_data.get("operations", [])
    results = []
    
    for i, operation in enumerate(operations):
        try:
            # Process each operation
            result = handler({"input": operation})
            results.append({
                "operation_index": i,
                "result": result
            })
        except Exception as e:
            results.append({
                "operation_index": i,
                "error": str(e)
            })
    
    return {"batch_results": results, "status": "success"}

if __name__ == "__main__":
    print("Starting RunPod Serverless Worker...")
    runpod.serverless.start({"handler": handler})
