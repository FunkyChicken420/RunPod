from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import time
import subprocess
import os
import base64
import zipfile
import io
import platform
import psutil
from pathlib import Path
import uvicorn
from typing import Optional, Dict, Any, List

# Initialize FastAPI app
app = FastAPI(
    title="RunPod Load Balancer Worker",
    description="General purpose compute worker for offloading Windows 11 laptop tasks",
    version="1.0.0"
)

# =====================================
# Data Models for API requests
# =====================================

class SystemInfoRequest(BaseModel):
    task_type: str = "system_info"

class PythonExecutionRequest(BaseModel):
    task_type: str = "python_execution"
    code: str
    timeout: Optional[int] = 60

class FileProcessingRequest(BaseModel):
    task_type: str = "file_processing"
    files: List[Dict[str, str]]
    operation: str = "info"

class DataAnalysisRequest(BaseModel):
    task_type: str = "data_analysis"
    data: str
    data_type: str = "json"
    analysis: str = "summary"

class TextProcessingRequest(BaseModel):
    task_type: str = "text_processing"
    text: str
    operation: str = "analyze"
    transform: Optional[str] = None

class CompressionRequest(BaseModel):
    task_type: str = "compression"
    data: Dict[str, Any]
    type: str = "zip"

class BatchOperationsRequest(BaseModel):
    task_type: str = "batch_operations"
    operations: List[Dict[str, Any]]

class GeneralTaskRequest(BaseModel):
    task_type: str
    code: Optional[str] = None
    timeout: Optional[int] = 60
    text: Optional[str] = None
    operation: Optional[str] = None
    data: Optional[Any] = None
    files: Optional[List[Dict[str, str]]] = None

# =====================================
# Health Check Endpoints
# =====================================

@app.get("/ping")
async def health_check():
    """Health check endpoint for Load Balancer"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health")
async def health_status():
    """Alternative health check with more details"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "worker_info": {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RunPod Load Balancer Worker",
        "status": "ready",
        "endpoints": [
            "/ping",
            "/health", 
            "/system-info",
            "/execute-python",
            "/process-text",
            "/analyze-data",
            "/process-files",
            "/compress",
            "/batch",
            "/docs"
        ]
    }

# =====================================
# Task Processing Endpoints
# =====================================

@app.post("/system-info")
async def get_system_info():
    """Get system information about the RunPod worker"""
    try:
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "status": "success",
            "timestamp": time.time()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-python")
async def execute_python_code(request: PythonExecutionRequest):
    """Execute Python code safely"""
    try:
        if not request.code:
            raise HTTPException(status_code=400, detail="No code provided")
        
        # Create a temporary file
        with open("/tmp/exec_code.py", "w") as f:
            f.write(request.code)
        
        # Execute with timeout
        result = subprocess.run(
            ["python", "/tmp/exec_code.py"], 
            capture_output=True, 
            text=True, 
            timeout=request.timeout
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": request.timeout,
            "status": "success"
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail=f"Code execution timed out after {request.timeout} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-text")
async def process_text(request: TextProcessingRequest):
    """Process text data"""
    try:
        text = request.text
        operation = request.operation
        
        if operation == "analyze":
            words = text.split()
            sentences = text.split('.')
            
            return {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "char_count": len(text),
                "line_count": len(text.split('\n')),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "status": "success"
            }
            
        elif operation == "transform":
            transform_type = request.transform or "uppercase"
            
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
        
        else:
            raise HTTPException(status_code=400, detail="Unknown text operation")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-data")
async def analyze_data(request: DataAnalysisRequest):
    """Analyze CSV/JSON data"""
    try:
        data = request.data
        data_type = request.data_type
        analysis_type = request.analysis
        
        if data_type == "json":
            parsed_data = json.loads(data) if isinstance(data, str) else data
            return {"analysis": "completed", "data_preview": str(parsed_data)[:200], "status": "success"}
            
        elif data_type == "csv":
            import pandas as pd
            df = pd.read_csv(io.StringIO(data))
            
            if analysis_type == "summary":
                return {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                    "memory_usage": int(df.memory_usage(deep=True).sum()),
                    "null_counts": df.isnull().sum().to_dict(),
                    "status": "success"
                }
            
            elif analysis_type == "statistics":
                numeric_cols = df.select_dtypes(include=['number']).columns
                stats = df[numeric_cols].describe().to_dict()
                return {"statistics": stats, "status": "success"}
        
        return {"analysis": "completed", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-files")
async def process_files(request: FileProcessingRequest):
    """Process files sent as base64 encoded data"""
    try:
        files = request.files
        operation = request.operation
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress")
async def compress_data(request: CompressionRequest):
    """Compress files or data"""
    try:
        data = request.data
        compression_type = request.type
        
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
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported compression type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_operations(request: BatchOperationsRequest):
    """Handle batch operations"""
    try:
        operations = request.operations
        results = []
        
        for i, operation in enumerate(operations):
            try:
                # Process each operation by calling the appropriate endpoint internally
                task_type = operation.get("task_type")
                
                if task_type == "system_info":
                    result = await get_system_info()
                elif task_type == "text_processing":
                    text_req = TextProcessingRequest(**operation)
                    result = await process_text(text_req)
                # Add more task types as needed
                else:
                    result = {"error": f"Unsupported task type in batch: {task_type}"}
                
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# General Task Endpoint (for compatibility)
# =====================================

@app.post("/task")
async def general_task(request: GeneralTaskRequest):
    """General task endpoint for backward compatibility"""
    try:
        task_type = request.task_type
        
        if task_type == "system_info":
            return await get_system_info()
            
        elif task_type == "python_execution":
            if not request.code:
                raise HTTPException(status_code=400, detail="No code provided for python_execution")
            python_req = PythonExecutionRequest(code=request.code, timeout=request.timeout or 60)
            return await execute_python_code(python_req)
            
        elif task_type == "text_processing":
            if not request.text:
                raise HTTPException(status_code=400, detail="No text provided for text_processing")
            text_req = TextProcessingRequest(text=request.text, operation=request.operation or "analyze")
            return await process_text(text_req)
            
        elif task_type == "data_analysis":
            if not request.data:
                raise HTTPException(status_code=400, detail="No data provided for data_analysis")
            data_req = DataAnalysisRequest(data=request.data, data_type="csv", analysis="summary")
            return await analyze_data(data_req)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# Additional Endpoints for Testing
# =====================================

@app.get("/stats")
async def get_stats():
    """Get worker statistics"""
    return {
        "uptime": time.time(),
        "requests_processed": "unknown",  # You could add a counter
        "worker_status": "active",
        "timestamp": time.time()
    }

@app.post("/generate")
async def generate_text(request: Request):
    """Generate text endpoint (common for AI models)"""
    try:
        body = await request.json()
        prompt = body.get("prompt", "Hello world")
        
        # Simple text generation (you could integrate actual AI models here)
        generated_text = f"Generated response for: {prompt}"
        
        return {
            "generated_text": generated_text,
            "prompt": prompt,
            "timestamp": time.time(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: Request):
    """Prediction endpoint (common for ML models)"""
    try:
        body = await request.json()
        input_data = body.get("input", {})
        
        # Simple prediction logic (you could integrate actual ML models here)
        prediction = {
            "prediction": "sample_prediction",
            "confidence": 0.95,
            "input_received": input_data,
            "timestamp": time.time()
        }
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# Server startup
# =====================================

if __name__ == "__main__":
    # Get port from environment variable (RunPod sets PORT)
    port = int(os.environ.get("PORT", 80))
    host = "0.0.0.0"
    
    print(f"🚀 Starting RunPod Load Balancer Worker on {host}:{port}")
    print(f"📋 Available endpoints:")
    print(f"   Health: /ping, /health")
    print(f"   Tasks: /system-info, /execute-python, /process-text")
    print(f"   Data: /analyze-data, /process-files, /compress")
    print(f"   Batch: /batch")
    print(f"   General: /task")
    print(f"   AI: /generate, /predict")
    print(f"   Docs: /docs")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
