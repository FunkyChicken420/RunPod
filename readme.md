# RunPod Windows 11 Compute Offloader

This repository contains a RunPod Serverless worker and Windows 11 client for intelligent compute offloading.

## 🚀 Features

- **Smart Offloading**: Automatically offloads tasks when system RAM > 80% or CPU > 85%
- **Multiple Task Types**: Text processing, data analysis, Python execution, file operations
- **Cost Efficient**: Only pay for compute time when RunPod is actually processing
- **Easy Integration**: Simple Python client for Windows 11

## 📁 Repository Structure

```
├── worker.py          # RunPod Serverless worker
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
├── runpod_client.py   # Windows 11 client script
└── README.md         # This file
```

## 🛠️ Supported Tasks

- **system_info**: Get system information
- **python_execution**: Execute Python code remotely
- **text_processing**: Analyze and transform text
- **data_analysis**: Process CSV data
- **file_processing**: Handle file operations
- **compression**: Compress data
- **batch_operations**: Run multiple tasks

## 🔧 Setup Instructions

### 1. RunPod Configuration
- **Queue vs Load Balancer**: Choose **Load Balancer**
- **CPU vs GPU**: Choose **CPU** for general compute tasks
- **CPU Configuration**: 
  - **Compute-Optimized**: 4-8 vCPUs for CPU-intensive tasks
  - **General Purpose**: 2-4 vCPUs for balanced workloads

### 2. Deploy to RunPod
1. Create new Serverless endpoint
2. Select "GitHub" as source
3. Connect to `FunkyChicken420/RunPod` repository
4. Choose CPU configuration based on your needs
5. Deploy and get your Endpoint ID

### 3. Windows 11 Setup
1. Install Python dependencies: `pip install requests psutil`
2. Get your RunPod API key from Account Settings
3. Update `runpod_client.py` with your credentials
4. Run: `python runpod_client.py`

## 💻 Usage Examples

```python
from runpod_client import RunPodOffloader

# Initialize client
client = RunPodOffloader("your-endpoint-id", "your-api-key")

# Test connection
client.test_system_info()

# Execute heavy Python code
result = client.execute_python_code("""
# Your intensive computation here
import numpy as np
data = np.random.rand(1000000)
result = np.fft.fft(data)
print("FFT computation completed")
""")

# Process text file
result = client.process_text_file("large_document.txt", operation="analyze")

# Analyze CSV data
result = client.analyze_csv_data("large_dataset.csv")

# Start automatic monitoring
client.start_monitoring(interval=10)
```

## 💰 Cost Optimization

- **CPU-only tasks**: ~$0.01-0.05 per minute
- **Smart thresholds**: Only offloads when system is stressed
- **Per-second billing**: Pay only for actual processing time
- **Automatic scaling**: Workers shut down when idle

## 🔍 Monitoring

The client automatically monitors:
- RAM usage percentage
- CPU usage percentage
- Automatic offloading triggers
- Real-time system status

## 🚨 Troubleshooting

**Connection Issues**:
- Verify Endpoint ID and API key
- Check RunPod dashboard for endpoint status
- Ensure endpoint is deployed and running

**High Costs**:
- Check monitoring thresholds
- Review task complexity
- Use CPU workers for non-GPU tasks

**Slow Performance**:
- Choose appropriate CPU configuration
- Optimize task data size
- Consider active workers for frequent use

## 📊 Performance Tips

1. **Choose Right Configuration**:
   - Light tasks: 2 vCPU General Purpose
   - Heavy computation: 8 vCPU Compute-Optimized

2. **Optimize Task Size**:
   - Break large tasks into smaller chunks
   - Use batch operations for multiple small tasks

3. **Monitor Costs**:
   - Set up cost alerts in RunPod dashboard
   - Review usage patterns regularly

## 🔧 Advanced Configuration

### Custom Task Types
Add new task types to `worker.py`:

```python
elif task_type == "custom_task":
    return handle_custom_task(input_data)
```

### Adjust Thresholds
Modify monitoring thresholds in client:

```python
client.ram_threshold = 90  # Higher threshold
client.cpu_threshold = 95  # Less sensitive
```

## 📝 License

MIT License - Feel free to modify and use for your projects.
