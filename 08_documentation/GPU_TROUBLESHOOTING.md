# GPU Troubleshooting Guide

**Complete guide for diagnosing and fixing GPU-related issues during training**

## **Quick Reference**

| Error | Quick Fix | Full Guide |
|-------|-----------|------------|
| `ProcessGroupNCCL is only supported with GPUs, no GPUs found!` | Install NVIDIA drivers | [Driver Installation](#driver-installation) |
| `CUDA out of memory` | Reduce batch size | [Memory Issues](#memory-issues) |
| `NVIDIA driver version is insufficient` | Update drivers | [Driver Issues](#driver-issues) |
| `No CUDA-capable device is detected` | Reinstall CUDA toolkit | [CUDA Issues](#cuda-issues) |
| `fatal error: Python.h: No such file or directory` | Install python3-dev | [Python Development Headers](#python-development-headers) |
| Multi-GPU not working | Check detection | [Multi-GPU Issues](#multi-gpu-issues) |
| Low GPU utilization | Optimize settings | [Performance Issues](#performance-issues) |

## **Driver Installation**

### **"ProcessGroupNCCL is only supported with GPUs, no GPUs found!" Error**

This error occurs when NVIDIA drivers are not installed or GPUs are not properly detected.

**Error Message:**
```
ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!
```

**Diagnosis Steps:**
```bash
# 1. Check if nvidia-smi works
nvidia-smi

# 2. If command not found, check if GPUs are visible to the system
lspci | grep -i nvidia

# 3. Check if NVIDIA kernel modules are loaded
lsmod | grep nvidia
```

**Solution (Ubuntu/Debian):**
```bash
# 1. Update package list
sudo apt update

# 2. Install ubuntu-drivers-common
sudo apt install -y ubuntu-drivers-common

# 3. List available drivers
ubuntu-drivers list

# 4. Install recommended driver (usually the latest stable)
sudo apt install -y nvidia-driver-550-server

# 5. Install kernel headers (required for driver compilation)
sudo apt install -y linux-headers-$(uname -r)

# 6. Reboot to load the new driver
sudo reboot
```

**Verification after reboot:**
```bash
# Check GPU status
nvidia-smi

# Check CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test training
torchrun --nproc_per_node=2 04_training/train_model_slm.py
```

**Expected output for working setup:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A30                     Off |   000070B2:00:00.0 Off |                    0 |
| N/A   25C    P0             32W /  165W |       0MiB /  24576MiB |      2%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

CUDA available: True
GPU count: 2
```

## **Memory Issues**

### **"CUDA out of memory" Error**

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 24.00 GiB total capacity; 22.15 GiB already allocated; 1.85 GiB free; 22.15 GiB reserved in total by PyTorch)
```

**Quick Fixes:**
1. **Reduce batch size** in `config.py`:
   ```python
   "batch_size": 8  # Instead of 18
   ```

2. **Reduce max_length**:
   ```python
   "max_length": 512  # Instead of 1024
   ```

3. **Use single GPU** instead of multi-GPU:
   ```bash
   python 04_training/train_model_slm.py
   ```

4. **Enable gradient checkpointing**:
   ```python
   "gradient_checkpointing": True
   ```

5. **Enable memory-efficient attention**:
   ```python
   "use_flash_attention_2": True
   ```

## **Driver Issues**

### **"NVIDIA driver version is insufficient" Error**

**Error Message:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**
```bash
# Check current driver version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"

# Update to latest driver
sudo apt update
sudo apt install -y nvidia-driver-570-server
sudo reboot
```

### **Driver Installation Troubleshooting**

**If driver installation fails:**

1. **Check for conflicting drivers**:
   ```bash
   # Remove any existing NVIDIA packages
   sudo apt purge nvidia-* libnvidia-*
   sudo apt autoremove
   ```

2. **Disable secure boot** (if applicable):
   ```bash
   # Check if secure boot is enabled
   mokutil --sb-state
   
   # If enabled, disable in BIOS/UEFI settings
   ```

3. **Use specific driver version**:
   ```bash
   # List available versions
   ubuntu-drivers list
   
   # Install specific version
   sudo apt install -y nvidia-driver-535-server
   ```

4. **Check kernel compatibility**:
   ```bash
   # Ensure kernel headers match running kernel
   uname -r
   sudo apt install -y linux-headers-$(uname -r)
   ```

## **Python Development Headers**

### **"fatal error: Python.h: No such file or directory" Error**

This error occurs when `torch.compile` tries to compile CUDA extensions but Python development headers are missing.

**Error Message:**
```
/tmp/tmpcz7kphnd/cuda_utils.c:5:10: fatal error: Python.h: No such file or directory
    5 | #include <Python.h>
      |          ^~~~~~~~~
compilation terminated.
torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', ... 'Python.h' ... 'returned non-zero exit status 1.
```

**Why This Happens:**
- `torch.compile` uses Triton backend to compile optimized CUDA kernels
- Compilation requires Python C API headers (`Python.h`)
- These headers are provided by `python3-dev` package, not installed by default

**Solution (Ubuntu/Debian):**
```bash
# Install Python development headers
sudo apt update
sudo apt install python3-dev

# Verify installation
python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())"
```

**Solution (CentOS/RHEL):**
```bash
# Install Python development headers
sudo yum install python3-devel
# or for newer versions:
sudo dnf install python3-devel
```

**Solution (macOS):**
```bash
# Install Xcode command line tools (includes Python headers)
xcode-select --install
```

**Verification:**
```bash
# Test torch.compile works
python -c "import torch; x = torch.randn(10, 10).cuda(); compiled = torch.compile(lambda x: x @ x); print('torch.compile working!')"
```

**Alternative (Disable Compilation):**
If you can't install headers, disable `torch.compile` in `config.py`:
```python
"enable_compile": False  # Disable torch.compile
```
**Note:** This will make training 1.5-2x slower, so installing headers is recommended.

## **CUDA Issues**

### **"No CUDA-capable device is detected" Error**

**Error Message:**
```
RuntimeError: No CUDA-capable device is detected
```

**Solution:**
```bash
# Check CUDA installation
nvcc --version

# If not installed, install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit

# Reboot and test
sudo reboot
```

## **Multi-GPU Issues**

### **Multi-GPU Training Problems**

**Problem:** Training only uses one GPU instead of multiple

**Diagnosis:**
```bash
# Check if multiple GPUs are detected
nvidia-smi

# Check PyTorch GPU count
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test multi-GPU training
torchrun --nproc_per_node=2 04_training/train_model_slm.py
```

**Common Causes:**
1. **NCCL not installed**: `pip install nccl`
2. **Firewall blocking**: Check if ports are open
3. **CUDA version mismatch**: Ensure all GPUs use same CUDA version
4. **Driver issues**: Update NVIDIA drivers

**Solution:**
```bash
# Install NCCL for multi-GPU communication
pip install nccl

# Check if all GPUs are visible
nvidia-smi

# Test with specific GPU count
torchrun --nproc_per_node=2 04_training/train_model_slm.py

# If still using single GPU, check for errors in logs
tail -f 04_training/model_training.log
```

## **Process Management**

### **Killing Training Processes**

When training with `torchrun` on multiple GPUs, you need to kill all processes properly to avoid resource conflicts.

**Method 1: Kill by Process Name (Recommended)**
```bash
# Kill all Python processes running train_model.py
pkill -f "train_model.py"

# Or more specifically for torchrun
pkill -f "torchrun.*train_model.py"
```

**Method 2: Kill by Process ID**
```bash
# Find the process IDs first
ps aux | grep train_model.py

# Then kill them (replace with actual PIDs)
kill -9 <PID1> <PID2>
```

**Method 3: Kill All Python Processes (Nuclear option)**
```bash
# Only use this if you're sure no other important Python processes are running
pkill -f python
```

**Verification:**
```bash
# Check if any training processes are still running
ps aux | grep train_model.py
ps aux | grep torchrun

# Check GPU usage (should drop to near 0%)
nvidia-smi
```

**If Processes Don't Die:**
```bash
# Find and force kill
ps aux | grep train_model.py | awk '{print $2}' | xargs kill -9
```

## **Performance Issues**

### **Low GPU Utilization**

**Problem:** Training is slow or GPU utilization is low

**Diagnosis:**
```bash
# Check GPU utilization during training
nvidia-smi -l 1

# Should show high GPU utilization (~80-90%)
```

**Common Causes & Solutions:**

1. **Batch size too small**:
   ```python
   "batch_size": 24  # Increase from 8
   ```

2. **Data loading bottleneck**:
   ```python
   "dataloader_num_workers": 6  # Increase from 2
   "dataloader_pin_memory": True
   ```

3. **CPU bottleneck**:
   ```python
   "dataloader_persistent_workers": True
   ```

4. **Memory issues**:
   ```python
   "enable_amp": True  # Enable mixed precision
   "enable_compile": True  # Enable torch.compile
   ```

5. **Sequence length too short**:
   ```python
   "max_length": 1024  # Increase from 512
   ```


## **Verification Commands**

### **Complete GPU Setup Verification**

```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Check CUDA toolkit
nvcc --version

# 3. Check PyTorch CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 4. Test CUDA operations
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('CUDA operations working!')"

# 5. Test multi-GPU
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### **Expected Output for Working Setup**

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A30                     Off |   000070B2:00:00.0 Off |                    0 |
| N/A   25C    P0             32W /  165W |       0MiB /  24576MiB |      2%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 2
CUDA operations working!
Available GPUs: 2
GPU 0: NVIDIA A30
GPU 1: NVIDIA A30
```

## **Common Error Patterns**

### **Error: "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"**
- **Cause**: NVIDIA drivers not installed
- **Solution**: Follow [Driver Installation](#driver-installation) steps
- **Prevention**: Always verify `nvidia-smi` works before training

### **Error: "CUDA out of memory"**
- **Cause**: GPU memory insufficient for batch size
- **Solution**: Reduce batch size or max_length in config.py
- **Prevention**: Monitor GPU memory usage with `nvidia-smi`

### **Error: "NVIDIA driver version is insufficient"**
- **Cause**: Driver version too old for PyTorch CUDA version
- **Solution**: Update NVIDIA drivers to latest version
- **Prevention**: Check compatibility before installing PyTorch

### **Error: "No CUDA-capable device is detected"**
- **Cause**: CUDA toolkit not installed or GPU not detected
- **Solution**: Install CUDA toolkit and verify GPU detection
- **Prevention**: Use system with CUDA-compatible GPU

### **Error: "fatal error: Python.h: No such file or directory"**
- **Cause**: Python development headers missing for torch.compile
- **Solution**: Install python3-dev package
- **Prevention**: Include python3-dev in system setup

## **Performance Optimization**

### **GPU Memory Optimization**
```python
# In config.py - optimize for memory
"enable_amp": True,           # Mixed precision
"gradient_checkpointing": True,  # Trade compute for memory
"use_flash_attention_2": True,   # Memory-efficient attention
"batch_size": 8,              # Smaller batch size
"max_length": 512,            # Shorter sequences
```

### **GPU Speed Optimization**
```python
# In config.py - optimize for speed
"enable_amp": True,           # Mixed precision
"enable_compile": True,       # Torch compile
"dataloader_num_workers": 6,  # More data loading workers
"dataloader_pin_memory": True, # Faster CPU-GPU transfers
"batch_size": 24,             # Larger batch size
"max_length": 1024,           # Longer sequences
```

## **Troubleshooting Checklist**

Before reporting issues, check:

- [ ] `nvidia-smi` shows GPUs
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `python -c "import torch; print(torch.cuda.device_count())"` shows correct GPU count
- [ ] Driver version matches PyTorch CUDA version
- [ ] Sufficient GPU memory for batch size
- [ ] No conflicting NVIDIA packages
- [ ] Kernel headers installed and matching
- [ ] Python development headers installed (`python3-dev`)
- [ ] Secure boot disabled (if applicable)
- [ ] **Process Management:** All training processes killed before restart

## **Getting Help**

If you're still having issues:

1. **Check the logs**: `tail -f 04_training/model_training.log`
2. **Run verification commands**: See [Verification Commands](#verification-commands)
3. **Check system requirements**: Ensure your system meets the requirements
4. **Search existing issues**: Check if others have reported similar problems
5. **Provide details**: Include error messages, system info, and steps to reproduce

---

**Related Guides:**
- [Training Guide](TRAINING_GUIDE.md) - Complete training documentation
- [GPU Tuning Guide](GPU_TUNING.md) - Performance optimization
- [Training Quick Start](TRAINING_QUICK_START.md) - Quick setup guide
