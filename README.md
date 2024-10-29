# vFLUX: High-Throughput FLUX Model Serving


A thread-free implementation that maximizes GPU utilization for serving FLUX model.

## Overview
This repository demonstrates the techniques described in [Maximizing PyTorch Throughput with FastAPI](https://jonathanc.net/blogs/maximizing_pytorch_throughput). It shows how to achieve high throughput by overlapping CPU work with GPU computation using asyncio and CUDA's asynchronous execution APIs.


## Requirements
This uses the FLUX.1-schnell model in bfloat16 precision, so a GPU with at least ~40GB GPU memory is required.

```bash
# Install PyTorch (tested on Stable (2.5.0) and nightly)
# pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# Install diffusers optimized to remove cuda sync
git submodule init && git submodule update
pip install ./diffusers/
```

## Usage

### Profile the server
```bash
python server.py
```
Note: First run will take a few minutes to compile the model. Subsequent runs will be faster.

In a different terminal, once the server has started:
```bash
python clients.py
```
This sends 4 parallel requests to the server and saves outputs to `outputs/`.
The server will save a trace file `server-trace.json` when stopped. View it using Chrome's trace viewer at `chrome://tracing`.

## Serve the model in production

When running the server with profiling enabled, the trace file will continue growing with each request. For production deployments or when handling many requests, disable profiling by setting the environment variable:
```bash
ENABLE_PROFILING=0 python server.py
```
Note: The trace is stored in memory and writing starts noly after server shutdown.