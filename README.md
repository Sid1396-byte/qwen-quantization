# Qwen-1.5B Quantization and vLLM Benchmark 🚀

This repository contains Python scripts to locally quantize the `Qwen/Qwen2.5-1.5B` model into **4-bit GPTQ** and **4-bit AWQ** formats. It also includes a benchmarking tool to test the throughput of these quantized models using **vLLM** and the ultra-fast Marlin kernel.

## 📁 Repository Structure

* `scripts/model_quantize.py`: Script to download the base Qwen model, apply WikiText-2 calibration, and save both GPTQ and AWQ 4-bit versions.
* `scripts/model_quantize_benchmark.py`: Script to load the quantized models into a vLLM engine, verify Marlin kernel activation, and benchmark token generation throughput.
* `requirements.txt`: Dependencies required to run the code.

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/Sid1396-byte/qwen-quantization.git](https://github.com/Sid1396-byte/qwen-quantization.git)
   cd qwen-quantization