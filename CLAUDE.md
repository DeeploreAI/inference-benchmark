# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning inference benchmark repository that compares performance between PyTorch and ONNX Runtime on GPU. The codebase benchmarks popular computer vision models (ResNet50/101, VGG16/19) for ImageNet classification tasks and includes hardware performance analysis tools.

## Core Architecture

### Model System
- **Configuration-driven models**: Models are defined via YAML configs in `configs/` and built dynamically using `models/utils.py:Model`
- **Modular components**: Custom neural network modules in `models/common.py` (Conv, ResNet layers, VGG blocks, etc.)
- **Pretrained weights**: Located in `pretrained/` directory with corresponding `.pth` files

### Benchmark Framework
- **PyTorch benchmarking**: `pytorch_gpu_benchmark.py` - Direct PyTorch inference timing
- **ONNX benchmarking**: `onnx_gpu_benchmark.py` - Exports to ONNX and benchmarks ONNX Runtime
- **Hardware analysis**: `test/server_compute_time.py` - Theoretical performance calculations based on hardware specs

### Key Data Flow
1. Load model config from `configs/{model_name}.yaml`
2. Build model using `Model(config)` class
3. Load pretrained weights from `pretrained/{model_name}_pretrained.pth`
4. Run inference benchmarks with warmup iterations
5. Generate performance reports in `results/`

## Running Commands

### Basic Benchmarking
```bash
# Run PyTorch GPU benchmark for all models
python pytorch_gpu_benchmark.py

# Run ONNX Runtime GPU benchmark for all models  
python onnx_gpu_benchmark.py

# Run hardware performance analysis
cd test/
python server_compute_time.py --config server_config.yaml
```

### Hardware Analysis Options
```bash
# Custom model and batch size
python server_compute_time.py --model-name resnet50 --batch-size 64

# Custom hardware specs
python server_compute_time.py --num-gpus 4 --gpu-vram 24 --sync-mode NVLINK
```

## File Structure Patterns

- **configs/**: YAML model definitions following `[n_repeat, module_name, args]` format
- **models/**: Neural network module definitions and model builder utilities
- **pretrained/**: Model weights and metadata files  
- **onnx/**: Generated ONNX model files (created during benchmarking)
- **results/**: Benchmark reports and performance summaries
- **test/**: Hardware analysis and testing utilities

## Development Notes

### Adding New Models
1. Create YAML config in `configs/` following existing ResNet/VGG patterns
2. Ensure all required modules exist in `models/common.py`
3. Add pretrained weights to `pretrained/` directory
4. Update model lists in benchmark scripts if needed

### Model Configuration Format
```yaml
backbone:
  [[n_repeat, module_name, [args...]]]
head:
  [[n_repeat, module_name, [args...]]]
```

### Dependencies
- PyTorch with CUDA support
- ONNX and ONNX Runtime with GPU providers
- torchinfo for model analysis
- PIL, numpy, PyYAML for data processing

### Testing
Run individual benchmark scripts to test specific components. No formal test framework is configured.