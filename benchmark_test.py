# Basic lib
import os
import time
import yaml

# DL lib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.transforms as transforms

# Local lib
from models.utils import Model


def image_preprocess(image_path):
    # Load image by PIL
    image = Image.open(image_path).convert('RGB')
    
    # Pre-processing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dim
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def gpu_inference(model, input_tensor, num_warmup_iters=10, num_iters=100):
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    # print(f"Warmup iteration {num_warmup_iters}")
    with torch.no_grad():
        for _ in range(num_warmup_iters):
            _ = model(input_tensor)
    
    # Sync the asynchronous execution of cpu and gpu, cpu wait for the gpu to complete all tasks
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # GPU inference
    # print(f"Inference iteration {num_iters}")
    inference_times = []
    with torch.no_grad():
        for i in range(num_iters):
            time_s = time.time()
            _ = model(input_tensor)  # model inference
            torch.cuda.synchronize() if device.type == 'cuda' else None
            time_e = time.time()
            inference_times.append((time_e - time_s) * 1000)  # ms

    # Inference time statistics
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\n=== Inference Time Statistics ===")
    print(f"Avg: {avg_time:.2f} ms")
    print(f"Min: {min_time:.2f} ms")
    print(f"Max: {max_time:.2f} ms")
    print(f"Std: {std_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'fps': 1000/avg_time
    }


def main():
    print("=== GPU PyTorch Inference Benchmark Test ===")
    
    # Test data
    image_path = "Tocal.jpg"
    image_tensor = image_preprocess(image_path)

    # Models
    model_names_list = ["resnet50", "resnet101", "vgg16", "vgg19"]
    for model_name in model_names_list:
        model_cfg_path = os.path.join("configs", f"{model_name}.yaml")
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)

        # Model summary
        model_summary = summary(model, input_size=image_tensor.shape, depth=3, verbose=1)
        # total_params = sum(p.numel() for p in model.parameters())
        
        # Inference
        benchmark_results = gpu_inference(model, image_tensor)

        # Save report
        with open(f"results/{model_name}_report.txt", "w") as f:
            f.write(str(model_summary))
            f.write("\n")
            f.write("Device: GPU\n")
            f.write("Framework: PyTorch\n")
            for key, value in benchmark_results.items():
                f.write(f"{key}: {value:.2f}\n")


if __name__ == "__main__":
    main()