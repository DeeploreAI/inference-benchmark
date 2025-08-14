# Basic lib
import json
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
import torchvision.models as models

# Local lib
from models.utils import Model
from models.utils import set_deterministic


def load_pretrained_weights(model, pretrained_weight_path):
    pretrained_dict = torch.load(pretrained_weight_path)
    model_dict = model.state_dict()

    # Check the model dict keys
    for v1, v2 in zip(pretrained_dict.values(), model_dict.values()):
        if v1.shape == v2.shape:
            continue
        else:
            raise ValueError("Pretrained params and current params do not have the same shape.")
        # param_name = k_1.split('.')[-1]
        # if "conv" in k_1.lower():
        #     layer_name = "conv"
        # elif "bn" in k_1.lower():
        #     layer_name = "bn"
        # elif

    # Rename the pretrained dict keys
    pretrained_dict = {k2: v1 for k2, v1 in zip(model_dict.keys(), pretrained_dict.values())}

    # Overwrite model dict
    model_dict.update(pretrained_dict)

    # Load the pretrained model dict
    model.load_state_dict(model_dict)
    return model


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


def gpu_inference_time_test(model, input_tensor, num_warmup_iters=10, num_iters=100):
    # Set to evaluation mode
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup_iters):
            _ = model(input_tensor)
    
    # Sync the asynchronous execution of cpu and gpu, cpu wait for the gpu to complete all tasks
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # GPU inference
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
    
    print(f"\n=== GPU PyTorch Inference Time ===")
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


def imagenet_classify_test(model, input_tensor, imagenet_classes_name):
    """
    Test image classification using ImageNet pretrained weights.
    Tests all four models: resnet50, resnet101, vgg16, vgg19.
    """
    # Set to evaluation mode
    model.eval()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Perform classification
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
        print(f"Top-5 Predictions:")

        for i in range(5):
            class_idx = top5_indices[0, i].item()
            prob = top5_prob[0, i].item()
            class_name = imagenet_classes_name[class_idx]
            print(f"{i:2d}. {class_name:<30} {prob:.4f}")

        # Show top prediction
        top1_idx = top5_indices[0, 0].item()
        top1_prob = top5_prob[0, 0].item()
        top1_class = imagenet_classes_name[top1_idx]

        print(f"\n=== ImageNet Classification ===")
        print(f"\n🎯 Predicted Class: {top1_class}")
        print(f"   Confidence: {top1_prob:.4f}")

        return {
            "pred_class": top1_class,
            "confidence": top1_prob,
        }


def main():
    print("=== GPU PyTorch Inference Benchmark Test ===")
    
    # Test data
    image_path = "Tocal.jpg"
    image_tensor = image_preprocess(image_path)

    # Load ImageNet class names
    with open("configs/imagenet_class_index.json", "r") as f:
        imagenet_classes = json.load(f)
    imagenet_classes_name = [imagenet_classes[f"{i}"][1] for i in range(len(imagenet_classes))]

    # Models
    model_names_list = ["resnet50", "resnet101", "vgg16", "vgg19"]
    for model_name in model_names_list:
        print(f"\n=== Testing {model_name} ===")
        model_cfg_path = os.path.join("configs", f"{model_name}.yaml")
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        model = Model(model_cfg)

        # Load pretrained weights
        pretrained_weight_path = os.path.join("pretrained", f"{model_name}_pretrained.pth")
        model = load_pretrained_weights(model, pretrained_weight_path)

        # Model summary
        model_summary = summary(model, input_size=image_tensor.shape, depth=3, verbose=1)
        
        # GPU Inference time server
        time_test_results = gpu_inference_time_test(model, image_tensor)

        # Classification server
        classify_results = imagenet_classify_test(model, image_tensor, imagenet_classes_name)

        # Save report
        with open(f"results/{model_name}_report.txt", "w") as f:
            f.write(str(model_summary))
            f.write("\n")
            f.write("Device: GPU (RTX-4090)\n")
            f.write("Framework: PyTorch\n")
            f.write("="*60)
            f.write("\nClassification Test\n")
            for key, value in classify_results.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 60)
            f.write("\nInference Time Test\n")
            for key, value in time_test_results.items():
                f.write(f"{key}: {value:.4f}\n")


if __name__ == "__main__":
    set_deterministic(random_seed=666)
    main()