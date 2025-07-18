import time
import yaml
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from models.resnet50 import ResNet50


def load_and_preprocess_image(image_path, input_size=(224, 224)):
    """加载并预处理图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换并添加batch维度
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def benchmark_inference(model, input_tensor, num_warmup=10, num_iterations=100):
    """测试模型推理时间"""
    model.eval()
    
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    print(f"使用设备: {device}")
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 预热
    print(f"正在进行 {num_warmup} 次预热...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # 如果使用CUDA，同步
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 正式测试
    print(f"正在进行 {num_iterations} 次推理测试...")
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            output = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{num_iterations} 次测试")
    
    # 统计结果
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    
    print(f"\n=== 推理时间统计结果 ===")
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"最短推理时间: {min_time:.2f} ms")
    print(f"最长推理时间: {max_time:.2f} ms")
    print(f"标准差: {std_time:.2f} ms")
    print(f"FPS (帧每秒): {1000/avg_time:.2f}")
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'std_time_ms': std_time,
        'fps': 1000/avg_time
    }


def main():
    print("=== ResNet-50 推理时间基准测试 ===")
    
    # 设置参数
    image_path = "Tocal.jpg"
    config_path = "configs/resnet50.yaml"
    
    try:
        # 加载并预处理图像
        print(f"正在加载图像: {image_path}")
        input_tensor = load_and_preprocess_image(image_path)
        print(f"图像预处理完成，输入形状: {input_tensor.shape}")
        
        # 构建模型
        print(f"正在从配置文件构建ResNet-50模型: {config_path}")
        model = ResNet50(config_path)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总数: {total_params:,}")
        
        # 进行推理时间测试
        benchmark_results = benchmark_inference(model, input_tensor)
        
        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()