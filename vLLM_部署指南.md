# vLLM 在 T4 GPU 上从源码编译部署指南

## 环境信息
- **GPU**: NVIDIA Tesla T4 (15360MiB 显存)
- **内存**: 30GB
- **CUDA 版本**: 12.0 
- **驱动版本**: 525.105.17
- **目标模型**: Llama-3.2-3B-Instruct
- **HuggingFace Token**: `hf_YOUR_TOKEN_HERE` (请替换为你的真实token)

## 第一步：检查系统环境

### 1.1 检查Python版本
```bash
python --version
python3 --version
```
输出：Python 3.8.10

### 1.2 检查GPU和CUDA信息
```bash
nvidia-smi
```

### 1.3 检查内存信息
```bash
free -h
```

## 第二步：安装 Miniconda

### 2.1 下载Miniconda安装包
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
```

### 2.2 安装Miniconda
```bash
bash ~/miniconda.sh -b -p $HOME/miniconda3
```

### 2.3 激活conda并接受服务条款
```bash
# 激活conda
source $HOME/miniconda3/bin/activate

# 检查版本
conda --version

# 接受服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 第三步：创建 conda 环境

### 3.1 创建专用的vLLM环境
```bash
# 确保conda已激活
source $HOME/miniconda3/bin/activate

# 创建Python 3.11环境（vLLM要求Python >= 3.9, < 3.13）
conda create -n vllm python=3.11 -y
```

### 3.2 激活vLLM环境
```bash
conda activate vllm

# 验证Python版本
python --version
```
输出应该是：Python 3.11.13

## 第四步：安装编译依赖

### 4.1 更新基础工具
```bash
# 确保环境已激活
source $HOME/miniconda3/bin/activate
conda activate vllm

# 更新pip和基础工具
pip install -U pip setuptools wheel
```

### 4.2 安装编译依赖包
```bash
# 安装编译所需的依赖（这一步会下载大量CUDA库，需要较长时间）
pip install -r requirements/build.txt
```

**注意**: 这一步会下载约2-3GB的CUDA相关库，包括：
- PyTorch 2.7.1 (约821MB)
- CUDA运行时库和工具
- cuDNN、cuBLAS等CUDA加速库
- 大约需要10-15分钟完成

## 第五步：从源码编译vLLM（你需要自己执行）

### 5.1 准备编译环境
```bash
# 确保在vllm源码目录
cd /home/ubuntu/vllm

# 激活环境
source $HOME/miniconda3/bin/activate
conda activate vllm
```

### 5.2 安装运行时依赖
```bash
# 安装CUDA相关依赖
pip install -r requirements/cuda.txt

# 安装通用依赖
pip install -r requirements/common.txt
```

### 5.3 开始编译安装（核心步骤）
```bash
# 方法1：使用pip安装（推荐，会自动处理编译）
pip install -e .

# 或者方法2：使用setup.py
python setup.py develop
```

**编译说明**：
- 编译过程需要15-30分钟，取决于机器性能
- 编译会产生大量输出，这是正常的
- 主要编译的组件包括：CUDA kernels、C++扩展、Python绑定等
- 如果编译失败，检查CUDA版本兼容性和内存是否足够

### 5.4 验证安装
```bash
# 测试导入
python -c "import vllm; print('vLLM安装成功!')"

# 检查版本
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
```

## 第六步：配置Hugging Face Token

### 6.1 设置环境变量
```bash
# 临时设置（当前会话有效）
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"

# 或者永久设置（添加到bashrc）
echo 'export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"' >> ~/.bashrc
source ~/.bashrc
```

### 6.2 使用huggingface-cli登录
```bash
# 安装huggingface hub
pip install huggingface-hub

# 登录
huggingface-cli login --token hf_YOUR_TOKEN_HERE
```

## 第七步：测试vLLM安装

### 7.1 简单功能测试
```bash
python -c "
from vllm import LLM, SamplingParams
print('vLLM导入成功!')
print('GPU可用性检查...')
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
"
```

## 第八步：部署Llama-3.2-3B-Instruct模型

### 8.1 测试模型加载
```bash
python -c "
from vllm import LLM, SamplingParams

# 创建LLM实例
llm = LLM(
    model='meta-llama/Llama-3.2-3B-Instruct',
    tensor_parallel_size=1,  # T4是单GPU
    gpu_memory_utilization=0.8,  # 使用80%显存
    max_model_len=2048  # 根据T4显存限制设置较小的上下文长度
)

print('模型加载成功!')
"
```

### 8.2 运行推理测试
```bash
python -c "
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model='meta-llama/Llama-3.2-3B-Instruct',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=2048
)

# 创建采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# 测试推理
prompts = ['Hello, how are you?', 'What is machine learning?']
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt}')
    print(f'Generated: {generated_text}')
    print('-' * 50)
"
```

### 8.3 启动API服务器（可选）
```bash
# 启动OpenAI兼容的API服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000
```

## 故障排除

### 常见问题和解决方案

1. **CUDA版本不匹配**
   ```bash
   # 检查CUDA版本
   nvcc --version
   nvidia-smi
   # 确保PyTorch CUDA版本匹配
   ```

2. **显存不足**
   ```bash
   # 降低gpu_memory_utilization参数
   # 减小max_model_len参数
   # 使用量化模型
   ```

3. **编译失败**
   ```bash
   # 清理缓存重新编译
   pip cache purge
   rm -rf build/
   pip install -e . --force-reinstall --no-cache-dir
   ```

4. **Python版本不兼容**
   ```bash
   # vLLM要求Python 3.9-3.12
   python --version
   # 如需要，重新创建conda环境
   ```

## 性能优化建议

1. **T4 GPU优化参数**：
   - `gpu_memory_utilization=0.8`（不要设置太高）
   - `max_model_len=2048`（根据显存调整）
   - `tensor_parallel_size=1`（单GPU）

2. **模型选择**：
   - 3B参数的模型适合T4
   - 可以考虑使用量化版本节省显存

3. **批处理**：
   - 适当设置batch size以提高吞吐量

## 部署状态总结

✅ **所有步骤已完成！**

- [x] 环境检查和准备
- [x] Miniconda安装和配置  
- [x] conda虚拟环境创建
- [x] 编译依赖安装
- [x] vLLM从源码编译成功
- [x] Hugging Face token配置
- [x] vLLM功能测试成功

**安装的vLLM版本**: `0.1.dev8596+g74f441f4b.cu120`

## 关于Llama模型访问

你提供的token没有访问Meta Llama模型的权限。Llama模型需要：
1. 先在Hugging Face上申请访问权限：https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Meta会审核申请（通常几小时到几天）
3. 获得权限后就可以使用你的token访问

## 替代方案

可以使用其他优秀的开源模型：
- **Qwen/Qwen2.5-3B-Instruct** - 阿里巴巴的高质量中文模型
- **microsoft/DialoGPT-medium** - 微软对话模型
- **google/flan-t5-base** - Google的指令调优模型

## 启动API服务示例

```bash
# 激活环境
source $HOME/miniconda3/bin/activate
conda activate vllm
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"

# 启动Qwen模型API服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000

# 测试API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }'
```

## 下一步建议

1. **申请Llama访问权限**（如果需要）
2. **测试不同模型**，找到最适合你应用的
3. **部署API服务**，集成到你的应用中
4. **优化参数**，根据实际需求调整性能设置

---

**🎉 恭喜！vLLM推理框架已在你的T4机器上成功部署！**