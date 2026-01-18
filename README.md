# CodeContests-O

Feedback-Driven Iterative Test Case Generation Framework

## 概述

CodeContests-O 是一个用于生成高质量竞赛编程测试用例的框架。它使用反馈驱动的迭代优化方法，通过大语言模型和代码执行沙箱协同工作，生成能够有效区分正确解和错误解的测试用例。

## 目录结构

```
codecontests_o/
├── pyproject.toml           # 构建配置
├── setup.py                 # 安装脚本
├── src/
│   └── codecontests_o/      # 源代码包
│       ├── __init__.py      # 主包入口
│       ├── main.py          # 命令行入口
│       ├── config/          # 配置管理
│       ├── data/            # 数据处理
│       ├── clients/         # API 客户端
│       ├── core/            # 核心逻辑
│       ├── parallel/        # 并行处理
│       ├── prompts/         # 提示模板
│       ├── utils/           # 工具函数
│       └── examples/        # 示例
└── README.md                # 文档
```

## 安装

确保你的环境中使用的是 Python 3.8 或更高版本。

```bash
# 克隆仓库
git clone <repository_url>
cd codecontests_o

# 安装 (开发模式)
pip install -e .

# 或者标准安装
pip install .
```

## 快速开始

### 使用本地 JSON 文件

```bash
python -m codecontests_o.main \
    --data_path /path/to/codecontests/json \
    --results_dir ./results \
    --api_key YOUR_OPENAI_API_KEY \
    --sandbox_hosts localhost \
    --testlib_path ./testlib.h
```

### 使用 HuggingFace 数据集

CodeContests-O 支持直接从 HuggingFace 加载数据集，如 `ByteDance-Seed/Code-Contests-Plus`：

```bash
# 从 HuggingFace Hub 加载
python -m codecontests_o.main \
    --data_path ByteDance-Seed/Code-Contests-Plus \
    --results_dir ./results \
    --api_key YOUR_OPENAI_API_KEY \
    --sandbox_hosts localhost \
    --testlib_path ./testlib.h
```

Python API 方式：

```python
from codecontests_o import CodeContestsReader, ParallelProcessor

# 自动检测数据源类型
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    split="test",  # train, test, or validation
    start=0,
    end=10
)

# 或者显式指定使用 HuggingFace
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    use_hf=True,
    split="test"
)
```

### 使用预设配置

```bash
# 开发环境（少量并行，调试模式）
python -m codecontests_o.main --preset development --data_path ./data

# 生产环境（高并行度）
python -m codecontests_o.main --preset production --data_path ./data

# 快速测试（仅生成，不验证）
python -m codecontests_o.main --preset quick --data_path ./data
```

### Python API 使用

```python
from codecontests_o import (
    Config,
    CodeContestsReader,
    ParallelProcessor,
    get_preset_config,
)
import base64

# 1. 创建配置
config = Config.from_dict(get_preset_config("development"))
config.openai.api_key = "your-api-key"
config.dataset.data_path = "/path/to/data"
config.dataset.results_dir = "./results"

# 2. 加载 testlib.h
with open("testlib.h", "rb") as f:
    testlib_files = {"testlib.h": base64.b64encode(f.read()).decode()}

# 3. 创建数据集读取器
# 方式 1: 从本地 JSON 文件
dataset = CodeContestsReader(
    data_path=config.dataset.data_path,
    start=0,
    end=10  # 只处理前10个样本
)

# 方式 2: 从 HuggingFace 数据集
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    split="test",
    start=0,
    end=10
)

# 4. 创建处理器并运行
processor = ParallelProcessor(config=config, testlib_files=testlib_files)
stats = processor.process_dataset(dataset, config.dataset.results_dir)

print(f"Completed: {stats['completed']}/{stats['total']}")
```

## 解决方案评估

如果你只想评估数据集中的解决方案在现有测试用例上的表现（检测假阴性和假阳性），可以使用 `solutions_eval` 模块。这对于分析数据集质量非常有用。

### 基本用法

```bash
python -m codecontests_o.solutions_eval \
    --data_path ByteDance-Seed/Code-Contests-Plus \
    --subset 1x \
    --results_dir ./results_eval \
    --start 0 --end 10 \
    --sandbox_hosts localhost
```

### 参数说明

*   `--data_path`: 数据集路径（本地目录或 HuggingFace 数据集名称）
*   `--subset`: 数据集子集（例如 `1x`, `2x`，仅适用于 Code-Contests-Plus）
*   `--results_dir`: 结果保存目录
*   `--start`/`--end`: 处理样本范围
*   `--sample_workers`: 样本级并行度
*   `--validation_workers`: 单个样本内的验证并行度

### 结果分析

在运行评估之后，使用 `analyze_results` 计算总体指标（TPR/TNR）：

```bash
python -m codecontests_o.analyze_results --results_dir ./results_eval
```

这将输出详细的统计信息，包括：
*   **TPR (True Positive Rate)**: 正确解被判定为正确的比例（越高越好）。
*   **TNR (True Negative Rate)**: 错误解被判定为错误的比例（越高越好）。
*   **Intersection**: 同时拥有有效 TPR 和 TNR 的样本的平均值。

## 自定义数据集接入

框架设计为可扩展的，你可以通过实现 `DatasetReader` 接口来接入自己的数据集。

### 步骤 1: 创建自定义读取器

```python
# my_reader.py
from codecontests_o.data import DatasetReader, Sample, Solution, TestCase, Language

class MyDatasetReader(DatasetReader):
    def __init__(self, data_path: str, start: int = 0, end: int = -1):
        self.data_path = data_path
        self._samples = self._load_data()
    
    def _load_data(self):
        samples = []
        # 加载你的数据...
        for item in your_data:
            sample = Sample(
                id=item['id'],
                name=item['name'],
                description=item['description'],
                generator=item.get('generator_code'),  # Optional. If None, will specific generated.
                canonical_solutions=[
                    Solution(code=item['solution'], language=Language.PYTHON)
                ],
                correct_solutions=[
                    Solution(code=item['solution'], language=Language.PYTHON)
                ],
                incorrect_solutions=[
                    Solution(code=item['wrong_solution'], language=Language.PYTHON)
                ],
                test_cases=[
                    # Required if using solutions_eval to evaluate existing test cases
                    TestCase(input="1\n", output="2\n")
                ]
            )
            samples.append(sample)
        return samples
    
    def __iter__(self):
        for sample in self._samples:
            yield sample
    
    def __len__(self):
        return len(self._samples)
    
    @property
    def name(self):
        return "MyDataset"
```

### 步骤 2: 使用自定义读取器

```bash
python -m codecontests_o.main \
    --custom_reader my_reader.py \
    --data_path /path/to/your/data \
    --results_dir ./results
```

或者通过 Python API:

```python
from my_reader import MyDatasetReader

dataset = MyDatasetReader(data_path="/path/to/data")
processor = ParallelProcessor(config=config, testlib_files=testlib_files)
processor.process_dataset(dataset, results_dir)
```

## Sample 数据模型

`Sample` 是框架的核心数据结构，包含以下字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | str | ✓ | 唯一标识符 |
| `name` | str | ✓ | 问题名称 |
| `description` | str | ✓ | 问题完整描述 |
| `generator` | str | | C++ 测试生成器代码（可选，若无则从头生成） |
| `checker` | str | | C++ checker 代码（可选） |
| `canonical_solutions` | List[Solution] | ✓ | 规范解（用于生成测试输出） |
| `correct_solutions` | List[Solution] | ✓ | 正确解列表 |
| `incorrect_solutions` | List[Solution] | ✓ | 错误解列表 |
| `test_cases` | List[TestCase] | | 测试用例（solutions_eval 时必需） |
| `metadata` | Dict | | 其他元数据 |

## 配置选项

### OpenAI 配置

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `api_base` | `https://api.openai.com/v1` | API 基础 URL |
| `api_key` | - | API 密钥 |
| `model` | `gpt-4o` | 模型名称 |
| `max_tokens` | `8000` | 最大 token 数 |
| `no_reasoning` | `True` | 是否禁用推理模式 |

### Sandbox 配置

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `hosts` | `["localhost"]` | 沙箱主机列表 |
| `base_port` | `8080` | 基础端口 |
| `port_range` | `4` | 每个主机的端口数 |
| `compile_timeout` | `20` | 编译超时（秒） |
| `run_timeout` | `20` | 运行超时（秒） |

### 处理配置

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `max_iterations` | `3` | 每个样本的最大迭代次数 |
| `sample_level_workers` | `4` | 样本级并行度 |
| `output_generation_workers` | `4` | 输出生成并行度 |
| `solution_validation_workers` | `4` | 验证并行度 |
| `only_generate` | `False` | 仅生成不验证 |

## 输出格式

每个样本的结果保存为 JSON 文件：

```json
{
    "id": "Codeforces_1234A",
    "status": "completed",
    "corner_cases": [
        {"input": {"stdin": "..."}, "output": {"stdout": "..."}}
    ],
    "commands": ["./gen --n 1000", "./gen --n 1 --edge"],
    "result": [
        {
            "iteration": 0,
            "corner_cases": [...],
            "generate_commands": [...],
            "improved_generator": "...",
            ...
        }
    ]
}
```

## 工作流程

1. **初始生成**: LLM 分析问题和生成器，生成初始命令列表
2. **执行命令**: 在沙箱中执行命令，生成测试输入
3. **生成输出**: 使用规范解运行测试输入，生成期望输出
4. **验证**: 在正确解和错误解上运行测试用例
5. **反馈优化**: 根据验证结果，LLM 优化命令和生成器
6. **迭代**: 重复步骤 2-5 直到达到最大迭代次数或所有解都被正确分类

## License

MIT License
