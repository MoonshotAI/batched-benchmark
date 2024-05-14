# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

# Benchmark with OB
## 获取OB
因为当前没有整理ob的代码，所以仅提供ob可执行文件给外部用户使用，有需求请邮件联系

## 数据准备
生成prompts，可使用`benchmarks/generate_ob_tests.py`，建议使用所测模型对应的tokenizer
```sh
# 可以设置自己的默认值，来减少命令行参数的配置; 建议直接用batched_benchamrk, 则不需要手动调用generate_ob_tests.py
python3 generate_ob_tests.py --min-tokens 1024 --max-tokens 1024 --count 1000 --tokenizer YOUR-HUGGINGFACE-TOKENIZER --output output.jsonl --dataset YOUR-DOWNLOADED-SHAREGPT-V3-DATASET
```

## OB 原生测试
```sh
ob -e "http://localhost:8888/v1" -m model-name -i ./corpora/tokens-1024-1024.jsonl -n 1000 --max-tokens 128 -c 100 --verbose
```

## Batched Benchmark

### 简介
* **基于 ob** 的批量测速脚本
* 依赖`generate_ob_tests.py`生成prompt
* 依赖`analyze_result.py`根据原始输出生成markdown表格
* 可使用`compare_result.py`比较两次测试的结果

### 完整benchmark流程
* 部署vllm
* 安装测速脚本需要的依赖项，比如直接 `pip install -r ./requirements.txt`
* 根据实际测试需求编写config文件，通常可以直接使用`full.yml`，如需修改可以参考`batched_benchmark_template.yml`的格式
* 调用`batched_benchmark.py`，参考命令
  ```sh
  # 可以设置自己的默认值，来减少命令行参数的配置
  python3 batched_benchmark.py -e ob -c ./full.yml -s http://localhost:8888/v1 -p /your/path/to/prompt -t /your/tokenism/path -d /your/path/to/ShareGPT_V3_unfiltered_cleaned_split.5000.json -o ./results
  ```
  * 具体命令含义可以`python3 ./batched_benchmark.py --help`查看，下面有简单解释
  * `-e ob`是指定测速可执行文件`ob`的路径
  * `-c ./full.yml`是指定测速配置文件的路径
  * `-s http://localhost:8888/v1`是指定vllm服务的地址，根据实际情况修改
  * `-p ...`是指定prompt路径，指向预先生成的prompt目录，缺少的部分会根据`-t -d`来生成，务必保证提供的prompt文件匹配待测模型，否则可能导致prompt长度不符合预期
  * `-t ...`是指定tokenizer模型路径，一般等于待测模型的地址，用于生成prompt，如果`-p`已满足需求则不需要设置该项
  * `-d ...`是指定dataset路径，作为生成prompt的语料来源，如果`-p`已满足需求则不需要设置该项
  * `-o ./outputs`是指定输出目录，注意事先不能存在该目录
  * 输出在`batched_benchmark.py`输出目录的`final_results`子目录
* (optional) `compare_result.py`可用来生成两次不同原始结果的比较
  * 方式一 `python3 compare_result.py -b <baseline_dir> -c <current_dir> -o <result.md>`
    * 此命令涉及的目录是`batched_benchmark.py`输出目录的`raw_results`子目录
  * 方式二 `python3 compare_result.py -b <baseline.json> -c <current.json> -f json -o <result.md>`
    * 此命令涉及的json文件是`batched_benchmark.py`输出目录的`final_results/summary.json`

### 备注
* `batched_benchmark.py`的输出文件夹不能事先存在（防止误操作覆盖之前的原始输出）
* 注意可能会出现失败请求数比较多的情况，会打印`WARN: fname=... num_failed=431`，此时的测速结果不可信，请排查问题重新测试
* full.yml中预置了一些benchmark的case，为了避免batched_benchmark每次重复generate prompt, 可以预生成一些常用长度的prompt文件(prompt-len.jsonl, 比如130000.jsonl)，然后用batched_benchamrk.py的`-p`选项传递prompt文件的所在目录

# Benchmark with vLLM Engine

### 简介
* **基于 vLLM Engine** 的批量测速脚本
* 依赖`generate_ob_tests.py`生成prompt
* 相比**Benchmark with OB**，会尽可能batch请求到给定的concurrency参数，方便测试不同的batch性能

support benchmarking using vllm engine sdk. Batch size is more accurate.
### 完整benchmark流程
* 部署vllm
* 安装测速脚本需要的依赖项，比如直接 `pip install -r ./requirements.txt`
* 根据实际测试需求编写config文件，通常可以直接使用`full.yml`，如需修改可以参考`batched_benchmark_template.yml`的格式
* 调用`batched_benchmark_vllm.py`，参考命令
  ```sh
  # 可以设置自己的默认值，来减少命令行参数的配置
  python3 batched_benchmark_vllm.py -c ./full.yml -p /your/path/to/prompt -t /your/tokenism/path -d /your/path/to/ShareGPT_V3_unfiltered_cleaned_split.5000.json -o ./results --model facebook/opt-125m
  ```
  * 具体命令含义可以`python3 ./batched_benchmark_vllm.py --help`查看，下面有简单解释
  * `-c ./full.yml`是指定测速配置文件的路径
  * `-p ...`是指定prompt路径，指向预先生成的prompt目录，缺少的部分会根据`-t -d`来生成，务必保证提供的prompt文件匹配待测模型，否则可能导致prompt长度不符合预期
  * `-t ...`是指定tokenizer模型路径，一般等于待测模型的地址，用于生成prompt，如果`-p`已满足需求则不需要设置该项
  * `-d ...`是指定dataset路径，作为生成prompt的语料来源，如果`-p`已满足需求则不需要设置该项
  * `-o ./outputs`是指定输出目录，注意事先不能存在该目录
  * `--model` 是vllm启动参数，可以继续追加其他vllm启动参数
  * 输出在`batched_benchmark_vllm.py`输出目录的`final_results`子目录
