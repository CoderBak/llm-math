# MathEval

基于 [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness/) 改造的 plug-and-play 数学评测包.

### Install

```bash
conda create -n matheval python=3.10
pip install -r requirements.txt
```

### Usage

#### 结果检查

- `basic_check(A, B)`

检查 A, B 两个**纯数学**表达式是否一致，返回 True / False.

### Notes

- 模型需支持 [vLLM](https://github.com/vllm-project/vllm).

- `test` 部分暂未适配多卡推理，请先使用单卡推理
