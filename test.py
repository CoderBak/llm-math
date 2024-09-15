import matheval

engine = matheval.MathEval("/root/autodl-tmp/qwen2-1.5b", seed=42)
engine.set_sampling_args(temperature=0.1)
print(engine.generate(["You are a helpful assistant. 今天天气如何"]))
engine.evaluation("gsm8k")
