import matheval

engine = matheval.MathEval("model_path", seed=42)
engine.set_sampling_args(temperture=0.1)
print(engine.generate(["今天天气如何"]))
engine.evaluation("gsm8k")
