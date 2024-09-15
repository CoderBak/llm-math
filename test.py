import matheval

# You can set load-model arguments: tensor_parallel_size, dtype, quantization, seed, etc.
engine = matheval.MathEval("/media/public/models/huggingface/Qwen/Qwen2-7B-Instruct", seed=42)

# You should set sampling args before inferencing.
# best_of, presence_penalty, frequency_penalty, repetition_penalty, temperature
# top_p, top_k, min_p, seed, use_beam_search, length_penalty, early_stopping
# stop, max_tokens, min_tokens, etc.
engine.set_sampling_args(temperature=0.1)

# Use generate to generate results.
print(engine.generate([
    "You are a helpful assistant. 今天天气如何",
    "讲个笑话吧"
]))

# Use evaluation to evaluate the model on the given dataset.
engine.evaluation("gsm8k")
