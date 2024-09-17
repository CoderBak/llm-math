import matheval

"""
matheval 的基本用法（需要 model 支持 vllm）

1. basic_check(A, B): 检查 A, B 两个 **纯数学** 表达式是否一致，返回 bool

2. check(
    prompt_type,
    data_name,
    target,
    pred
): 检查两个解题过程给出的最终答案是否一致，由于需要先提取最终答案，故准确率不如 basic_check，返回 bool

target 部分即为数据集的某一行，例如：
{"question":"In Sweden, the red fox (Vulpes vulpes) severely limits populations of its prey, including hares. However, red fox populations are sometimes attacked by a fatal parasite, the mange mite. As mite population sizes increase at a given site, how are hare and fox populations most likely to respond at the same site? (Assume that hares have no major predators at this site other than foxes.)","type":"college_biology","choices":["Both fox and hare populations will decrease.","Both fox and hare populations will increase.","Fox populations will decrease and hare populations will increase.","Fox populations will increase and hare populations will decrease."],"answer":2}

3. MathEval 类

使用下面的方法加载模型，可补充加载模型时的参数，参考 vllm 的 LLM 类
engine = MathEval(model_path, args)
tensor_parallel_size 默认设为 CUDA_VISIBLE_DEVICES 数量（假定不超过10张显卡）

使用下面的命令来指定后续运行时的参数，例如 top_k 等。再次使用该命令可以更新参数
engine.set_sampling_args(args)

使用下面的命令来进行**批量**推理，如果一定要单次推理可以使用 [input]
由于 vllm 针对批量推理进行优化，建议使用 inputs
results = engine.generate(inputs)

使用下面的命令来进行一次对话
results = engine.chat(messages)

使用下面的命令来在某些数据集上进行评测
engine.test(datasets=["gsm8k"], prompt_type="direct")

评测机可以接收评测指令，例如
engine.test(
    datasets=["gsm8k"],
    shuffle=True,
)

下面的指令列表第一个是默认值
prompt_type: "tool-integrated", "direct", "cot", "pal", "self-instruct", "self-instruct-boxed", "tora", "pal", "cot", "wizard_zs", "platypus_fs", "deepseek-math", "kpmath"
split: test, ...
num_test_sample: -1, 随机选取这些数量的做测试
shuffle: True, 是否随机打乱测试集
save_outputs: True, 是否保存

注：之前在 set_sampling_args 设定的 stop_word 在 test 中不起作用

4. set_seed(seed)：设置全局种子，该种子将用于所有地方（所以请不要在模型加载处设置 seed）
"""

matheval.set_seed(42)

print(matheval.basic_check('(0.6,3.6667]', '(\\frac{3}{5},\\frac{8}{3} + 1]'))

print(matheval.check(
    prompt_type="cot",
    data_name="gsm8k",
    target={"question":"Kalinda is working on a 360 piece puzzle with her mom. Kalinda can normally add 4 pieces per minute. Her mom can typically place half as many pieces per minute as Kalinda.  How many hours will it take them to complete this puzzle?","answer":"Her mom places 2 pieces per minute because 4 \/ 2 = <<4\/2=2>>2\nOn average they get in 6 pieces per minute because 4 + 2 = <<4+2=6>>6\nIt will take 60 minutes to complete the puzzle because 360 \/ 6 = <<360\/6=60>>60\nIt will take one hour because 60 \/ 60 = <<60\/60=1>>1\n#### 1","idx":228},
    pred="Kalinda can add 4 pieces per minute. Her mom can add half as many pieces per minute as Kalinda. So her mom can add 2 pieces per minute. 360 pieces divided by 4 is 90. 90 divided by 2 is 45. 45 minutes is 1 hour. The answer is 1 hour."
))

print(matheval.check(
    prompt_type="cot",
    data_name="mmlu_stem",
    target={"question":"In Sweden, the red fox (Vulpes vulpes) severely limits populations of its prey, including hares. However, red fox populations are sometimes attacked by a fatal parasite, the mange mite. As mite population sizes increase at a given site, how are hare and fox populations most likely to respond at the same site? (Assume that hares have no major predators at this site other than foxes.)","type":"college_biology","choices":["Both fox and hare populations will decrease.","Both fox and hare populations will increase.","Fox populations will decrease and hare populations will increase.","Fox populations will increase and hare populations will decrease."],"answer":2},
    pred="Kalinda can add 4 pieces per minute. Her mom can add half as many pieces per minute as Kalinda. So her mom can add 2 pieces per minute. 360 pieces divided by 4 is 90. 90 divided by 2 is 45. 45 minutes is 1 hour. The answer is 1 hour."
))

# You can set load-model arguments: tensor_parallel_size, dtype, quantization, etc.
engine = matheval.MathEval("/media/public/models/huggingface/meta-llama/Llama-2-7b-hf", enforce_eager=False)

# You should set sampling args before inferencing.
# best_of, presence_penalty, frequency_penalty, repetition_penalty, temperature
# top_p, top_k, min_p, use_beam_search, length_penalty, early_stopping
# stop, max_tokens, min_tokens, etc.
engine.set_sampling_args(temperature=0, max_tokens=2048)

# Use test to evaluate the model on the given dataset.
engine.test(
    datasets=["gsm8k", "math", "svamp", "asdiv", "mawps", "tabmwp", "mathqa", "mmlu_stem", "sat_math"],
    prompt_type="cot",
    num_test_sample=5
)

engine.test(
    datasets=["gsm8k", "math", "svamp", "asdiv", "mawps", "tabmwp", "mathqa", "mmlu_stem", "sat_math"],
    prompt_type="cot"
)

"""
# Use generate to generate results.
print(engine.generate([
    "---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6=",
    "Answer this question directly: The sum of two numbers is 10. The difference of the same two numbers is 4. What are the two numbers?"
]))

# Use chat to generate a response.
print(engine.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "我今天18岁生日，你能写句话祝福我吗"},
    {"role": "assistant", "content": "祝你生日快乐！"},
    {"role": "user", "content": "你知道我今年多少岁吗？"}
]))
"""
