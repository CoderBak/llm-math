import llm_math

llm_math.set_seed(42)

print(llm_math.basic_check('(0.6,3.6667]', '(\\frac{3}{5},\\frac{8}{3} + 1]'))

print(llm_math.check(
    prompt_type="cot",
    data_name="gsm8k",
    target={"question":"Kalinda is working on a 360 piece puzzle with her mom. Kalinda can normally add 4 pieces per minute. Her mom can typically place half as many pieces per minute as Kalinda.  How many hours will it take them to complete this puzzle?","answer":"Her mom places 2 pieces per minute because 4 \/ 2 = <<4\/2=2>>2\nOn average they get in 6 pieces per minute because 4 + 2 = <<4+2=6>>6\nIt will take 60 minutes to complete the puzzle because 360 \/ 6 = <<360\/6=60>>60\nIt will take one hour because 60 \/ 60 = <<60\/60=1>>1\n#### 1","idx":228},
    pred="Kalinda can add 4 pieces per minute. Her mom can add half as many pieces per minute as Kalinda. So her mom can add 2 pieces per minute. 360 pieces divided by 4 is 90. 90 divided by 2 is 45. 45 minutes is 1 hour. The answer is 1 hour."
))

print(llm_math.check(
    prompt_type="cot",
    data_name="mmlu_stem",
    target={"question":"In Sweden, the red fox (Vulpes vulpes) severely limits populations of its prey, including hares. However, red fox populations are sometimes attacked by a fatal parasite, the mange mite. As mite population sizes increase at a given site, how are hare and fox populations most likely to respond at the same site? (Assume that hares have no major predators at this site other than foxes.)","type":"college_biology","choices":["Both fox and hare populations will decrease.","Both fox and hare populations will increase.","Fox populations will decrease and hare populations will increase.","Fox populations will increase and hare populations will decrease."],"answer":2},
    pred="Kalinda can add 4 pieces per minute. Her mom can add half as many pieces per minute as Kalinda. So her mom can add 2 pieces per minute. 360 pieces divided by 4 is 90. 90 divided by 2 is 45. 45 minutes is 1 hour. The answer is 1 hour."
))

# You can set load-model arguments: tensor_parallel_size, dtype, quantization, etc.
engine = llm_math.Model("/media/public/models/huggingface/meta-llama/Llama-2-7b-hf", enforce_eager=False)

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
