import llm_math

llm_math.set_seed(42)
print(llm_math.basic_check('(0.6,3.6667]', '(\\frac{3}{5},\\frac{8}{3} + 1]'))
print(llm_math.check(
    prompt_type="cot",
    data_name="gsm8k",
    target={"question":"Kalinda is working on a 360 piece puzzle with her mom. Kalinda can normally add 4 pieces per minute. Her mom can typically place half as many pieces per minute as Kalinda.  How many hours will it take them to complete this puzzle?","answer":"Her mom places 2 pieces per minute because 4 \/ 2 = <<4\/2=2>>2\nOn average they get in 6 pieces per minute because 4 + 2 = <<4+2=6>>6\nIt will take 60 minutes to complete the puzzle because 360 \/ 6 = <<360\/6=60>>60\nIt will take one hour because 60 \/ 60 = <<60\/60=1>>1\n#### 1","idx":228},
    pred="Kalinda can add 4 pieces per minute. Her mom can add half as many pieces per minute as Kalinda. So her mom can add 2 pieces per minute. 360 pieces divided by 4 is 90. 90 divided by 2 is 45. 45 minutes is 1 hour. The answer is 1 hour."
))
engine = llm_math.Model("/media/public/models/huggingface/meta-llama/Llama-2-7b-hf", enforce_eager=False)
engine.set_sampling_args(temperature=0, max_tokens=2048)
engine.test(
    datasets=["gsm8k", "math", "svamp", "asdiv", "mawps", "tabmwp", "mathqa", "mmlu_stem", "sat_math"],
    prompt_type="cot",
    num_test_sample=5
)
print(engine.generate([
    "---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6=",
    "Answer this question directly: The sum of two numbers is 10. The difference of the same two numbers is 4. What are the two numbers?"
]))
print(engine.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "我今天18岁生日，你能写句话祝福我吗"},
    {"role": "assistant", "content": "祝你生日快乐！"},
    {"role": "user", "content": "你知道我今年多少岁吗？"}
]))
