import llm_math

llm_math.set_seed(0)

engine = llm_math.Model("/media/public/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct")
engine.set_sampling_args(temperature=0, top_p=1.0, max_tokens=4096)

"""
engine.test(
    datasets=["math_oai", "math_oai_wrong_50", "math_oai_wrong", "math_oai_correct", "4o_correct", "4o_wrong"],
    prompt_type="test",
    pattern=[
        {"role": "user", "content": "%%./extra/template_1.md%%\n[Question]\n%%Question%%"}
    ]
)

"""
engine.test(
    datasets=["math_oai", "math_oai_wrong_50", "math_oai_wrong", "math_oai_correct", "4o_correct", "4o_wrong"],
    prompt_type="test",
    pattern=[
        {"role": "system", "content": "%%./extra/template_1_1.md%%"},
        {"role": "user", "content": "%%./extra/template_1_2.md%%"},
        {"role": "assistant", "content": "%%./extra/template_1_3.md%%"},
        {"role": "user", "content": "%%./extra/template_1_4.md%%[Question]\n%%Question%%"},
    ]
)

llm_math.dry_run(
    data_name="math_oai",
    prompt_type="cot",
    file_path="./extra/s40_e10_r10.jsonl",
    input="problem",
    target="model_output"
)
