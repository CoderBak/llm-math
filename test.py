import llm_math

llm_math.set_seed(0)
engine = llm_math.Model("/media/public/models/huggingface/Qwen/Qwen2-0.5B-Instruct")
#engine = llm_math.Model("/home/sunhaoxiang/models/meta-llama/Llama-3.2-3B-Instruct")
#engine = llm_math.Model("/media/public/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct")
engine.set_sampling_args(temperature=0, top_p=1.0, max_tokens=4096)

if False:
    """
    engine.test(
        datasets=["math_oai", "math_oai_wrong_50", "math_oai_wrong", "math_oai_correct", "4o_correct", "4o_wrong"],
        prompt_type="test",
        test_prompt="",
        template_path="./template_1.md"
    )

    """
    engine.test(
        datasets=["math_oai", "math_oai_wrong_50", "math_oai_wrong", "math_oai_correct", "4o_correct", "4o_wrong"],
        prompt_type="test",
        test_prompt="{Question}"
    )
else:
    engine.test(
        datasets=["math_oai"],
        prompt_type="test",
        test_prompt="{Question}"
    )
