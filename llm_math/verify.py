from .grader import math_equal
from .evaluate import evaluate
from .utils import construct_prompt
from .parser import *
from .trajectory import *
from .python_executor import PythonExecutor


def basic_check(pred, answer):
    """
    This function compares the prediction and the ground truth (answer),
    and returns True if they are the same, otherwise returns False.

    Args:
    pred: The predicted value or output from a model or function.
    answer: The ground truth to compare against.

    Returns:
    bool: True if the prediction is correct, otherwise False.
    """
    return math_equal(pred, answer, timeout=True)


def check(prompt_type, data_name, target, pred):
    if "idx" not in target:
        target["idx"] = 0

    if "pal" in prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    idx = target['idx']

    # parse question and answer
    gt_cot, gt_ans = parse_ground_truth(target, data_name)

    base_path = os.path.dirname(__file__)
    if base_path.endswith("/"):
        base_path = base_path[:len(base_path) - 1]
    prompt_path = base_path + "/prompts"

    full_prompt = construct_prompt(target, data_name, prompt_type, prompt_path)
    sample = {'idx': idx, 'gt': gt_ans, 'prompt': full_prompt}

    # add remain fields
    for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
        'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
        if key in target:
            sample[key] = target[key]

    samples = [sample]

    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []
    current_prompts = remain_prompts

    # get all outputs
    prompts = [item[1] for item in current_prompts]
    outputs = [pred]

    # process all outputs
    remain_prompts = []
    remain_codes = []
    for (i, query), output in zip(current_prompts, outputs):
        output = output.rstrip()
        query += output
        if prompt_type == "pal":
            remain_prompts.append((i, query))
            if "```python" in output:
                output = extract_program(query)
            remain_codes.append(output)
        elif prompt_type == "cot":
            end_prompts.append((i, query))
        elif ("boxed" not in output and output.endswith("```")):
            program = extract_program(query)
            remain_prompts.append((i, query))
            remain_codes.append(program)
        else:
            end_prompts.append((i, query))

    # execute the remain prompts
    remain_results = executor.batch_apply(remain_codes, use_tqdm=False)
    for k in range(len(remain_prompts)):
        i, query = remain_prompts[k]
        res, report = remain_results[k]
        exec_result = res if res else report
        if "pal" in prompt_type:
            exec_result = "\\boxed{" + exec_result + "}"
        exec_result = f"\n```output\n{exec_result}\n```\n"
        query += exec_result
        remain_prompts[k] = (i, query)

    # unsolved samples
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, prompt_type, data_name) for code in codes]

    # put results back to targets
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i: i + 1]
        result = results[i: i + 1]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=prompt_type, execute=True, use_tqdm=False)

    return result_json["acc"] == 100.0
