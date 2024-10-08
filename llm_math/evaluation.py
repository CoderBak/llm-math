import os
import torch
import vllm
import random
import time
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from .evaluate import evaluate
from .utils import save_jsonl, construct_prompt, postprocess
from .parser import *
from .trajectory import *
from .data_loader import load_data
from .python_executor import PythonExecutor

class Model():
    def __init__(self, model_path, **model_args):
        """
        This function loads the model following model_args.
        Please refer to vllm documentation to know the list of arguments.
        You can set: tokenizer, tensor_parallel_size, dtype, quantization,
        gpu_memory_utilization, enforce_eager, etc.
        """
        self.seed = int(os.environ["PYTHONHASHSEED"])
        model_args["seed"] = self.seed
        if "tensor_parallel_size" not in model_args:
            model_args["tensor_parallel_size"] = torch.cuda.device_count()
        self.model_path = model_path
        self.llm = vllm.LLM(model=model_path, trust_remote_code=True, **model_args)


    def set_sampling_args(self, **sampling_args):
        """
        This function sets the sampling params. It must be called before evaluation.
        You can set: best_of, presence_penalty, frequency_penalty, repetition_penalty, temperature,
        top_p, top_k, min_p, use_beam_search, length_penalty, early_stopping,
        stop, max_tokens, min_tokens, etc.
        """
        self.sampling_args = sampling_args
        self.sampling_args["seed"] = self.seed
        self.sampling_params = vllm.SamplingParams(**sampling_args)


    def generate(self, inputs):
        # This function generates the inputs based on the sampling args.
        assert(hasattr(self, 'sampling_params'))
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return [output.outputs[0].text for output in outputs]


    def chat(self, messages):
        # This function generates the response based on the given conversation.
        assert(hasattr(self, 'sampling_params'))
        response = self.llm.chat(messages, sampling_params=self.sampling_params, use_tqdm=False)
        return response[0].outputs[0].text


    def test(self, datasets, prompt_type, **kwargs):
        results = []
        for data_name in datasets:
            cur_result = self.evaluation(data_name, prompt_type, **kwargs)
            results.append([data_name, cur_result])

        print("############## Statistics ##############")
        print(f"Model: {self.model_path}")
        print(f"prompt_type: {self.prompt_type}")
        for result in results:
            print("-" * 30)
            print(f"{result[0]}: {result[1]}")


    def evaluation(self, data_name, prompt_type, split="test",
                   num_test_sample=-1, shuffle=True, test_prompt="",
                   save_outputs=True, pattern=[]):
        # This function evaluates the model on a specific dataset.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.data_name = data_name
        self.base_path = os.path.dirname(__file__)
        if self.base_path.endswith("/"):
            self.base_path = self.base_path[:len(self.base_path) - 1]
        self.data_path = self.base_path + "/data"
        self.prompt_path = self.base_path + "/prompts"
        self.prompt_type = prompt_type
        self.split = split
        self.num_test_sample = num_test_sample
        self.shuffle = shuffle
        self.save_outputs = save_outputs
        self.test_prompt = test_prompt
        self.pattern = pattern
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        result = self.run()
        return result


    def run(self):
        examples = load_data(self.data_name, self.split, self.data_path)
        if self.num_test_sample > 0:
            examples = random.sample(examples, self.num_test_sample)
        if self.shuffle:
            random.shuffle(examples)

        out_file_prefix = f'./output/{datetime.now().strftime("%m-%d_%H-%M")}/'
        out_file = out_file_prefix + f'results_{self.data_name}.jsonl'
        os.makedirs(out_file_prefix, exist_ok=True)

        print("=" * 50)
        print("data:", self.data_name, " ,remain samples:", len(examples))
        print(examples[0])

        if "pal" in self.prompt_type:
            executor = PythonExecutor(get_answer_expr='solution()')
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        samples = []
        for example in tqdm(examples, total=len(examples)):
            idx = example['idx']

            # parse question and answer
            example['question'] = parse_question(example, self.data_name)
            gt_cot, gt_ans = parse_ground_truth(example, self.data_name)
            full_prompt = construct_prompt(example, self.data_name, self.prompt_type, self.prompt_path)

            if idx == 0:
                print(full_prompt)

            sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

            # add remain fields
            for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
                'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
                if key in example:
                    sample[key] = example[key]
            samples.append(sample)


        # repeat n times
        input_prompts = [sample['prompt'] for sample in samples]
        remain_prompts = input_prompts
        remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
        end_prompts = []

        max_func_call = 1 if self.prompt_type in ['cot', 'pal'] else 4

        # stop words TODO: make it more general
        stop_words = ["</s>"]

        if self.prompt_type in ['cot']:
            stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
        if self.prompt_type in ['pal', 'tool-integrated', 'tora']:
            stop_words.extend(["\n\n---", "```output"])
        elif self.prompt_type in ['wizard_zs', 'platypus_fs']:
            stop_words.extend(["Instruction", "Response"])
        print("Stop words:", stop_words)

        new_sampling_args = self.sampling_args.copy()
        new_sampling_args["stop"] = stop_words
        self.sampling_params = vllm.SamplingParams(**new_sampling_args)

        # start inference
        # measure time use
        start_time = time.time()
        for epoch in range(max_func_call):
            print("-" * 20, "Epoch", epoch, "-" * 20)
            current_prompts = remain_prompts
            if len(current_prompts) == 0:
                break

            # get all outputs
            prompts = [item[1] for item in current_prompts]

            new_prompts = []

            if self.prompt_type == "test":
                for prompt in prompts:
                    new_elem = []
                    for elem in self.pattern:
                        new_elem.append({"role": elem["role"], "content": postprocess(elem["content"], prompt)})
                    if len(new_prompts) == 0:
                        print(new_elem)
                    new_prompts.append(self.tokenizer.apply_chat_template(
                        new_elem,
                        tokenize=False,
                        add_generation_prompt=True
                    ))

            else:
                for prompt in prompts:
                    if len(new_prompts == 0):
                        print([{"role": "user", "content": prompt}])

                    new_prompts.append(self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    ))

            outputs = self.generate(new_prompts)
            assert len(outputs) == len(current_prompts)

            # process all outputs
            remain_prompts = []
            remain_codes = []
            for (i, query), output in zip(current_prompts, outputs):
                output = output.rstrip()
                query += output
                if self.prompt_type == "pal":
                    remain_prompts.append((i, query))
                    if "```python" in output:
                        output = extract_program(query)
                    remain_codes.append(output)
                elif self.prompt_type == "cot":
                    end_prompts.append((i, query))
                elif ("boxed" not in output and output.endswith("```")):
                    program = extract_program(query)
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))

            # execute the remain prompts
            remain_results = executor.batch_apply(remain_codes)
            for k in range(len(remain_prompts)):
                i, query = remain_prompts[k]
                res, report = remain_results[k]
                exec_result = res if res else report
                if "pal" in self.prompt_type:
                    exec_result = "\\boxed{" + exec_result + "}"
                exec_result = f"\n```output\n{exec_result}\n```\n"
                query += exec_result
                # not end
                if epoch == max_func_call - 1:
                    query += "\nReach max function call limit."
                remain_prompts[k] = (i, query)

        # unsolved samples
        print("Unsolved samples:", len(remain_prompts))
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
        results = [run_execute(executor, code, self.prompt_type, self.data_name) for code in codes]
        time_use = time.time() - start_time

        # put results back to examples
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
        all_samples, result_json = evaluate(samples=all_samples, data_name=self.data_name, prompt_type=self.prompt_type, execute=True)

        if self.save_outputs:
            save_jsonl(all_samples, out_file)

        result_json['time_use_in_second'] = time_use
        result_json['time_use_in_minite'] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"

        with open(out_file.replace(".jsonl", f"_{self.prompt_type}_metrics.json"), "w") as f:
            json.dump(result_json, f, indent=4)

        self.sampling_params = vllm.SamplingParams(**self.sampling_args)

        return result_json



def dry_run(data_name, prompt_type, file_path, input, target, split="test"):
    ori_inputs = []

    # This function evaluates the model on a specific dataset.
    base_path = os.path.dirname(__file__)
    if base_path.endswith("/"):
        base_path = base_path[:len(base_path) - 1]
    data_path = base_path + "/data"
    prompt_path = base_path + "/prompts"
    examples = load_data(data_name, split, data_path)

    out_file_prefix = f'./output/{datetime.now().strftime("%m-%d_%H-%M")}/'
    out_file = out_file_prefix + f'results_{data_name}.jsonl'
    os.makedirs(out_file_prefix, exist_ok=True)

    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    print(examples[0])

    if "pal" in prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []

    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, prompt_type, prompt_path)
        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)


    # repeat n times
    input_prompts = [sample['prompt'] for sample in samples]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if prompt_type in ['cot', 'pal'] else 4

    valid_count = 0

    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch, "-" * 20)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]

        outputs = []

        with open(file_path, "r", encoding="utf-8") as f:
            appeared = set()
            for line in f:
                entry = json.loads(line)
                ori_inputs.append(entry)
                for cur_index in range(len(prompts)):
                    cur_prompt = prompts[cur_index]
                    if entry[input] in cur_prompt:
                        valid_count += 1
                        outputs.append([cur_index, entry[target]])
                        appeared.add(cur_index)
                        break

            for cur_index in range(len(prompts)):
                if cur_index not in appeared:
                    outputs.append([cur_index, " "])

        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = [_[1] for _ in outputs]


        assert len(outputs) == len(current_prompts)

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
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
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

    # put results back to examples
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

    result_json['time_use_in_second'] = 0
    result_json['time_use_in_minite'] = 0

    correct = int(result_json["correct"])

    print(f"Valid samples: {valid_count} . Correct: {correct} . Acc: {100 * correct / valid_count} %")

    with open(out_file.replace(".jsonl", f"_{prompt_type}_metrics_dry_run.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    for i in range(len(ori_inputs)):
        entry = ori_inputs[i]
        cur_input = entry[input]
        for sample in all_samples:
            if sample['question'] == cur_input:
                ori_inputs[i]["score"] = sample["score"][0]
                break
    
    return ori_inputs
