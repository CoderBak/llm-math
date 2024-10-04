import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from .grader import *
from .parser import *
from .utils import load_jsonl
from .python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False, use_tqdm=True):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        if use_tqdm:
            for sample in tqdm(samples, desc="Execute"):
                sample['pred'] = []
                sample['report'] = []
                for code in sample['code']:
                    pred, report = run_execute(executor, code, prompt_type, data_name, execute=True)
                    sample['pred'].append(pred)
                    sample['report'].append(report)
        else:
            for sample in samples:
                sample['pred'] = []
                sample['report'] = []
                for code in sample['code']:
                    pred, report = run_execute(executor, code, prompt_type, data_name, execute=True)
                    sample['pred'].append(pred)
                    sample['report'].append(report)

    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()

        if use_tqdm:
            with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
                while True:
                    try:
                        result = next(iterator)
                        scores.append(result)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        print(error)
                        scores.append(False)
                        timeout_cnt += 1
                    except Exception as error:
                        print(error.traceback)
                        exit()
                    progress_bar.update(1)

        else:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    print(np.array(score_mat).sum(axis=0))
    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=5))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    if use_tqdm:
        print(result_json)

    return samples, result_json
