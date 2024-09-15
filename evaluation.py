import vllm

class MathEval():
    def __init__(self, model_path, **model_args):
        """
        This function loads the model following model_args.
        Please refer to vllm documentation to know the list of arguments.
        For example,
            - tokenizer
            - tensor_parallel_size
            - dtype
            - quantization
            - seed
            - gpu_memory_utilization
            - enforce_eager
        """

        self.llm = vllm.LLM(
            model=model_path,
            trust_remote_code=True,
            **model_args
        )

    def set_sampling_args(self, **sampling_args):
        """
        This function sets the sampling params.
        It must be called before evaluation.
        For example,
            - best_of
            - presence_penalty
            - frequency_penalty
            - repetition_penalty
            - temperature
            - top_p
            - top_k
            - min_p
            - seed
            - use_beam_search
            - length_penalty
            - early_stopping
            - stop
            - max_tokens
            - min_tokens
        """

        self.sampling_args = sampling_args

    def generate(self, inputs):
        """
        This function generates the inputs based on the sampling args.
        """

        assert(self.sampling_args is not None)
        sampling_params = vllm.SamplingParams(**self.sampling_args)
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        return outputs

    def evaluation(self, dataset_name):
        """
        This function evaluates the model on a specific dataset.
        """

        dataset = Dataset(dataset_name)
        preds = self.generate(dataset.inputs)
        print(preds)
        print(dataset.labels)
