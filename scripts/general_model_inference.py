import os
import sys
import json
from vllm import LLM, SamplingParams
import torch
import argparse
import logging
from utils.extract_ans import get_gsmplus
from utils.prompt_template import gsm8k_nshot_prompt, gsm8k_nshot_prompt_pot, gsm8k_nshots_ltm_solution_best, gsm8k_1shot_ltm_1, gsm8k_complex_cot, gsm8k_contrastive_cot

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%d-%d %H:%M:%S')
MAX_INT = sys.maxsize

prompt_mapping = {
    "cot-nshot": gsm8k_nshot_prompt,
    "pot-nshot": gsm8k_nshot_prompt_pot,
    "ltm-nshot": gsm8k_nshots_ltm_solution_best,
    "ltm-1shot": gsm8k_1shot_ltm_1,
    "complex": gsm8k_complex_cot,
    "contrastive": gsm8k_contrastive_cot,
}

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def inference_vllm(args, raw_queries, batch_size=1, num_cpus=56, gpus='0,1,2,3'):
    # 1 we set the model
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    num_gpus = torch.cuda.device_count()
    logging.info('num_gpus: {}'.format(num_gpus))

    # 2 we prepare raw queries and wrap them with target prompt
    prompt = prompt_mapping[args.prompt_type]
    processed_prompts = [prompt.format(input=query) for query in raw_queries]

    logging.info('>>>>>> one processed prompt:\\{}'.format(processed_prompts[0]))
    processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts  # sample_num=-1
    logging.info('>>>>>> size of the processed prompts: {}\n'.format(len(processed_prompts)))

    import ray
    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
    logging.info('>>>>>> ray initialized')

    llm = LLM(model=args.model_dir,
              tensor_parallel_size=num_gpus)
    logging.info('>>>>>> model loaded')

    # 3 we set the sampling params
    sampling_params = SamplingParams(temperature=args.temperature,  # 0, Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.
                                     top_p=args.top_p,  # 1, Set to 1 to consider all tokens.
                                     max_tokens=args.max_tokens,  # 2048, Maximum number of tokens to generate per output sequence.
                                     stop=args.stop,
                                     presence_penalty=args.presence_penalty,
                                     frequency_penalty=args.frequency_penalty)


    # batch_ins = batch_data(processed_prompts, batch_size=batch_size)
    outputs = llm.generate(processed_prompts, sampling_params)
    sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
    logging.info('>>>>>> generation done')
    return sorted_outputs



def inference(args):
    questions, answers, types = get_gsmplus()
    print(f'model: {args.model_name}\noutput_file: {args.output_file}\nckpt_path: {args.model_dir}\nsample: {len(questions)}\nnshots: {args.nshots}')

    predictions = inference_vllm(args, raw_queries=questions, num_cpus=56, gpus=args.specify_your_gpus)
    print('size of predictions: ', len(predictions))

    outputs = []
    for idx, output in enumerate(predictions):
        model_prediction = output.outputs[0].text
        outputs.append({
            'idx': idx,
            'question': questions[idx],
            'answer': answers[idx],
            'type': types[idx],
            'model': args.model_name,
            'model_prediction': model_prediction,
            'prompt': output.prompt,
        })

    if os.path.exists(args.output_file):
        opt_data = json.load(open(args.output_file))
        opt_data.extend(outputs)
        outputs = opt_data  # record previous data

    print(outputs[0])
    json.dump(outputs, open(args.output_file, 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=False, help="llama2-70b-chat")
    parser.add_argument("--model_dir", default=None, type=str, required=False, help="model path")
    parser.add_argument("--output_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument("--specify_your_gpus", default=None, type=str, required=False, help="your available gpus")
    parser.add_argument("--nshots", default=None, type=int, required=False, help="few shot input")
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--stop', type=str, nargs='+', default=[],
                        help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='cot-nshot', choices=["cot-nshot", "pot-nshot", "ltm-nshot", "ltm-1shot", "complex", "contrastive"])
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)

    args = parser.parse_args()

    inference(args=args)



