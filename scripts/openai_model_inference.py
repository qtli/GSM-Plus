# -*- coding: utf-8 -*-
from utils.extract_ans import invoke_openai, get_gsmplus
from utils.prompt_template import cot_prompt_map_func, pot_prompt_map_func, ltm_prompt_map_func, complex_prompt_map_func, contrastive_prompt_map_func
import json
import os
import argparse


def inference(input_file="", output_file="", model="turbo", prompt_type="cot", sc_size=5, attack_type=None):
    questions, answers, types = get_gsmplus(input_file, specify_attack=attack_type)
    print(f"data size: {len(questions)}")
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        output_data = []

    for idx, (question, answer, type) in enumerate(zip(questions, answers, types)):
        if idx < len(output_data):
            continue
        item = {
            "question": question,
            "answer": answer,
            "type": type
        }
        if "sc" not in prompt_type:
            if prompt_type == "cot":
                template, instruction = cot_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": template},
                ]
            elif prompt_type == "pot":
                pot_prompt, instruction = pot_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": pot_prompt}  # + '\nLet\'s think step by step\n'
                ]
            elif prompt_type == "complex":
                complex_prompt, instruction = complex_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": complex_prompt},
                ]
            elif prompt_type == "contrastive":
                contrastive_prompt, instruction = contrastive_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": contrastive_prompt},
                ]
            elif prompt_type == "ltm":
                ltm_prompt, instruction = ltm_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": ltm_prompt},
                ]
            else:
                messages = None
            prediction = invoke_openai(messages=messages, model=model)
            item["model_prediction"] = prediction
            item["prompt_type"] = prompt_type
        else:
            predictions = []
            if prompt_type == "cot_sc":
                template, instruction = cot_prompt_map_func(question)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": template},
                ]
                for _ in range(sc_size):
                    prediction = invoke_openai(messages=messages, model=model, temperature=0.7)
                    predictions.append(prediction)
            item["model_prediction"] = predictions
            item["prompt_type"] = prompt_type
        output_data.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    parser.add_argument("--input_file", default="", type=str, required=False, help="")
    parser.add_argument("--output_file", default="", type=str, required=True, help="")
    parser.add_argument("--attack_type", default=None, type=list, required=False, help="")
    parser.add_argument("--prompt_type", default="cot", type=str, required=False, choices=["cot", "pot", "complex", "contrastive", "ltm", "cot_sc"])
    parser.add_argument("--sc_size", default=5, type=int, required=False, help="")

    args = parser.parse_args()

    inference(model=args.model_name,
              input_file=args.input_file,
              output_file=args.output_file,
              prompt_type=args.prompt_type,
              sc_size=args.sc_size,
              attack_type=args.attack_type)
