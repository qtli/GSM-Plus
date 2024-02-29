import json
import argparse
import pdb
import sys
import os
sys.path.append(os.getcwd().split("GSM-Plus")[0] + "GSM-Plus/")
from scripts.utils.extract_ans import write_confusion_matrix, extract_gold_ans, write_results_to_pred_file, parse_pred_ans, get_gsm8k_gsmplus_map

def get_results(pred_file, source="GSM8K", fine_grained=False, gsm8k_value=0.0, prompt_type="", match_pattern="", mv=1, neglect_ncr=False, write=False, model_name="", prompt_analysis=False):
    golds_str = []
    properties = []
    preds_str = []
    types = []

    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        print(len(pred_data))

        for item_idx, item in enumerate(pred_data):
            type = item["type"]
            if neglect_ncr and type == "critical thinking":
                continue
            target = extract_gold_ans(item["answer"])
            golds_str.append(target)

            if not fine_grained:
                if type != "gsm8k":
                    type = "gsmplus"

            properties.append({"source": source if "gsm8k" in pred_file else type, "tag": {}})
            types.append(item["type"])
            preds_str.append(item["model_prediction"])

        assert len(preds_str) == len(golds_str) == len(properties) == len(types)
        prompt_type_list = [prompt_type for _ in range(len(preds_str))]
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties_list=properties, true_type_list=types,
                   prompt_type=prompt_type_list, match_pattern=match_pattern, mv=mv, fine_grained=fine_grained,
                   gsm8k_value=gsm8k_value, neglect_ncr=neglect_ncr, model_name=model_name, prompt_analysis=prompt_analysis)

        if write:
            cur_dir = os.getcwd()
            write_results_to_pred_file(original_pred_data=pred_data, preds=preds, golds=golds, results=results,
                                       pred_file=os.path.join(cur_dir, pred_file), neglect_ncr=neglect_ncr)


def get_confusion_matrix(pred_file, neglect_ncr=False, model_name=""):
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    gsm8k_pred_data = {}
    for item in pred_data:
        if item["type"] == "gsm8k":
            gsm8k_pred_data[item["question"]] = item
    assert len(gsm8k_pred_data) == 1319

    conf_mat = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0
    }
    sample_mat = {
        "tp": [], "tn": [], "fp": [], "fn": []
    }

    init_to_plus, plus_to_init = get_gsm8k_gsmplus_map(input_path="../dataset/gsmplus_test.json")
    print(f"init_to_plus: {len(init_to_plus)}, plus_to_init: {len(plus_to_init)}")

    for i, item in enumerate(pred_data):
        type = item["type"]
        if type == "gsm8k":
            continue
        if neglect_ncr and type == "critical thinking":
                continue
        aq = item["question"]
        aq = aq.strip("\n").strip(" ")
        variation_judge = item["result"]
        gsm8k_q = plus_to_init[aq]
        gsm8k_judge = gsm8k_pred_data[gsm8k_q]["result"]

        if gsm8k_judge and variation_judge:
            conf_mat["tp"] += 1
            sample_mat["tp"].append(i)
        if gsm8k_judge is False and variation_judge is False:
            conf_mat["tn"] += 1
            sample_mat["tn"].append(i)
        if gsm8k_judge and variation_judge is False:
            conf_mat["fn"] += 1
            sample_mat["fn"].append(i)
        if gsm8k_judge is False and variation_judge:
            conf_mat["fp"] += 1
            sample_mat["fp"].append(i)

    write_confusion_matrix(conf_mat=conf_mat, neglect_ncr=neglect_ncr, model_name=model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=True, help="")
    parser.add_argument("--input_file", default="", type=str, required=False, help="")
    parser.add_argument("--prompt_type", default="cot", type=str, required=False)

    args = parser.parse_args()

    fgs = [False, True] # true we show fine-grained performance under each perturbation type
    neglect_ncrs = [False, True] # true we ignore "critical thinking" type, which is challenging for current open-source models


    for neglect_ncr in neglect_ncrs:
        for fg in fgs:
            get_results(pred_file=args.input_file,
                        model_name=args.model_name,
                        prompt_type=args.prompt_type,
                        fine_grained=fg,
                        neglect_ncr=neglect_ncr,
                        write=False)

        get_confusion_matrix(pred_file=args.input_file,
                             neglect_ncr=neglect_ncr,
                             model_name=args.model_name)

