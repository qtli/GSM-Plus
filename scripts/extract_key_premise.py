import os
import json
import time
import random
import argparse
import string
import sys
import os
sys.path.append(os.getcwd().split("GSM-Plus")[0] + "GSM-Plus/")
from scripts.utils.extract_ans import remove_numbered_prefixes, extract_gold_ans, invoke_openai, get_gsmplus, split_sentences, is_question_sentence

def extract_premise_prompt_map(question, mode="extract_q_premise"):
    natural_program_format_few_shot = [
        "Question:\nJames gets 10 new CDs.  Each CD cost $15.  He gets them for 40% off.  He decides he doesn't like 5 of them and sells them for 40. How much money was he out? \n\nAnswer:\nFirst, let us rewrite the question with labels.\n#1. James gets 10 new CDs. \n#2. Each CD cost $15, and he gets them for 40% off.\n#3. He sells 5 of them for 40.\n#4. How much money was he out?",
    ]
    natural_program_format_suffix = "\n\nAnswer:\nFirst, let us rewrite the question with labels.\n"
    prompts = {
        "extract_q_premise": {"prefix": natural_program_format_few_shot, "suffix": natural_program_format_suffix},
    }
    return "\n".join(prompts[mode]["prefix"]) + "\n\nQuestion:\n" + question + prompts[mode]["suffix"]

def extract_premise_func(args):
    if os.path.exists(args.output_file):
        with open(args.output_file, encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        output_data = []

    questions, answers, types = get_gsmplus(args.input_file)[args.start_idx: args.end_idx]
    print(f"data_size: {len(questions)}")

    for idx, (question, answer, type) in enumerate(zip(questions, answers, types)):
        if idx < len(output_data):
            continue
        question = question.rstrip("\n")
        item = {
            "idx": idx,
            "question": question,
            "answer": answer,
            "gold": extract_gold_ans(answer),
            "type": type,
        }
        time.sleep(random.uniform(1,3))
        extract_prompt = extract_premise_prompt_map(question=question, mode="extract_q_premise")
        model_predictions = []
        messages = [{"role": "user", "content": extract_prompt}]
        model_prediction = invoke_openai(messages=messages, stop=['\n\n'])
        model_predictions.append(model_prediction)

        for _ in range(4):
            model_prediction = invoke_openai(messages=messages, temperature=0.7, stop=['\n\n'])
            model_predictions.append(model_prediction)

        item["model_prediction"] = model_predictions
        output_data.append(item)
        json.dump(output_data, open(args.output_file, "w"), indent=4)
    json.dump(output_data, open(args.output_file, "w"), indent=4)


def sents_similarities(model, qc_cand_list):
    def select_string_with_certain_words(string_list, mode=""):
        '''
        # Example usage
        strings = ["Hello world", "This is a sample", "Python code", "Least words", "yes!"]

        least_words_string = select_string_with_least_words(strings)
        print("String with least words:", least_words_string)
        '''
        if mode == "longest":
            least_words_string = max(string_list, key=lambda x: len(x.split()))
        else:
            least_words_string = min(string_list, key=lambda x: len(x.split()))
        return least_words_string

    def remove_punctuations(input_string):
        # Create a translation table using the string.punctuation constant
        translator = str.maketrans('', '', string.punctuation)

        # Remove punctuations from the input string
        result = input_string.translate(translator)

        return result

    # print("cand_list: ", qc_cand_list)
    similarities = model.similarity(qc_cand_list, qc_cand_list)
    cands_to_sim_cands = {}
    cands_recorded = []
    for i in range(len(qc_cand_list)):
        if qc_cand_list[i] in cands_recorded:
            continue
        for j in range(i + 1, len(qc_cand_list)):
            sim_score = similarities[i][j]
            # print(sim_score)
            if sim_score > 0.8:
                if qc_cand_list[i] not in cands_to_sim_cands:
                    cands_to_sim_cands[qc_cand_list[i]] = [qc_cand_list[j]]
                else:
                    cands_to_sim_cands[qc_cand_list[i]].append(qc_cand_list[j])
                cands_recorded.append(qc_cand_list[j])
            else:
                if qc_cand_list[i] not in cands_to_sim_cands:
                    cands_to_sim_cands[qc_cand_list[i]] = []

                if qc_cand_list[i] not in cands_recorded:
                    cands_recorded.append(qc_cand_list[i])

    distinct_cands = []
    for c in cands_to_sim_cands:
        if len(cands_to_sim_cands[c]) <= 1:
            continue  # voting 少数，淘汰掉
        cands = [c] + cands_to_sim_cands[c]
        select_one = select_string_with_certain_words(cands, mode="longest")
        distinct_cands.append(select_one)

    # print("distinct_cands: ", distinct_cands)
    new_distinct_cands = []
    for c in distinct_cands:
        if c not in new_distinct_cands:
            new_distinct_cands.append(c)
    distinct_cands = new_distinct_cands

    # print("distinct_cands: ", distinct_cands)

    distinct_cands_str = ""
    distinct_cands_new = []
    distinct_cands_new_wo_predix = []
    count = 1
    for c in distinct_cands:
        if remove_punctuations(c).lower() in distinct_cands_str:
            continue
        else:
            prefix = f"#{count}. "
            distinct_cands_new.append(prefix + c)
            distinct_cands_new_wo_predix.append(c)
            count += 1
        distinct_cands_str += (remove_punctuations(c).lower() + " ")
    # print(distinct_cands_new)
    # pdb.set_trace()
    return distinct_cands_new, distinct_cands_new_wo_predix, cands_to_sim_cands


def extract_distinct_qcs(input_file="", output_file=""):
    def rank_lists_by_size(lists):
        ranked_lists = sorted(lists, key=len, reverse=True)
        return ranked_lists

    from simcse import SimCSE
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    data = json.load(open(input_file))

    mew_data = []
    for item in data:
        model_prediction = item["model_prediction"]
        all_qcs = []
        ranked_qcs = []
        for pred_qc in model_prediction:
            qcs = pred_qc.split("\n")
            ranked_qcs.append(qcs)
        ranked_qcs = rank_lists_by_size(ranked_qcs)
        for qcs in ranked_qcs:
            for qc in qcs:
                qc = remove_numbered_prefixes(qc)
                all_qcs.append(qc)
        distinct_qc, distinct_qc_wo_predix, cands_to_sim_cands = sents_similarities(model, all_qcs)

        cursor = -1
        lq = split_sentences(item["question"])[cursor]
        if len(lq.split(" ")) < 4:
            cursor -= 1
            lq = " ".join(split_sentences(item["question"])[cursor:])
        while is_question_sentence(lq) is False:
            lq = " ".join(split_sentences(item["question"])[cursor:])
            cursor -= 1
        item["last_query"] = lq
        item["premise"] = distinct_qc
        item["premise_wo_prefix"] = distinct_qc_wo_predix
        mew_data.append(item)

    json.dump(mew_data, open(output_file, "w"), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=False, help="")
    parser.add_argument("--input_file", default="", type=str, required=False, help="save your result")
    parser.add_argument("--output_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument("--start_idx", default=0, type=int, required=False, help="save your result")
    parser.add_argument("--end_idx", default=5, type=int, required=False, help="save your result")
    parser.add_argument('--prompt_type', type=str, default='cot-nshot', choices=["cot-nshot", "pot-nshot", "ltm-nshot", "ltm-1shot", "complex", "contrastive"])
    args = parser.parse_args()

    # 1 extract premise candidates
    extract_premise_func(args)

    # 2 combine these premise candidates
    extract_distinct_qcs(
        input_file=args.output_file,
        output_file=args.output_file)

