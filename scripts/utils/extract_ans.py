import json
import re
import pdb
from sympy.parsing.latex import parse_latex
import numpy as np
import openai
import random
import time
import ast
import sympy
import copy
import func_timeout
from tqdm import tqdm
from collections import defaultdict, Counter
from fraction import Fraction
import sys
import os
sys.path.append(os.getcwd().split("GSM-Plus")[0] + "GSM-Plus/")
from scripts.utils.prompt_template import pot_prompt_map_func


OPENAI_KEY = ""
openai.api_key = OPENAI_KEY

def invoke_openai(messages, model="gpt-3.5-turbo-0613", mode="", max_num_tokens=512, temperature=0.0, stop=None):
    top_p = 1
    if mode == 'pot':
        stop = ['\n\n']
    max_try = 1
    prediction = ""
    while max_try<5:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_num_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            prediction = response['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            time.sleep(random.uniform(3,6))
            max_try +=1
        else:
            break
    return prediction


# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    formatted_target_str = delete_extra_zero(formatted_target_str)
    formatted_prediction_str = delete_extra_zero(formatted_prediction_str)

    flag = False
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True

    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True

    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass  # 占位语句，不做任何处理
    try:
        import unicodedata
        unicodedata.numeric(s) # 把一个表示数字的字符串，转换成浮点数返回
        return True
    except (TypeError, ValueError):
        pass
    return False


def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n

def synthesize_program_my(result: str, prefix: str) -> str:
    program = prefix
    program_cand = prefix

    count = 0
    for i, line in enumerate(result.split('\n')):
        if line.strip(' ') in prefix:
            continue
        if count == 0:
            program += line + '\n'
        else:
            if line.startswith('    '):
                program += line + '\n'
            else:
                break
            if line.startswith('    return '):
                break
        count += 1

    count = 0
    for i, line in enumerate(result.split('\n')):
        if line.strip(' ') in prefix:
            continue

        if line.startswith('    '):
            program_cand += line + '\n'
        else:
            # break
            line = '    ' + line
            program_cand += line + '\n'
        if line.startswith('    return '):
            break
        count += 1

    program += 'ans = solver()'
    program = program.replace(' [/INST]', '    # [/INST]')
    program_cand += 'ans = solver()'
    program_cand = program_cand.replace(' [/INST]', '    # [/INST]')

    return program, program_cand


def synthesize_program_sego(result: str, prefix: str) -> str:
    result += '\nans = solve()'
    return result


def safe_execute_sego(code_string: str, keys=None):
    GLOBAL_DICT = {}
    _global_vars = copy.copy(GLOBAL_DICT)
    def execute(x):
        try:
            exec(x, globals()) in globals(), globals()
            locals_ = locals()
            globals_ = globals()

            if keys is None:
                res = locals_.get('ans', None) or globals_.get('ans', None)
                if res is None:
                    globals_.get('ans', None)
                    exec(x)
                    locals_ = locals()
                    res = locals_.get('ans', None)
                return res
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(10, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None
    return ans



def safe_execute(code_string: str, keys=None):
    GLOBAL_DICT = {}
    _global_vars = copy.copy(GLOBAL_DICT)
    def execute(x):
        try:
            exec(x, globals()) in globals(), globals()
            locals_ = locals()
            globals_ = globals()
            if keys is None:
                res = locals_.get('ans', None) or globals_.get('ans', None)
                if res is None:
                    globals_.get('ans', None)
                    exec(x)
                    locals_ = locals()
                    res = locals_.get('ans', None)
                return res
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(10, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans


def simplify_ans(ans, convert_to_str: bool = True):
    if 'relational' in str(type(ans)):
        return str(ans)
    elif 'numpy' in str(type(ans)):
        if ans.shape == ():
            # scalar value
            ans = round(float(ans), 2)
        else:
            # array value
            ans = round(float(ans[0]), 2)
        if convert_to_str:
            return str(ans)
        else:
            return ans
    elif not ans:
        return None
    else:
        if type(ans) in [list, tuple]:
            if 'sympy' in str(type(ans[0])):
                try:
                    ans = [round(float(x), 2) for x in ans]
                except Exception:
                    ans = [str(x) for x in ans]
            if len(ans) == 1:
                ans = ans[0]
        else:
            if 'sympy' in str(type(ans)):
                try:
                    ans = round(float(ans), 2)
                except Exception:
                    ans = str(ans)
        if convert_to_str:
            return str(ans)
        else:
            return ans


def extract_gold_ans(answer_str):
    answer_str = answer_str.strip("\n").strip(" ").rstrip(".").replace(",", "")
    pattern = "####(.*)"
    if len(re.findall(pattern, answer_str)) >= 1:
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    else:
        pattern = "boxed{(.*)}"
        if len(re.findall(pattern, answer_str)) < 1:
            pdb.set_trace()
        target = re.findall(pattern, answer_str)[-1].strip(' ')
    if target != "None":
        if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', target)) < 1:
            print(answer_str)
            pdb.set_trace()
        temp_ans = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', target)[0]
        temp_ans = delete_extra_zero(temp_ans)
    else:
        temp_ans = "None"
    return temp_ans


def extract_pred_ans(pred_str, prompt_type="", match_pattern="", input_str=""):
    if "code" not in prompt_type:
        pred_str = pred_str.rstrip(".").replace(",", "")
    if prompt_type == "complex":
        if "The answer is " not in pred_str:
            pred = ""
        else:
            pred_str = pred_str.split("The answer is ")[1].strip(".").strip(" ")
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)) >= 1:
                pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)[-1])
            else:
                pred = ""

    elif prompt_type == "cot":
        pattern = "####(.*)"
        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]
        preds = re.findall(pattern, pred_str)
        pred = delete_extra_zero(preds[-1].strip(" ")) if len(preds) >= 1 and bool(re.search(r"\d", preds[-1])) else ""
        if pred == "":
            pred = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)
            if len(pred) >= 1:
                pred = delete_extra_zero(pred[-1].replace(",", "").strip(".").strip(" "))
            else:
                pred = ""
        else:
            pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred.replace(",", ""))[0].strip(".").strip(" "))
        if "</s>" in pred:
            pred = pred[:-4]

    elif prompt_type == "pot":
        input_str = pot_prompt_map_func(input_str)
        pred_str = pred_str.replace("\n\n", "\n")
        program, program_cand = synthesize_program_my(pred_str, input_str)
        pred = safe_execute(program)
        pred_cand = safe_execute(program_cand)
        pred = pred or pred_cand
        pred = floatify_ans(simplify_ans(pred, True))  # convert_to_str = True
        if "(" in str(pred):
            # print(pred)
            pred = ast.literal_eval(str(pred))
            try:
                pred = sum(pred)
            except Exception:
                pred = pred[-1]
        pred = normalize_final_answer(str(pred))
        pred = delete_extra_zero(pred)
        # if pred == "None":
        #     pdb.set_trace()

    elif prompt_type == "ltm" or prompt_type == "decompose" or prompt_type == "metamath":
        if "The answer is:" not in pred_str:
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)) >= 1:
                pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)[-1])
            else:
                pred = ""
        else:
            pred_str = pred_str.split("The answer is:")[1].strip(".").strip(" ")
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)) < 1:
                pred = ""
            else:
                pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)[-1])

    elif prompt_type == "llama":
        pattern = "#### (.*)$"

        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]
        preds = re.findall(pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""
        if "</s>" in pred:
            pred = pred[:-4]
        pred = normalize_final_answer(pred)

    elif prompt_type == "codellama":
        result_counter = Counter()
        ans = safe_execute(pred_str)
        ans = floatify_ans(ans)
        if ans is not None:
            result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = "None"
        pred = delete_extra_zero(prediction)
        if pred == "":
            pred = "None"
    elif prompt_type == "mammoth" or prompt_type == "metamath" or prompt_type == "tora" or prompt_type == "sego1" or prompt_type == "codellama1":
        pred = pred_str  # todo

    elif prompt_type == "sego":
        pred_code = synthesize_program_sego(pred_str, prefix="")
        result_counter = Counter()
        ans = safe_execute_sego(pred_code)
        ans = floatify_ans(ans)
        if ans is not None:
            result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = "None"
        pred = delete_extra_zero(prediction)
        if pred == "":
            pred = "None"

    else:
        pred_str = pred_str.lower()
        if match_pattern not in pred_str:
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)) >= 1:
                pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)[-1])
            else:
                pred = ""
        else:
            if "\n" in pred_str:
                if match_pattern in pred_str.split("\n")[-1]:
                    pred_str = pred_str.split("\n")[-1]
            pred_str = pred_str.split(match_pattern)[1].strip(".").strip(" ")
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)) < 1:
                pred = ""
            else:
                pred = delete_extra_zero(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred_str)[0])
    if "code" not in prompt_type:
        pred = pred.rstrip(".").strip(" ")
    return pred


def extract_pred_ans_none(pred_str, prompt_type="", match_pattern="####"):
    pred_str = pred_str.rstrip(".").replace(",", "").lower()
    pred = ""
    patterns = ["does not provide enough information", "does not specify", "does not provide", "can't provide", "can not provide", "don't know", "do not know", "doesn't specify", "not specify", "not mention", "doesn't mention", "don't have enough information", "do not have enough", "not provide", "doesn't provide", "cannot calculate", "can't calculate", "can't determine", "cannot determine", "missing necessary information", "none"]
    for p in patterns:
        if p in pred_str:
            pred = "None"
    if "cot" in prompt_type:
        match_pattern = "####"
    elif prompt_type == "ltm":
        match_pattern = "the answer is:"
    elif prompt_type == "complex":
        match_pattern = "the answer is "
    elif prompt_type == "codellama":
        result_counter = Counter()
        ans = safe_execute(pred_str)
        ans = floatify_ans(ans)
        if ans is not None:
            result_counter.update([ans])

        if len(result_counter) > 0:
            prediction = result_counter.most_common(1)[0][0]
        else:
            prediction = "None"
        pred = delete_extra_zero(prediction)
        if pred == "":
            pred = "None"
    else:
        match_pattern = match_pattern.lower()
    if pred != "None" and match_pattern not in pred_str:
        pred = "None"
    return pred


def extract_answer_number_mistral(completion):
    text = completion.split('####')
    if len(text) > 1:
        extract_ans = text[-1].strip().strip("\n")
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def parse_pred_ans(preds_str, golds_str, properties_list, true_type_list,
                   prompt_type="cot", match_pattern="", mv=1, fine_grained=False,
                   gsm8k_value=0.0, neglect_ncr=False, model_name="", prompt_analysis=False, input_str_list=None):
    num_q = 0
    acc = 0
    results = []
    preds = []
    golds = []
    correct_table = {}
    cnt_table = {}
    source_set = set()

    dir_name = os.getcwd().split("GSM-Plus")[0] + "GSM-Plus/"
    all_model_performance_file = os.path.join(dir_name, "results", "model_performance_all.json")
    all_model_performance = json.load(open(all_model_performance_file))
    if model_name not in all_model_performance:
        all_model_performance[model_name] = {}
    if input_str_list is None:
        input_str_list = ["" for _ in range(len(preds_str))]
    for pred_str, gold_str, properties, true_type, input_str, ptype in tqdm(zip(preds_str, golds_str, properties_list, true_type_list, input_str_list, prompt_type), total=len(preds_str)):
        source = properties['source']
        source_set.add(source)

        num_q += 1
        result, pred, gold = test_answer(prompt_type=ptype, match_pattern=match_pattern,
                                         pred_str=pred_str, ans_str=gold_str, mv=mv, source=true_type, input_str=input_str)
        results.append(result)
        preds.append(pred)
        golds.append(gold)
        if result:
            acc += 1

        if source not in correct_table.keys():
            correct_table[source] = 1 if result else 0
            cnt_table[source] = 1
        else:
            correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
            cnt_table[source] += 1


    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    acc_table = {}
    for key in correct_table.keys():
        acc_table[key] = correct_table[key] / cnt_table[key]
    acc_table = list(zip(acc_table.keys(), acc_table.values()))
    acc_table.sort(key=lambda x: x[1])

    fine_type_list = ["numerical substitution", "digit expansion", "integer-decimal-fraction conversion",
                 "problem understanding", "adding operation", "distractor insertion", "reversing operation"]
    coarse_type_list = ["gsm8k", "gsmplus"]

    if fine_grained:
        type_list = fine_type_list
    else:
        type_list = coarse_type_list

    if fine_grained and gsm8k_value != 0:
        all_data_file = "decay_rate_models.json"
        perf_change_of_all_models_data = json.load(open(all_data_file))

        all_acc_table_color = {}
        all_decay_table_color = {}
        print(acc_table)
        for key, acc in acc_table:
            this_acc = str(round(acc*100, 2))
            all_acc_table_color[key] = float(this_acc)
            all_decay_table_color[key] = str(round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))
        all_acc_str = ""
        all_decay_str = ""
        all_acc_list = [gsm8k_value]
        all_decay_list = [-50]

        if neglect_ncr is False:
            type_list.append("critical thinking")

        for key in type_list:
            all_acc_str += str(all_acc_table_color[key]) + ", "
            all_acc_list.append(all_acc_table_color[key])
            all_decay_str += all_decay_table_color[key] + ", "
            all_decay_list.append(all_decay_table_color[key])
        print("all_acc: ", all_acc_str)
        print("all_decay: ", all_decay_str)

        perf_change_of_all_models_data["decay_rates"][model_name] = all_decay_list
        perf_change_of_all_models_data["accuracies"][model_name] = all_acc_list
        if prompt_analysis is False:
            json.dump(perf_change_of_all_models_data, open(all_data_file, "w"), indent=4)

    else:
        for key, acc in acc_table:
            if key in source_set:
                print(key + ": " + str(acc))
                if key == "gsmplus" and neglect_ncr:
                    key = "gsmplus_wo_ncr"
                all_model_performance[model_name][key] = str(round(acc*100, 2))
            else:
                print("    " + key.split(",")[-1] + " : " + str(acc))

    if prompt_analysis is False:
        json.dump(all_model_performance, open(all_model_performance_file, "w"), indent=4)
    else:
        ps = "["
        if neglect_ncr is False:
            type_list.append("critical thinking")
        for k in type_list:
            ps += (str(round(dict(acc_table)[k]*100,2)) + ", ")
        ps += "]"
        print(ps)
    return results, preds, golds


def write_results_to_pred_file(original_pred_data, preds, golds, results, pred_file, neglect_ncr=False, mv=1, type="json"):
    new_pred_data = []
    for i, item in enumerate(original_pred_data):
        if neglect_ncr and item["type"] == "critical thinking":
            new_pred_data.append(item)
            continue
        if mv == 1:
            item['pred'] = preds[i]
        else:
            item['pred'] = preds[i][0]
            item["pred_list"] = preds[i][1]
        item['gold'] = golds[i]
        item['result'] = results[i]
        new_pred_data.append(item)

    # Save the updated list of dictionaries back to the jsonl file
    assert len(new_pred_data) == len(original_pred_data)
    if type == "json":
        with open(pred_file, 'w') as file:
            json.dump(new_pred_data, file, indent=4)
    else:
        with open(pred_file, 'w') as file:
            for item in new_pred_data:
                file.write(json.dumps(item) + "\n")


def write_confusion_matrix(conf_mat, neglect_ncr=False, model_name=""):
    all_model_performance_file = "model_performance_all.json"
    all_model_performance = json.load(open(all_model_performance_file))
    if model_name not in all_model_performance:
        all_model_performance[model_name] = {}

    conf_mat_percentage = conf_mat.copy()
    true_preference = conf_mat["tp"] + conf_mat["fn"]
    false_prefernce = conf_mat["tn"] + conf_mat["fp"]
    conf_mat_percentage["tp"] = conf_mat["tp"]/true_preference
    conf_mat_percentage["fn"] = conf_mat["fn"]/true_preference
    conf_mat_percentage["tn"] = conf_mat["tn"]/false_prefernce
    conf_mat_percentage["fp"] = conf_mat["fp"]/false_prefernce

    print(conf_mat_percentage)

    if neglect_ncr:
        all_model_performance[model_name]["conf_mat_wo_ncr"] = conf_mat_percentage
        conf_mat["true"] = true_preference
        conf_mat["false"] = false_prefernce
        all_model_performance[model_name]["conf_mat_wo_ncr_count"] = conf_mat
    else:
        all_model_performance[model_name]["conf_mat"] = conf_mat_percentage
        conf_mat["true"] = true_preference
        conf_mat["false"] = false_prefernce
        all_model_performance[model_name]["conf_mat_count"] = conf_mat

    json.dump(all_model_performance, open(all_model_performance_file, "w"), indent=4)



def test_answer(pred_str="", ans_str="", prompt_type="", match_pattern="", mv=1, source="", input_str=""):
    gold = ans_str
    if gold != "None":
        gold = normalize_final_answer(gold)

    if mv == 1:
        if source == "critical thinking":
            pred = extract_pred_ans_none(pred_str, prompt_type=prompt_type, match_pattern=match_pattern)
        else:
            pred = extract_pred_ans(pred_str, prompt_type=prompt_type, match_pattern=match_pattern, input_str=input_str)
    else:
        pss = defaultdict(int)
        pss_list = []
        for ps in pred_str[:mv]:
            if source == "critical thinking":
                pred = extract_pred_ans_none(ps, prompt_type=prompt_type, match_pattern=match_pattern)
            else:
                pred = extract_pred_ans(ps, prompt_type=prompt_type, match_pattern=match_pattern, input_str=input_str)
            pss[pred] += 1
            pss_list.append(pred)

        results_sort = list(zip(pss.keys(), pss.values()))
        results_sort.sort(key=lambda x: x[1], reverse=True)
        pred = results_sort[0][0]

    if pred != "None":
        pred = normalize_final_answer(str(pred))
    if mv == 1:
        return check_sympy_equivalence(gold, pred), pred, gold
    else:
        return check_sympy_equivalence(gold, pred), [pred, pss_list], gold


def check_acc(input_file, fine_grained=False, new_file=False):
    input_data = json.load(open(input_file))
    results = {}
    new_data = []
    for item in input_data:
        gold = item["gold"]
        if "pred" not in item:
            item["pred"] = extract_pred_ans(item["model_prediction"])
        pred = item["pred"]
        if fine_grained:
            type = item["type"]
        else:
            type = "default"
        if type not in results:
            results[type] = []
        results[type].append(gold == pred)
        if new_file:
            item["result"] = (gold==pred)
            new_data.append(item)
    for key in results:
        acc = np.array(results[key]).sum()/len(results[key])
        print(f"{key}: {acc}")
    if new_file:
        json.dump(new_data, open(input_file, "w"), indent=4)
    return results


def split_sentences(text):
    sentences = re.split(r'(?<=[.,])\s+', text)
    return sentences


def is_question_sentence(sentence):
    question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'calculate', 'determine']

    # Check if the sentence ends with a question mark
    if sentence.endswith('?'):
        return True

    # Check if any question words are present in the sentence
    for word in question_words:
        if word in sentence.lower():
            return True

    return False


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # Unicode范围包含中文字符
    matches = re.findall(pattern, text)
    return bool(matches)


def get_fine_grained_perf_change(acc_table, gsm8k_value=0.0, only_delta=False):
    color_list = [
        "\cellcolor{DarkGreen!75}", "\cellcolor{DarkGreen!65}", "\cellcolor{DarkGreen!55}",
        "\cellcolor{DarkGreen!45}",
        "\cellcolor{DarkGreen!35}", "\cellcolor{DarkGreen!25}", "\cellcolor{DarkGreen!18}",
        "\cellcolor{DarkGreen!13}"
    ]
    acc_table_color = {}
    all_acc_table_color = {}
    all_decay_table_color = {}
    prev_number = 0
    color_idx = -1
    for key, acc in acc_table:
        this_acc = str(round(acc * 100, 2))
        all_acc_table_color[key] = float(this_acc)
        if this_acc != prev_number:
            color_idx += 1
        if float(this_acc) < gsm8k_value:
            if only_delta:
                acc_table_color[key] = color_list[color_idx] + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))
            else:
                acc_table_color[key] = color_list[color_idx] + "$\\text{" + this_acc + "}_{\\text{" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 1)) + "}}$"
        else:
            if only_delta:
                acc_table_color[key] = "\cellcolor{lightred}" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))
            else:
                acc_table_color[key] = "\cellcolor{lightred}$\\text{" + this_acc + "}_{\\text{" + str(
                    round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 1)) + "}}$"
        all_decay_table_color[key] = str(round(((gsm8k_value - float(this_acc)) / gsm8k_value) * 100, 2))

    color_str = ""
    all_acc_str = ""
    all_decay_str = ""

    for key in ["numerical substitution", "digit expansion", "integer-decimal-fraction conversion",
                "problem understanding", "adding operation", "distractor insertion", "reversing operation"]:
        print(key + ": " + acc_table_color[key])
        color_str += (acc_table_color[key] + " & ")
        all_acc_str += str(all_acc_table_color[key]) + ", "
        all_decay_str += all_decay_table_color[key] + ", "
    color_str = color_str.strip(" ").rstrip("&") + " \\"
    print(color_str)
    print("all_acc: ", all_acc_str)
    print("all_decay: ", all_decay_str)

def remove_numbered_prefixes(string):
    pattern = r'^#\d+\.\s'  # Regular expression pattern to match numbered prefixes
    result = re.sub(pattern, '', string)
    return result


def get_gsmplus(input_file="", specify_attack=None):
    if input_file=="" or input_file is None:
        from datasets import load_dataset
        dataset = load_dataset("qintongli/GSM-Plus")['test']
    else:
        dataset = open(input_file).readlines()

    questions = []
    answers = []
    types = []

    for i, item in enumerate(dataset):
        if specify_attack is not None and item["perturbation_type"] not in specify_attack:
            continue
        questions.append(item["question"])
        answers.append(item["solution"])
        types.append(item["perturbation_type"])
    return questions, answers, types


def get_gsm8k(input_path):
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            item = json.loads(line)
            data.append({
                "question": item["question"],
                "answer": item["answer"]
            })
    return data

def get_gsm8k_gsmplus_map(input_path="gsmplus_test.json"):
    plus_to_init = {}
    init_to_plus = {}
    with open(input_path, encoding='utf-8') as f:
        annotation_data = json.load(f)
        for i, question in enumerate(annotation_data):
            item = annotation_data[question]
            init_to_plus[question] = []
            for attack in item["perturbation_questions"]:
                aq = item["rewritten_questions"][attack]["question"].strip("\n").strip(" ")
                init_to_plus[question].append(aq)
                plus_to_init[aq] = question
    return init_to_plus, plus_to_init
