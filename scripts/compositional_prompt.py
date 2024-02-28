import os
import json
import pdb
import time
import random
import re
import argparse
from utils.extract_ans import invoke_openai, get_gsmplus, remove_numbered_prefixes

REQUIREMENTS = f"""Solve the math question step by step. Please start each step with "Step :" and split steps with "\n\n". There are a few things you should be aware of:

- Opt for the most straightforward and simplest approach to perform each reasoning step.
- Carefully process the numerical variables in the question and perform each calculation sequentially.
- Carefully consider each key information provided and incorporate them systematically into your solution.
- Discern any distractions that are irrelevant to the queried answer.
- If there is no valid answer, please conclude with "So the answer is None" at the end of the output."""


def compose_question_context(question, premise):
    question_context = "This is a math question:\nQuestion: " + question + "\n\n" + "The following is key information extracted from the question:\n" + "\n".join(premise)
    return question_context


def compose_only_question_context(question):
    question_context = "This is a math question:\n\nQuestion: " + question
    return question_context

def start_solve_a_question(question_context):
    demonstration_question_context = compose_question_context(
        question="James gets 10 new CDs.  Each CD cost $15.  He gets them for 40% off.  He decides he doesn't like 5 of them and sells them for 40. How much money was he out?",
        premise="#1. James gets 10 new CDs. \n#2. Each CD cost $15, and he gets them for 40% off.\n#3. He sells 5 of them for 40.\n#4. How much money was he out?".split("\n")
    )
    demonstration_first_step = """Step 1: Calculate the price of each CD after the 40% discount.
- Since the original price per CD is $15 and the discount is 40% off, the price per CD after discount is $15 * (1 - 0.40) = $15 * 0.60 = $9\n- Result: $9\n\n"""
    demonstration_question_context = REQUIREMENTS + "\n\n" + demonstration_question_context + "\n\n" + demonstration_first_step

    prompt = demonstration_question_context + question_context + "\n"
    return prompt


def calculate_a_step(question_context, goal, original_goal_calculation="", step_i=1):
    regen_ins = "\n\nIf you have a strong belief that errors exist in the above reasoning step, explain why and regenerate this step.\nOtherwise, output \"Pass\"."

    if step_i == 1:
        to_check_context = "\n\nThe following is the first reasoning step. Carefully review each item in the first reasoning step.\n\n" + goal + "\n" + original_goal_calculation + regen_ins
        prompt = REQUIREMENTS + "\n\n" + question_context + to_check_context
    else:
        to_check_context = "\n\nThe next step is as follows. Carefully review each item in the next reasoning step.\n\n" + goal + "\n" + original_goal_calculation + regen_ins
        prompt = REQUIREMENTS + "\n\n" + question_context + to_check_context
    return prompt


def compose_solution_context(checked_steps, select_mode=False, qs="", next_id=""):
    if select_mode is False:
        instruction = f"""\n\nSolve the math question step by step. Please start each step with "Step :" and split steps with "\n\n". There are a few things you should be aware of:

- Carefully process the numerical variables in the question and perform each calculation sequentially.
- Avoid missing important information.
- Discern any distractions or additional details that are not necessary to solve the question or are irrelevant to the queried answer.
- If there is no valid answer, please conclude with "So the answer is None" at the end of the output."""
    else:
        instruction = f"""\n\nSolve the math question step by step. Please start each step with "Step :" and split steps with "\n\n". There are a few things you should be aware of:

- Carefully process the numerical variables in the question and perform each calculation sequentially.
- List or recall all relevant conditions before conducting each reasoning step to avoid missing important information that could lead to a wrong answer.
- Discern any distractions or additional details that are not necessary to solve the question or are irrelevant to the queried answer."""
    if qs == "":
        solution_context = "\n\nThe following are the first few steps in a solution to the question:\n\n" + "\n\n".join(checked_steps)
    else:
        solution_context = f"\n\nIf the answer of \"{qs}\" has already been calculated, output \"So the answer is [VALUE]\". Otherwise, follow the above reasoning steps and continue with the Step {next_id}.\n\n" + "\n\n".join(checked_steps) + "\n\n"
    return instruction, solution_context


def compose_to_check_context(to_check_steps, step_i=1):
    if step_i == 1:
        instruction = f"""\n\nSolve the math question step by step. Please start each step with "Step :" and split steps with "\n\n". There are a few things you should be aware of:

- Carefully process the numerical variables in the question and perform each calculation sequentially.
- List or recall all relevant conditions before conducting each reasoning step to avoid missing important information that could lead to a wrong answer.
- Discern any distractions or additional details that are not necessary to solve the question or are irrelevant to the queried answer.
- If there is no valid answer, please conclude with "So the answer is None" at the end of the output."""
        to_check_context = "\n\nThe following is the first reasoning step. Each item in the first reasoning step may potentially contain reasoning errors:\n\n" + to_check_steps + "\n\nRegenerate this step from another perspective. Do not copy the given step."
        to_check_context = (instruction + to_check_context)
    else:
        to_check_context = "\n\nThe next step is as follows. Each item in the following step may potentially contain reasoning errors:\n\n" + to_check_steps + "\n\nRegenerate this step from another perspective. Do not copy the given step."
    return to_check_context


def select_step_context(to_select_candidates, step_i=1):
    to_select_candidates_str = ""
    step_targets = []
    for i in range(len(to_select_candidates)):
        t = to_select_candidates[i].split("\n")[0]
        for j in range(len(t)):
            if t[j] == ":":
                t = t[j+1:].strip(" ").lower().rstrip(".")
                break
        step_targets.append(t)
        to_select_candidates_str += ("Candidate " + str(i+1) + ": " + to_select_candidates[i]) + "\n\n"

    if len(set(step_targets)) == 1:
        step_target = step_targets[0]
    else:
        step_target = None

    if step_target:
        instruction = f"Choose the most reasonable candidate that is closest to the final answer and accurately calculates the target value regarding \"{step_target}\". Please only output the index to summarize the final conclusion. Note that candidates with lengthy content do not necessarily imply correctness. Avoid copying the selected candidate's content."
    else:
        instruction = "Choose the most reasonable candidate that takes into account all relevant information regarding its target (first line of the candidate) and accurately calculates its value. Please only output the index to summarize the final conclusion. Note that candidates with lengthy content do not necessarily imply correctness. Avoid copying the selected candidate's content."

    if step_i == 1:
        to_select_context = "\n\nThe following options are potential targets for the first reasoning step:\n\n" + to_select_candidates_str + instruction
    else:
        to_select_context = "\n\nThe following options are potential targets for the current reasoning step:\n\n" + to_select_candidates_str + instruction
    return to_select_context


def select_thought_goal(to_select_candidates, step_i=1):
    to_select_candidates_str = ""
    step_targets = []
    for i in range(len(to_select_candidates)):
        t = to_select_candidates[i].split("\n")[0]
        for j in range(len(t)):
            if t[j] == ":":
                t = t[j + 1:].strip(" ").lower().rstrip(".")
                break
        t = t[0].upper()+t[1:]
        step_targets.append(t)
        to_select_candidates_str += ("Option " + str(i + 1) + ": " + t) + "\n"

    instruction = "\nPlease carefully read the math question and the extracted key information before proceeding. Select the option that receives the highest number of votes among the available choices. In the event of a tie in votes, prioritize selecting a direct and specific option.\nPlease output the index to summarize the final conclusion."  # , explain why, and finish the calculation of the selected option. Keep in mind that lengthy candidates do not necessarily indicate correctness.
    if step_i == 1:
        to_select_context = "\n\nThe following options are potential targets for the first reasoning step:\n\n" + to_select_candidates_str + instruction
    else:
        to_select_context = f"\n\nThe following options represent potential targets for the subsequent reasoning Step {step_i}:\n\n" + to_select_candidates_str + instruction
    return to_select_context


def compose_continue_reason_context(queried_answer, next_id=""):
    qa = remove_numbered_prefixes(queried_answer)
    if next_id == "":
        next_id = "[ID]"
    continue_context = f"""\n\nIf the answer of \"{qa}\" has already been calculated, output "So the answer is [VALUE]". Otherwise, follow the above reasoning steps and continue with the Step {next_id}."""
    return continue_context


def check_answer_status(queried_answer, check_pred_steps, last_goal=""):
    history = "\n\nThe following is the first a few steps in a solution to the problem:\n\n" + "\n\n".join(check_pred_steps)
    if last_goal == "":
        check_context = history + f"\n\nHave we already determined the answer to \"{queried_answer}\"? If we have determined the answer, please output \"So the answer is [VALUE]\". Otherwise, please output \"No\" and explain why."
    else:
        check_context = history + f"\n\nDoes the goal of last step \"{last_goal}\" meets up the target of the math question \"{queried_answer}\"? If yes, please output \"So the answer is [VALUE]\". Otherwise, please output \"No\" and explain why."

    return check_context

def check_premise_consider_status(check_pred_steps, question_context):
    history = "\n\nThe following is the first a few steps in a solution to the problem:\n\n" + "\n\n".join(check_pred_steps)
    a = question_context.split("The following is key information extracted from the question:\n")
    psize = len(a[1].split("\n\n")[0].split("\n"))
    check_context = history + f"\n\nAssess whether the above solution has overlooked any essential key information (#1 - #{psize}). If yes, please output the overlooked information and explain why. Otherwise, ONLY output \"No\"."
    return check_context

def select_one_pred(preds, mode="", target="cal"):
    def remove_step_prefixes(string):
        pattern = r'^Step \d+\:\s'  # Regular expression pattern to match numbered prefixes
        result = re.sub(pattern, '', string)
        if "pass" in result.lower():
            result = preds[0]
        return result

    results = {}
    for i, p in enumerate(preds):
        tmp_p = p.replace(",", "")
        tmp_p = remove_step_prefixes(tmp_p)
        if target == "cal":
            if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', tmp_p)) >= 1:
                tmp_p = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', tmp_p)[-1].replace(",", "")
        if tmp_p not in results:
            results[tmp_p] = [1, [p], [i]]
        else:
            tmp = results[tmp_p][1]
            idx = results[tmp_p][2]
            tmp.append(p)
            idx.append(i)
            results[tmp_p] = [results[tmp_p][0]+1, tmp, idx]

    if mode == "mv":
        if results != {}:
            results_sort = list(zip(results.keys(), results.values()))
            results_sort.sort(key=lambda x: x[1][0], reverse=True)
            high_v = results_sort[0][1][1][0]
            high_i = results_sort[0][1][2][0]
        else:
            high_v = preds[0]
            high_i = 0
        if target == "cal": return high_v
        else:
            return high_v, high_i
    else:
        results_sort = results
        results_sort = dict(results_sort)
        distincts = []
        for k in results_sort:
            distincts.append(results_sort[k][1][0])

        return distincts


def get_id_from_candidate_output(output):
    output = output.lower()

    candidate_id_list = None
    for i in range(len(output)):
        if output[i:].startswith('candidate'):
            try:
                candidate_id_list = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', output[i + 9:])[0]
            except:
                print("no id!!!")
                pdb.set_trace()

    return candidate_id_list


def get_history(intermediate_results, goal_mode=""):
    history = []
    for sid in intermediate_results:
        this_step = intermediate_results[sid]
        if goal_mode == "greedy":
            this_step_goal = this_step["selected_goal"]
        else:
            this_step_goal = this_step["goal_list"][int(this_step["selected_goal_index"])]
        if "selected_calculation" in this_step:
            this_step_calculation = this_step["selected_calculation"]
            history.append(this_step_goal + "\n" + this_step_calculation)
        else:
            history.append(this_step_goal)
    return history


def goal_generation(intermediate_results, question_context, qs="", step_id=1, mode=""):
    print(f"======= goal generation - step {step_id} =======")

    def avoid_hard_mode(pred_str):
        ms = ["set up an equation", "set up equation", "assign variables"]
        ind = False
        for m in ms:
            if m in pred_str.lower():
                ind = True
                break
        return ind

    if step_id == 1:
        prompt = start_solve_a_question(question_context=question_context)
    else:
        history = get_history(intermediate_results, goal_mode=mode)
        next_id = int(list(intermediate_results.keys())[-1]) + 1
        instruction, solution_context = compose_solution_context(history, qs=qs, next_id=str(next_id))
        prompt = instruction.strip("\n") + "\n\n" + question_context + solution_context

    if str(step_id) not in intermediate_results:
        intermediate_results[str(step_id)] = {}

    messages = [{"role": "user", "content": prompt}]
    if mode == "greedy":
        step_pred = invoke_openai(messages=messages, stop=["\n\n"])
        while avoid_hard_mode(step_pred):
            step_pred = invoke_openai(messages=messages, stop=["\n\n"], temperature=0.7)
        intermediate_results[str(step_id)]["selected_goal_raw"] = step_pred
        intermediate_results[str(step_id)]["selected_goal"] = step_pred.split("\n")[0]
        intermediate_results[str(step_id)]["selected_goal_remaining_part"] = '\n'.join(step_pred.split("\n")[1:])
    else:
        step_pred_list = []
        for _ in range(3):
            step_pred = invoke_openai(messages=messages, stop=["\n\n"], temperature=0.7)
            step_pred_list.append(step_pred)
            time.sleep(random.uniform(1, 3))

        intermediate_results[str(step_id)]["goal_list_raw"] = step_pred_list
        intermediate_results[str(step_id)]["goal_list"] = [fs.split("\n")[0] for fs in step_pred_list]
        intermediate_results[str(step_id)]["goal_remaining_part_list"] = ['\n'.join(fs.split("\n")[1:]) for fs in step_pred_list]

    return intermediate_results, prompt


def goal_selection(intermediate_results, question_context, step_id=1, goal_mode="mv"):
    print(f"======= goal selection step {step_id} =======")

    if step_id != 1:
        step_id = list(intermediate_results.keys())[-1]

    if goal_mode == "mv":
        select_pred, select_idx = select_one_pred(intermediate_results[str(step_id)]["goal_list"], mode=goal_mode, target="goal")
        intermediate_results[str(step_id)]["selected_goal"] = select_pred
        intermediate_results[str(step_id)]["selected_goal_index"] = select_idx
        prompt = ""
    else:
        if step_id == 1:
            first_step_checks = intermediate_results[str(step_id)]["goal_list"]
            select_context = select_thought_goal(to_select_candidates=first_step_checks)
            prompt = question_context + select_context
        else:
            history = get_history(intermediate_results, goal_mode="mv")
            instruction, solution_context = compose_solution_context(history)
            step_checks = intermediate_results[step_id]["goal_list"]
            select_context = select_thought_goal(to_select_candidates=step_checks, step_i=int(step_id))
            prompt = question_context + solution_context + select_context

        messages = [{"role": "user", "content": prompt}]
        select_pred = invoke_openai(messages=messages, temperature=0)
        intermediate_results[str(step_id)]["selected_goal"] = select_pred
        intermediate_results[str(step_id)]["selected_goal_index"] = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', select_pred)[0]
        time.sleep(random.uniform(1, 3))

    return intermediate_results, prompt


def goal_calculation(intermediate_results, question_context, step_id=1, goal_mode="", cal_mode=""):
    print(f"======= goal calculation step {step_id} =======")

    def last_value(pred):
        ls = re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', pred.split("\n")[-1])
        if len(ls) >= 1:
            v = ls[-1]
        else:
            v = -999
        return v

    def reformat(first_step_pred, original_goal_calculation):
        if len(first_step_pred.split("\n")) <= 1:
            first_step_pred = original_goal_calculation
        elif len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', first_step_pred.split("\n")[-1])) == 0:
            first_step_pred = original_goal_calculation
        elif first_step_pred.startswith("Step"):
            first_step_pred = "\n".join(first_step_pred.split("\n")[1:])
        return first_step_pred

    def step_reformat_pred(messages, original_goal_calculation):
        first_step_pred = invoke_openai(messages=messages, stop=["\n\n"])
        intermediate_results[str(step_id)]["the_first_correction_calculation"] = first_step_pred
        print("first_step_pred: ", first_step_pred)
        first_step_pred = reformat(first_step_pred, original_goal_calculation)
        p_v = last_value(first_step_pred)
        o_v = last_value(original_goal_calculation)
        if o_v != p_v:
            print(f"======= goal calculation of judge -  step {step_id} =======")
            judge_step_pred = invoke_openai(messages=messages, stop=["\n\n"], temperature=0.7)
            judge_step_pred = reformat(judge_step_pred, original_goal_calculation)
            intermediate_results[str(step_id)]["the_judge_correction_calculation"] = judge_step_pred
            j_v = last_value(judge_step_pred)
            if j_v == o_v:
                first_step_pred = original_goal_calculation
            if j_v != o_v and j_v != p_v:
                first_step_pred = original_goal_calculation

        if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', first_step_pred)) ==0:
            first_step_pred = original_goal_calculation
        return first_step_pred

    # if step_id != 1:
    assert str(step_id) == str(list(intermediate_results.keys())[-1]), pdb.set_trace()
    if goal_mode == "greedy":
        cur_goal = intermediate_results[str(step_id)]["selected_goal"]
        original_goal_calculation = intermediate_results[str(step_id)]["selected_goal_remaining_part"]
    else:
        cur_goal = intermediate_results[str(step_id)]["goal_list"][int(intermediate_results[str(step_id)]["selected_goal_index"])]
        original_goal_calculation = intermediate_results[str(step_id)]["goal_remaining_part_list"][int(intermediate_results[str(step_id)]["selected_goal_index"])]

    if step_id == 1:
        prompt = calculate_a_step(question_context=question_context, goal=cur_goal, original_goal_calculation=original_goal_calculation)
    else:
        history = get_history(intermediate_results, goal_mode=goal_mode)
        instruction, solution_context = compose_solution_context(history)
        prompt = calculate_a_step(question_context=question_context + solution_context, goal=cur_goal,
                                  original_goal_calculation=original_goal_calculation, step_i=int(step_id))
    messages = [{"role": "user", "content": prompt}]

    cal_pred_list = []
    first_pred = step_reformat_pred(messages, original_goal_calculation)
    cal_pred_list.append(first_pred)

    if cal_mode == "greedy":
        intermediate_results[str(step_id)]["selected_calculation"] = first_pred
    else:
        for _ in range(3):
            cal_pred = invoke_openai(messages=messages, stop=["\n\n"], temperature=0.7)
            if cal_pred != "" and "\n" in cal_pred:
                cal_pred = "\n".join(cal_pred.split("\n")[1:])
            cal_pred_list.append(cal_pred)
            time.sleep(random.uniform(1, 3))
        intermediate_results[str(step_id)]["calculation_list"] = cal_pred_list

    return intermediate_results, prompt


def calculation_selection(intermediate_results, step_id=1, goal_mode=""):
    print(f"======= goal generation - step {step_id}=======")

    if step_id == 1:
        calculation_list = intermediate_results[str(step_id)]["calculation_list"]
        select_pred = select_one_pred(calculation_list, mode="mv")
        intermediate_results[str(step_id)]["selected_calculation"] = select_pred
    else:
        cur_step_id = list(intermediate_results.keys())[-1]
        calculation_list = intermediate_results[str(cur_step_id)]["calculation_list"]
        select_pred = select_one_pred(calculation_list, mode="mv")
        if len(re.findall('-?\d+(?:\.\d+)?(?:/\d+)?', select_pred)) <= 0:
            if goal_mode == "greedy":
                select_pred = intermediate_results[str(cur_step_id)]["selected_goal_remaining_part"]
            else:
                select_pred = intermediate_results[str(cur_step_id)]["goal_remaining_part_list"][0]
        intermediate_results[str(cur_step_id)]["selected_calculation"] = select_pred
    return intermediate_results



def answer_progress_check(intermediate_results, question_context, qs, goal_mode=""):
    print("======= answer progress check =======")
    history = get_history(intermediate_results, goal_mode=goal_mode)
    last_goal = history[-1].split("\n")[0]
    for j in range(len(last_goal)):
        if last_goal[j] == ":":
            last_goal = last_goal[j + 1:].strip(" ").rstrip(".")
            break
    check_answer_only_target = check_answer_status(qs, history, last_goal)
    prompt = question_context + check_answer_only_target
    messages = [{"role": "user", "content": prompt}]
    check_answer_result = invoke_openai(messages=messages)
    last_step_id = list(intermediate_results.keys())[-1]
    intermediate_results[last_step_id]["answer_status_by_last_goal"] = check_answer_result
    return intermediate_results, prompt


def premise_consider_check(intermediate_results, question_context, goal_mode=""):
    history = get_history(intermediate_results, goal_mode=goal_mode)
    check_answer_only_target = check_premise_consider_status(history, question_context)
    prompt = question_context + check_answer_only_target
    messages = [{"role": "user", "content": prompt}]
    check_answer_result = invoke_openai(messages=messages)
    if "No" in check_answer_result:
        stp = True
    else:
        stp = False
    return stp

def one_search(item, goal_mode="mv", cal_mode="mv", max_step=15):
    def stop_flag(intermediate_results):
        stop_label = False
        history = get_history(intermediate_results, goal_mode=goal_mode)
        last_goal = history[-1].split("\n")[0]
        last_sid = list(intermediate_results.keys())[-1]
        if "the answer is \"no\"" not in intermediate_results[last_sid]["answer_status_by_last_goal"].lower():
            if "so the answer is" in last_goal.lower():
                stop_label = premise_consider_check(intermediate_results, question_context, goal_mode)
        return stop_label

    question = item["question"]
    premise = item["premise"]
    qs = item["last_query"]

    if "intermediate_res" not in item: intermediate_results = {}
    else: intermediate_results = item["intermediate_res"]
    question_context = compose_question_context(question=question, premise=premise)

    # 1. generate the goal candidates of the first step
    if goal_mode == "greedy": goal_key_name = "selected_goal"
    else: goal_key_name = "goal_list"
    if "1" not in intermediate_results or goal_key_name not in intermediate_results["1"]:
        intermediate_results, prompt = goal_generation(intermediate_results=intermediate_results, question_context=question_context, mode=goal_mode)

    if goal_mode != "greedy":
        # 2. select one as the first step
        if "selected_goal_index" not in intermediate_results["1"]:
            intermediate_results, prompt = goal_selection(intermediate_results=intermediate_results, question_context=question_context)

    # 3. calculate_first_goal
    if (cal_mode == "mv" and "calculation_list" not in intermediate_results["1"]) or (
            cal_mode == "greedy" and "selected_calculation" not in intermediate_results["1"]):
        intermediate_results, prompt = goal_calculation(intermediate_results=intermediate_results, question_context=question_context, goal_mode=goal_mode, cal_mode=cal_mode)

    if cal_mode != "greedy":
        # 4 select_first_calculation
        if "selected_calculation" not in intermediate_results["1"]:
            intermediate_results = calculation_selection(intermediate_results=intermediate_results)

    # 5 check_answer_status
    if "answer_status_by_last_goal" not in intermediate_results["1"]:
        intermediate_results, prompt = answer_progress_check(intermediate_results=intermediate_results, question_context=question_context, qs=qs, goal_mode=goal_mode)

    for _ in range(max_step):
        # 6 generate_next_goals
        if "answer_status_by_last_goal" not in intermediate_results[str(int(list(intermediate_results.keys())[-1]))]:
            next_id = int(list(intermediate_results.keys())[-1])
        else:
            next_id = int(list(intermediate_results.keys())[-1]) + 1

        if str(next_id) not in intermediate_results or goal_key_name not in intermediate_results[str(next_id)]:
            if stop_flag(intermediate_results):
                break

            if str(next_id) not in intermediate_results or "selected_goal" not in intermediate_results[str(next_id)]:
                intermediate_results, prompt = goal_generation(intermediate_results=intermediate_results, question_context=question_context, qs=qs, step_id=next_id, mode=goal_mode)
            if goal_mode == "greedy" and "so the answer is" in intermediate_results[str(next_id)]["selected_goal"].lower():
                intermediate_results[str(next_id)]["selected_calculation"] = intermediate_results[str(next_id)]["selected_goal"]

        # 7 select_next_goal
        if goal_mode != "greedy":
            if "selected_goal_index" not in intermediate_results[str(next_id)]:
                intermediate_results, prompt = goal_selection(intermediate_results=intermediate_results, question_context=question_context, step_id=next_id)

        # 8 calculate_next_goal
        if (cal_mode == "mv" and "calculation_list" not in intermediate_results[str(next_id)]) or (cal_mode=="greedy" and "selected_calculation" not in intermediate_results[str(next_id)]):
            intermediate_results, prompt = goal_calculation(intermediate_results=intermediate_results, question_context=question_context, step_id=next_id, goal_mode=goal_mode, cal_mode=cal_mode)

        if cal_mode != "greedy":
            # 9 select_next_calculation
            if "selected_calculation" not in intermediate_results[str(next_id)]:
                intermediate_results = calculation_selection(intermediate_results=intermediate_results, step_id=next_id)

        # 5 check_answer_status
        last_step_id = list(intermediate_results.keys())[-1]
        if "answer_status_by_last_goal" not in intermediate_results[last_step_id]:
            intermediate_results, prompt = answer_progress_check(intermediate_results=intermediate_results,
                                                                 question_context=question_context, qs=qs, goal_mode=goal_mode)

    # organize the final reasoning chain:
    chain = []
    for step_idx in intermediate_results:
        step_item = intermediate_results[step_idx]
        if goal_mode == "greedy":
            c = step_item["selected_goal"] + "\n" + step_item["selected_calculation"]
        else:
            c = step_item["goal_list"][int(step_item["selected_goal_index"])-1] + "\n" + step_item["selected_calculation"]
        chain.append(c)

    if "answer_status_by_last_goal" in intermediate_results[list(intermediate_results.keys())[-1]] and "so the answer is" in intermediate_results[list(intermediate_results.keys())[-1]]["answer_status_by_last_goal"].lower():
        chain.append(intermediate_results[list(intermediate_results.keys())[-1]]["answer_status_by_last_goal"])
    intermediate_results["final_chain"] = "\n\n".join(chain)

    return intermediate_results


def comp_prompt(input_file, output_file, goal_mode="", cal_mode="", max_step=12):
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        output_data = []

    questions, answers, types = get_gsmplus(input_file=input_file)
    print(f"data_size: {len(questions)}")

    for idx, (q,a,t) in enumerate(zip(questions, answers, types)):
        if idx < len(output_data):
            continue
        item = {"question": q, "answer": a, "type": t}
        intermediate_res = one_search(item, goal_mode=goal_mode, cal_mode=cal_mode, max_step=max_step)
        item["intermediate_res"] = intermediate_res
        output_data.append(item)

        json.dump(output_data, open(output_file, "w"), indent=4)
    json.dump(output_data, open(output_file, "w"), indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", type=str, required=False, help="")
    parser.add_argument("--input_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument("--output_file", default=None, type=str, required=False, help="save your result")
    parser.add_argument('--goal_mode', type=str, default="greedy", choices=["greedy", "mv"])
    parser.add_argument('--cal_mode', type=str, default="greedy", choices=["greedy", "mv"])
    args = parser.parse_args()

    comp_prompt(
        input_file=args.input_file,
        output_file=args.output_file,
        goal_mode=args.gaol_mode,
        cal_mode=args.cal_mode
    )








