# GSM-Plus: Evaluating the Robustness of LLMs as Mathematical Problem Solvers

![Mathematical Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-red) 
![LLMs](https://img.shields.io/badge/Model-LLMs-green)

Dataset and Code for the Paper "[A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers](https://arxiv.org/abs/2402.19255)".

<div align="center">
  🌐 <a href="https://qtli.github.io/GSM-Plus/">Project Page</a> |
  📚 <a href="https://huggingface.co/datasets/qintongli/GSM-Plus">Data</a> |
  📃 <a href="https://arxiv.org/abs/2402.19255">Paper</a>
</div>


## Introduction

Large language models (LLMs) have achieved impressive performance across various mathematical reasoning benchmarks. 
However, there are increasing debates regarding whether these models truly understand and apply mathematical knowledge or merely rely on shortcuts for mathematical reasoning. 
One essential and frequently occurring evidence is that when the math questions are slightly changed, LLMs can behave incorrectly.

In response to these issues, we present GSM-Plus, an adversarial evaluation benchmark that is crafted for systematically evaluating the mathematical reasoning capability of LLMs. 
we identify 5 perspectives to guide the development of GSM-Plus: 
1. **Numerical Variation** refers to altering the numerical data or its types, including 3 subcategories: **numerical substitution**, **digit expansion**, and **integer-decimal-fraction conversion**.
2. **Arithmetic Variation** refers to reversing or introducing additional operations (e.g., addition, subtraction, multiplication, and division) to math problems, including 2 subcategories: **adding operation** and **reversing operation**.
3. **Problem Understanding** refers to rephrasing the text description of the math problems.
4. **Distractor Insertion** refers to inserting topic-related but useless sentences to the problems.
5. **Critical Thinking** focuses on question or doubt ability when the question lacks necessary statements. 

Based on the 1,319 test questions from GSM8K, we create eight variations for each question, the yielding GSM-Plus comprises **10,552** question variations.

## What's New
- **[2024.07.07]** 🌟 An updated version of GSM-Plus, along with a testmini set are accessible at [Huggingface Datasets](https://huggingface.co/datasets/qintongli/GSM-Plus). This updated version, labeled v1, includes fixes for unrealistic numbers and ambiguous descriptions that were present in the initial version of GSM-Plus (v0).
- **[2024.05.16]** 🎉 Our GSM-Plus paper has been accepted by ACL 2024! 🍻 Cheers!
- **[2024.02.25]** 📣 Dataset [GSM-Plus v0](https://huggingface.co/datasets/qintongli/GSM-Plus-v0) is released.

## TODO

  - [x] Release a compact test set for quick evaluation

## Dataset Example

Examples of eight adversarial questions based on a seed question:

<p align="center">
    <img src="./assets/example1.jpg" width="70%"> <br>
</p>

## Dataset Usage

### Data Downloading

You can download GSM-Plus **test** set and a smaller subset **testmini** by the following command (make sure that you have installed [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart)):

We highly recommend downloading the latest version of GSM-Plus, which was released on July 7th and can be found on [Huggingface Datasets](https://huggingface.co/datasets/qintongli/GSM-Plus). 
This version, GSM-Plus v1, addresses some issues that were present in the previous version, GSM-Plus v0, which was released in February. 
Specifically, it resolves the problem of unrealistic numbers (e.g., ages over 100) and question contexts that do not strictly adhere to specific perturbations.

We have prepared the entire test split of GSM-Plus, as well as a smaller randomly-sampled subset called testmini. 
To ensure consistency, we have re-run the main experiments and verified that the LLMs exhibit similar performance on both versions of the GSM-Plus dataset, including its testmini subset.

- test: 10,552 examples for standard evaluation. Each question of GSM8K's test set is associated with eight variations.
- testmini: 2,400 examples used for model development, fast evaluation, or for those with limited computing resources.


```python
from datasets import load_dataset

dataset = load_dataset("qintongli/GSM-Plus")
# print the first example
print(dataset["test"][0])
print(dataset["testmini"][0])
```

The dataset is provided in `.jsonl` format and contains the following attributes:

```
{
    "question": the adversarial question,
    "solution": the solution chain for the adversarial question ,
    "answer": the gold answer of the adversarial question,
    "perturbation_type": the perturbation type,
    "seed_question": the seed question used to craft the adversarial question,
    "seed_solution": the solution chain for the seed question,
    "seed_answer": the gold answer of the seed question,
}
```

## Evaluations on GSM-Plus

### Usage Examples
The evaluation of GSM-Plus is equivalent to that of the GSM8K dataset. 
Here, we offer a few examples for evaluating LLMs on GSM-Plus, as follows:

- GPT-4, GPT-Turbo-3.5

```
python scripts/openai_model_inference.py --model_name [MODEL_NAME] --output_file [OUTPUT_PATH]/[MODEL_NAME]_prediction.json --prompt_type [PROMPT]
# [MODEL_NAME] can be: gpt-4-0613, gpt-3.5-turbo-0613, etc.
# [PROMPT] can be: cot, pot, ltm, complex, contrastive, cot_sc
```

- Mistral, LLaMA, CodeLlama

```
python scripts/general_model_inference.py \
--model_name [MODEL_NAME] \
--output_file [OUTPUT_PATH]/[MODEL_NAME]_prediction.json \
--model_dir [MODEL_PATH] \  # path to checkpoint files
--nshots 8 \
--prompt_type [PROMPT] \
--specify_your_gpus [GPUT_IDs]

# [PROMPT] can be: cot-nshot, pot-nshot, ltm-nshot, ltm-1shot, complex, contrastive
```

## Compositional Prompting

Extract key premise from a math word problem:

```
python scripts/extract_key_premise.py --outut_file comp_test.json --start_idx 0 --end_idx 10
```

Iteratively generate each reasoning thought based on the problem and its premise:

```
python scripts/compositional_prompt.py --input_file comp_test.json --output_file comp_prediction.json --goal_mode greedy --cal_mode greedy
```

## ️Citation
If you find **GSM-Plus** useful, please consider giving star and citing our paper:
```
@misc{li2024gsmplus,
      title={GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers}, 
      author={Qintong Li and Leyang Cui and Xueliang Zhao and Lingpeng Kong and Wei Bi},
      year={2024},
      eprint={2402.19255},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

