# When to Retrieve

- This is the official repo of the paper [ACL'24: When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://arxiv.org/pdf/2402.11457.pdf)

# Basic Usage

There are four steps to get the desired responses.

## Inference

- **Step1:** Run `run_llm.py` to get the basic results

  ```sh
  python run_llm.py --source data/nq_sample.jsonl --ra none --type prior --outfile ./examples/test.jsonl --model chatgpt
  ```

  - You can specify `--type `[qa/qa_explain/qa_cot/qa_gene/prior/prior_punish/prior_explain/prior_pun_exp/prior_cot/prior_gene]

- Note
  - The desired format of the response is [Answer, Confidence]. For example, when the question is `What is the capital of Chain`, we expect the response to be `"Beijing, Certain"`
  - LLMs might produce responses that don't adhere to the expected format. It could simply provide the answer `Beijing` or the confidence `Uncertain`.
  - We perform post-processing to obtain the desired output format.

### Post-process & Evaluate

- **Step2**: Get the indices of samples that do not match the expected output format.

  ```sh
  python collect.py --mode preprocess --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/confidence.jsonl --answer ./examples/answer.jsonl --model chatgpt
  ```

- **Step3**: Generate the missing results for these samples.

  ```sh
  python run_llm.py --source data/nq_sample.jsonl --ra none --type qa --outfile ./examples/post_answer.jsonl --idx ./examples/answer.jsonl --model chatgpt
  
  python run_llm.py --source ./examples/test.jsonl --ra none --type post --outfile ./examples/post_confidence.jsonl --idx ./examples/confidence.jsonl --model chatgpt
  ```

  - If a strategy including punish (i.e., `prior_punish/prior_pun_exp`) is used **during inference**, `--type` should be set to `post_punish`.

- **Step4**: Merge results and evaluate

  ```sh
  python collect.py --mode evaluate --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/post_confidence.jsonl --answer ./examples/post_answer.jsonl
  ```

## RAG

### Static RAG

```sh
python run_llm.py --source data/nq_sample.jsonl --ra [sparse/dense/gold] --type qa --outfile ./examples/test_gold_static.jsonl --model chatgpt
```

- For NQ, you can also specify `--ra dpr` as the gold documents. (We do this in our paper)

### Adaptive RAG

**Evaluation**

- `Static RAG`

  ```bash
  python collect.py \
  	--mode eval_rag \
  	--input ./examples/test_gold_static.jsonl
  ```

- `Adaptive RAG`

  ```python
  python collect.py \
  	--mode eval_adaptive_rag \
  	--input ./examples/test_gold_static.jsonl \ 
  	--origin ./examples/test_new.jsonl 
  ```

## Note

You can find necessary commands in `scripts/` and the demo data in `examples/`

> The repository is continuously being updated.
>
> Feel free to propose any issue.

## Citation

If you find our paper/repo useful, please cite:

```latex
@article{ni2024llms,
  title={When Do LLMs Need Retrieval Augmentation? Mitigating LLMs' Overconfidence Helps Retrieval Augmentation},
  author={Ni, Shiyu and Bi, Keping and Guo, Jiafeng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2402.11457},
  year={2024}
}
```

