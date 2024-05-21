# When to Retrieve

- This is the official repo of the paper [ACL'24: When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://arxiv.org/pdf/2402.11457.pdf)

# Basic Usage

## Inference

There are four steps to get the results

- Run `run_llm.py` to get the basic results
  - We hope the responses consist of [Answer, Confidence]. For example, when the question is `What is the capital of Chain`, we hope the answer can be `"Beijing, Certain"`
  - LLMs might generate responses that do not conform to the expected format. It may only contain the answer `Beijing` or the confidence `Uncertain`
  - We perform post-processing to ensure a standardized output format.

- Example

  ```sh
  python run_llm.py --source data/nq_sample.jsonl --ra none --type qa --outfile ./examples/test.jsonl --model chatgpt
  ```

## Post-process & Evaluate

- Get the indices of non-standard samples.

  ```sh
  python collect.py --mode preprocess --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/confidence.jsonl --answer ./examples/answer.jsonl --model chatgpt
  ```

- Generate the missing results for these samples.

  ```sh
  python run_llm.py --source data/nq_sample.jsonl --ra none --type qa --outfile ./examples/post_answer.jsonl --idx ./examples/answer.jsonl --model chatgpt
  
  python run_llm.py --source ./examples/test.jsonl --ra none --type post --outfile ./examples/post_confidence.jsonl --idx ./examples/confidence.jsonl --model chatgpt
  ```

- Merge results and evaluate

  ```sh
  python collect.py --mode evaluate --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/post_confidence.jsonl --answer ./examples/post_answer.jsonl
  ```

## Adaptive RAG

This will be updated soon

## Note

You can find necessary commands in `scripts/` and the demo data in `examples/`

> The repository is continuously being updated.
>
> Feel free to propose any issue.
