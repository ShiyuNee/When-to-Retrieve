# standard
python run_llm.py --source data/nq_sample.jsonl --ra none --type qa --outfile ./test.jsonl --model chatgpt

python run_llm.py --source data/nq_sample.jsonl --ra none --type qa --outfile ./post_answer.jsonl --idx ./answer.jsonl --model chatgpt

python run_llm.py --source ./test_new.jsonl --ra none --type post --outfile ./post_confidence.jsonl --idx ./confidence.jsonl --model chatgpt

# challenge
python -u run_llm.py --response ./examples/test.jsonl --source data/nq_sample.jsonl --ra none --type prior --outfile ./examples/nq_challenge.jsonl --model chatgpt