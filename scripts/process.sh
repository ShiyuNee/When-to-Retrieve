python collect.py --mode preprocess --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/confidence.jsonl --answer ./examples/answer.jsonl

python collect.py --mode evaluate --source ./data/nq_sample.jsonl --input ./examples/test.jsonl --output ./examples/test_new.jsonl --confidence ./examples/post_confidence.jsonl --answer ./examples/post_answer.jsonl