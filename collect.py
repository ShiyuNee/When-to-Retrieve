from utils.compute import *
from utils.preprocess import *
import argparse

def get_score(path):
    data = read_json(path)
    compute_giveup_score(data) # 基础分

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/nq_sample.json')
    parser.add_argument('--input', type=str, default='./examples/test.jsonl')
    parser.add_argument('--output', type=str, default='./examples/test_new.jsonl')
    parser.add_argument('--confidence', type=str, default='./examples/confidence.jsonl')
    parser.add_argument('--answer', type=str, default='./examples/answer.jsonl')
    parser.add_argument('--mode', type=str, default='preprocess', choices=['preprocess', 'evaluate'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # step 1: python run_llm.py --source data/source/nq_sample.jsonl --ra none --type qa --outfile ./test.jsonl --usechat

    # step2: 
    if args.mode == 'preprocess':
        get_post_idx(args.input, args.output, args.source, args.confidence, args.answer)

    # step3: 
    #python run_llm.py --source data/source/nq_sample.jsonl --ra none --type qa --outfile ./post_answer.jsonl --idx ./answer.jsonl --usechat

    #python run_llm.py --source ./test_new.jsonl --ra none --type post --outfile ./post_confidence.jsonl --idx ./confidence.jsonl --usechat

    # step 4:
    elif args.mode == 'evaluate':
        merge_post_data(args.input, args.output, args.source, args.confidence, args.answer)
        get_score(args.output)
    else:
        print('The mode is wrong')


    
