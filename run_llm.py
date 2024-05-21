import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.llm import get_llm_result
from utils.prompt import get_prompt


ra_dict = {
    'none': 'none',
    'sparse': {'sparse_ctxs': 1},
    'dense': {'dense_ctxs': 1},
    'gold': {'gold_ctxs': 1},
    'dpr': {'dpr_ctx': 1},
    'dpr_wrong': {'dpr_ctx_wrong': 1}
}


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/source/nq.json')
    parser.add_argument('--response', type=str, default='')
    parser.add_argument('--usechat', action='store_true')
    parser.add_argument('--type', type=str, choices=['qa', 'qa_explain', 'qa_cot', 'qa_gene', 
                                                     'prior', 'prior_punish', 'prior_explain', 'prior_pun_exp', 'prior_cot', 'prior_gene',
                                                     'post', 'post_punish'], default='qa')
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--model', type=str, default="chatgpt", choices=['chatgpt', 'gpt-instruct', 'gpt'])
    parser.add_argument('--outfile', type=str, default='data/qa/chatgpt-nq-none.json')   
    parser.add_argument('--idx', type=str, default="")   
    args = parser.parse_args()
    args.ra = ra_dict[args.ra]
    args.usechat == True if args.model == 'chatgpt' or args.model == 'gpt4' else False

    return args


def main():

    args = get_args()
    print(f'Model: {args.model}')
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    all_data = load_source(args.source)

    # prepare for multi-ture chat(not necessary most of the time)
    response_data = []
    if os.path.exists(args.response):
        response_data = load_source(args.response)
    idx_list = load_source(args.idx) if args.idx != "" else range(len(all_data))
    num_output = 0
    idx = 0
    try:
        for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
            response_sample = response_data[idx] if len(response_data) != 0 else ""
            res = {'info': 'no need to get results'} # initialize the results

            if 'info' not in sample and idx in idx_list: # need to get results for this sample
                if 'dpr' in args.ra:
                    if 'dpr_ctx_wrong' in sample: # only part of the data has the key "dpr_ctx_wrong"
                        prompt = get_prompt(sample, args)
                        res = get_llm_result(prompt, args.usechat, sample, args.type, response_sample, args.model)
                else:
                    prompt = get_prompt(sample, args)
                    res = get_llm_result(prompt, args.usechat, sample, args.type, response_sample, args.model)

            idx += 1
            outfile.write(json.dumps(res) + "\n")
            num_output += 1
    except Exception as e:
        logging.exception(e)
        
    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()


if __name__ == '__main__':
    main()
