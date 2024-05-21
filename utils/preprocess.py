from utils.utils import has_answer, read_json, write_jsonl
import string

pattern = ['uncertainty', 'certainty', 'uncertain', 'certainly', 'certain', 'unsure']

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

def remove_pattern(text, patterns):
    text = text.lower()
    for item in patterns:
        text = text.replace(item, '')
    return text

def get_post_idx(path, out_path, qa_path, confidence_idx_path, answer_idx_path):
    """
    Gold: get samples which do not contain confidence-words or answers.
    """
    answer_idx, confidence_idx = [], []
    data = read_json(path) # answer + confidence
    qa_data = read_json(qa_path)
    for idx in range(len(data)):
        if 'Res' not in data[idx] or data[idx]['Res'] == None: # filter
            continue
        # merge necessary information into one file
        data[idx]['question'] = qa_data[idx]['question']
        if 'dpr_ctx_wrong' in qa_data[idx]: 
            data[idx]['dpr_ctx'] = qa_data[idx]['dpr_ctx']
            data[idx]['dpr_ctx_wrong'] = qa_data[idx]['dpr_ctx_wrong']

        # check confidence-words: certain, uncertain, et.al.
        new_res = remove_pattern(data[idx]['Res'], pattern).strip()
        if new_res == data[idx]['Res'].lower():
            confidence_idx.append(idx)
        # check answer
        if len(new_res) <= 1:
            answer_idx.append(idx)
    print(f'Responses without confidence count: {len(confidence_idx)}')
    print(f'Responses without answer count: {len(answer_idx)}')

    write_jsonl(data, out_path)
    write_jsonl(confidence_idx, confidence_idx_path)
    write_jsonl(answer_idx, answer_idx_path)

    
def merge_post_data(path, out_path, qa_path, post_confidence_path, post_answer_path):
    """
    """
    data = read_json(path) # answer + confidence
    qa_data = read_json(qa_path)
    post_answer = read_json(post_answer_path)
    post_confidence = read_json(post_confidence_path)
    # 替换答案中在pattern中存在的字符串
    for idx in range(len(data)):
        # filter 
        if 'Res' not in data[idx] or data[idx]['Res'] == None:
            continue
        new_res = remove_pattern(data[idx]['Res'], pattern).strip()
        save_res = new_res
        # replace confidence with post-confidence
        if new_res == data[idx]['Res'].lower():
            data[idx]['Giveup'] = post_confidence[idx]['Giveup']
        # replace the answer with post-answer
        if len(new_res) <= 1: 
            new_res = post_answer[idx]['Res']
        has_temp = has_answer(qa_data[idx]['reference'], new_res)
        data[idx]['has_answer'] = has_temp
        data[idx]['Res'] = save_res 
    write_jsonl(data, out_path)

if __name__ == '__main__':
    path = './data/source/test_res/text-davinci-003/nq_prompt7_giveup_res_ra.jsonl'
    out_path = './data/source/test_res/text-davinci-003/nq_prompt7_giveup_res_ra_new.jsonl'