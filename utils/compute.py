def adaptive_retrieval(model_data, ra_data):
    """
    compute scores for results with adaptive RAG
    """
    score_list =[]
    for idx in range(len(model_data)):
        if 'info' in model_data[idx]:
            continue
        if model_data[idx]['Giveup'] == True:
            score_list.append(ra_data[idx]['has_answer']) # append ra results
        else:
            score_list.append(model_data[idx]['has_answer']) # append results without ra
    print(f'count: {len(score_list)}')
    print(f'has_answer: {sum(score_list) / len(score_list)}')


def compute_score(data):
    """
    compute scores for results with RAG
    """
    score_list = []
    em_list = []
    for idx in range(len(data)):
        sample = data[idx]
        if 'has_answer' not in sample:
            continue
        score_list.append(sample['has_answer'])
    print(f'count: {len(em_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')

def compute_giveup_score(data):
    """
    compute scores for results with any strategy
    """
    giveup_list, score_list, align = [], [], []
    overconf_count = 0
    conserv_count = 0
    for idx in range(len(data)):
        sample = data[idx]
        if 'has_answer' not in sample: # filter
            continue
        score_list.append(sample['has_answer'])
        if sample['has_answer'] != sample['Giveup']:
            align.append(1)

        if sample['Giveup'] == True:
            if sample['has_answer'] == 1:
                conserv_count +=1
        else:
            if sample['has_answer'] == 0:
                overconf_count += 1
        giveup_list.append(sample['Giveup'])
    print(f'conut: {len(giveup_list)}')
    print(f'uncertain ratio: {sum(giveup_list) / len(giveup_list)}')
    print(f'has answer: {sum(score_list) / len(score_list)}')
    print(f'overconf: {format(overconf_count / len(giveup_list), ".4f")}')
    print(f'conserv: {format(conserv_count / len(giveup_list), ".4f")}')
    print(f'alignment: {format(sum(align) / len(giveup_list), ".4f")}')





    