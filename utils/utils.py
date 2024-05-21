import argparse
import collections
import json
import copy
import os
import re
import logging
import string
import regex
import unicodedata
from tqdm import tqdm
from nltk.corpus import stopwords


logger = logging.getLogger()

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def remove_punc(text):
    # punc替换成 " ", 匹配时空格比空字符好 
    exclude = set(string.punctuation)
    return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

def is_digital(text):
    return text.isdigit()

def remove_stopwords(text):
    words = stopwords.words('english')
    text = [w for w in text if w not in words]
    return text 

def get_clean(data, clean_data):
    assert len(data) == len(clean_data)
    for idx in range(len(data)):
        data[idx]['clean_pred'] = data[idx]['pred']
    return data

def get_data_before_and_after_prompt(origin_data, prompt_data, criterion):
    new_res = []
    for sample in origin_data:
        # if 'idx' not in prompt_data[sample['nq_idx']]:
        #     continue
        if criterion == 'same':
            if sample['Giveup_origin'] == prompt_data[sample['nq_idx']]['Giveup']:
                new_res.append(sample)
        else:
            if sample['Giveup_origin'] != prompt_data[sample['nq_idx']]['Giveup']:
                new_res.append(sample)
    return new_res

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join([ch if ch in text and ch not in exclude else ' ' for ch in text])

    def lower(text):
        return text.lower()
    # print(white_space_fix(remove_articles(remove_punc(lower(s)))))
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def has_answer(answers, text, match_type="string"):
    """
    text中是否包含answers列表中的任意一个answer
    - answers: a list of candidate answers
    - text: str
    """
    class Tokens(object):
        """A class to represent a list of tokenized text."""
        TEXT = 0
        TEXT_WS = 1
        SPAN = 2
        POS = 3
        LEMMA = 4
        NER = 5

        def __init__(self, data, annotators, opts=None):
            self.data = data
            self.annotators = annotators
            self.opts = opts or {}

        def __len__(self):
            """The number of tokens."""
            return len(self.data)

        def slice(self, i=None, j=None):
            """Return a view of the list of tokens from [i, j)."""
            new_tokens = copy.copy(self)
            new_tokens.data = self.data[i: j]
            return new_tokens

        def untokenize(self):
            """Returns the original text (with whitespace reinserted)."""
            return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

        def words(self, uncased=False):
            """Returns a list of the text of each token
            Args:
                uncased: lower cases text
            """
            if uncased:
                return [t[self.TEXT].lower() for t in self.data]
            else:
                return [t[self.TEXT] for t in self.data]

        def offsets(self):
            """Returns a list of [start, end) character offsets of each token."""
            return [t[self.SPAN] for t in self.data]

        def pos(self):
            """Returns a list of part-of-speech tags of each token.
            Returns None if this annotation was not included.
            """
            if 'pos' not in self.annotators:
                return None
            return [t[self.POS] for t in self.data]

        def lemmas(self):
            """Returns a list of the lemmatized text of each token.
            Returns None if this annotation was not included.
            """
            if 'lemma' not in self.annotators:
                return None
            return [t[self.LEMMA] for t in self.data]

        def entities(self):
            """Returns a list of named-entity-recognition tags of each token.
            Returns None if this annotation was not included.
            """
            if 'ner' not in self.annotators:
                return None
            return [t[self.NER] for t in self.data]

        def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
            """Returns a list of all ngrams from length 1 to n.
            Args:
                n: upper limit of ngram length
                uncased: lower cases text
                filter_fn: user function that takes in an ngram list and returns
                True or False to keep or not keep the ngram
                as_string: return the ngram as a string vs list
            """

            def _skip(gram):
                if not filter_fn:
                    return False
                return filter_fn(gram)

            words = self.words(uncased)
            ngrams = [(s, e + 1)
                    for s in range(len(words))
                    for e in range(s, min(s + n, len(words)))
                    if not _skip(words[s:e + 1])]

            # Concatenate into strings
            if as_strings:
                ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

            return ngrams

        def entity_groups(self):
            """Group consecutive entity tokens with the same NER tag."""
            entities = self.entities()
            if not entities:
                return None
            non_ent = self.opts.get('non_ent', 'O')
            groups = []
            idx = 0
            while idx < len(entities):
                ner_tag = entities[idx]
                # Check for entity tag
                if ner_tag != non_ent:
                    # Chomp the sequence
                    start = idx
                    while (idx < len(entities) and entities[idx] == ner_tag):
                        idx += 1
                    groups.append((self.slice(start, idx).untokenize(), ner_tag))
                else:
                    idx += 1
            return groups


    class Tokenizer(object):
        """Base tokenizer class.
        Tokenizers implement tokenize, which should return a Tokens class.
        """

        def tokenize(self, text):
            raise NotImplementedError

        def shutdown(self):
            pass

        def __del__(self):
            self.shutdown()


    class SimpleTokenizer(Tokenizer):
        ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
        NON_WS = r'[^\p{Z}\p{C}]'

        def __init__(self, **kwargs):
            """
            Args:
                annotators: None or empty set (only tokenizes).
            """
            self._regexp = regex.compile(
                '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
                flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
            )
            if len(kwargs.get('annotators', {})) > 0:
                logger.warning('%s only tokenizes! Skipping annotators: %s' %
                            (type(self).__name__, kwargs.get('annotators')))
            self.annotators = set()
        

        def tokenize(self, text):
            data = []
            matches = [m for m in self._regexp.finditer(text)]
            # print(f'matches: {matches}')
            for i in range(len(matches)):
                # Get text
                token = matches[i].group()

                # Get whitespace
                span = matches[i].span()
                start_ws = span[0]
                if i + 1 < len(matches):
                    end_ws = matches[i + 1].span()[0]
                else:
                    end_ws = span[1]

                # Format data
                data.append((
                    token,
                    text[start_ws: end_ws],
                    span,
                ))
            return Tokens(data, self.annotators)

    tokenizer = SimpleTokenizer()
    text = _normalize_answer(unicodedata.normalize('NFD', text)) # pred_text
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers: # candidate answers
            single_answer = _normalize_answer(unicodedata.normalize('NFD', single_answer))
            single_answer = tokenizer.tokenize(single_answer).words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i+ len(single_answer)]:
                    return 1
    return 0

def EM_compute(answer_list, prediction):
    return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])

def F1_compute(answers, pred):
    def get_tokens(s):
        if not s: return []
        return _normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        # print(f'ans: {gold_toks}')
        # print(f'pred: {pred_toks}')
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    return max([compute_f1(x, pred) for x in answers])


def deal_judge(pred):
    if pred is None:
        return True
    if has_answer(["unknown", "no specific answer", "not provide", "cannot answer", "no information provided", "no answer", "not contain", "no definitive answer"], pred):
        return True
    return False

def deal_judge_new(pred):
    if pred is None:
        return True
    if has_answer(["sorry", "apologize", "apologies", "uncertain", "false", "no", 'unsure', "cannot", "unknown", "no specific answer", "not provide", "cannot answer", "no information provided", "no answer", "not contain", "no definitive answer"], pred):
        return True
    return False

def deal_no_info(pred):
    if pred is None:
        return True
    if has_answer(["cannot", "unknown", "provide", 'information', 'assistant', 'artificial', 'unsure', 'robot'], pred):
        return True
    return False

def deal_answer(pred, answers):
    if pred is None:
        return 0, 0
    if pred.lower().startswith("answer:"):
        pred = pred[7:]
    return EM_compute(answers, pred), F1_compute(answers, pred)

def str2paras(s):
        if s is None:
            return None
        paras = []
        for text in s.split('\n'):
            if text.strip() != '':
                paras.append(": " + text)
        return paras

def load_source(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data
