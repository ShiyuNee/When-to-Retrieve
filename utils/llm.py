import openai
import time
from .utils import deal_answer, deal_judge_new, has_answer

Your_Key='openai-key'


model2api = {
    'gpt-instruct': 'gpt-3.5-turbo-instruct',
    'chatgpt': 'gpt-3.5-turbo-0301',
    'gpt4': 'gpt-4-1106-preview'
}

def get_llm_result(prompt, chat, sample, deal_type, response_sample, model='chatgpt'):
    prompt_challenge = "I do not think your answer is right. If you are still sure that your answer is accurate and correct, please say \"certain\". Otherwise, say \"uncertain\"."
    def get_multi_turn_message_chat():
        message = {"role": "user", "content": prompt}
        message_assis = {'role': 'assistant', 'content': response_sample['Res']}
        message_challenge = {'role': 'user', 'content': prompt_challenge}
        messages = [message, message_assis, message_challenge]
        return messages
    
    def get_multi_turn_message():
        message = f"User: {prompt}\n"
        message_assis = f"Assistant: {response_sample['Res']}\n\n"
        message_challenge = f"User: {prompt_challenge}\n"
        messages = message + message_assis + message_challenge + "Assistant:"
        return messages

    def get_res_batch(prompt_list):
        max_tokens=256
        res = openai.Completion.create(
            model=model2api[model],
            prompt=prompt_list,
            max_tokens=max_tokens,
            # logprobs=5
        )
        text_list = []
        for choice in res['choices']:
            text = choice['text'].strip()
            text_list.append(text)         
        return text_list

    def get_res_from_chat(messages):
        max_tokens = 256 
        res = openai.ChatCompletion.create(
            model=model2api[model],
            messages=messages,
            max_tokens=max_tokens,
        )
        steps_list = []
        for choice in res['choices']:
            steps = choice['message']['content'].strip()
            steps_list.append(steps)
        return steps_list

        # 处理访问频率过高的情况
    def get_res(prompt, chat=False):
        openai.api_key = Your_Key
        
        while True:
            try:
                if chat:
                    message = {"role": "user", "content": prompt}
                    messages = [message]
                    if response_sample != "":
                        messages = get_multi_turn_message_chat()
                    res = get_res_from_chat(messages)
                else:
                    print('get_res_batch')
                    if response_sample != "":
                        prompt = get_multi_turn_message()
                        print(prompt)
                    res = get_res_batch(prompt)
                break
            except openai.error.RateLimitError as e:
                print('\nRateLimitError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.ServiceUnavailableError as e:
                print('\nServiceUnavailableError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.Timeout as e:
                print('\nTimeout\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIError as e:
                print('\nAPIError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIConnectionError as e:
                print('\nAPIConnectionError\t', e, '\tRetrying...')
                time.sleep(5)
            except Exception as e:
                print(e)
                res = None
                break
        return res


    def request_process(prompt, chat, sample, deal_type):
        res = get_res(prompt, chat=chat) # res_list
        prediction = None
        prediction = res[0] if res is not None else None
        res_sample = {}
        if 'qa' in deal_type: # qa
            res_sample['prompt'] = prompt
            res_sample['question'] = sample['question']
            res_sample['Res'] = prediction
            res_sample['has_answer'] = has_answer(sample['reference'], prediction)
        else: # prior or post
            res_sample['prompt'] = prompt
            res_sample['Res'] = prediction
            res_sample['Giveup'] = deal_judge_new(prediction)
        return res_sample
    
    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    return request_process(prompt, chat, sample, deal_type)
