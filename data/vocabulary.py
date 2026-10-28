import json
import os

'''
用于生成词汇表。
'''

tokens = {}
count = 1
for i in range(0, 10):
    tokens.update({f"{i}": count})
    count += 1
tokens.update({"_":11, "+":12, "-":13, "^":14, "p":15})
count = 16
for i in range(1, 6):
    for j in range(i+1, 7):
        for k in [0, 1]:
            if k == 0:
                tokens.update({f"<{i}{j}>": count})
            else:
                tokens.update({f"[{i}{j}]": count})
            count += 1

_tokenizor_json_path = os.path.join(os.path.dirname(__file__), 'tokenizor.json')
with open(_tokenizor_json_path, 'w', encoding='utf-8') as f:
    json.dump(tokens, f, ensure_ascii=False, indent=4)

def reverse_tokenize(tokenizer: dict)->dict:
    '''
    仅仅只针对spinor bracket逆操作。
    '''
    start_idx = 16
    end_idx = 46

    reverse_tokens = {value: key for key, value in tokenizer.items() if start_idx <= value < end_idx}
    return reverse_tokens