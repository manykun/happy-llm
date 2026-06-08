import json
from tqdm import tqdm

pretrain_data = 'your local pretrain_data'
output_pretrain_data = 'seq_monkey_datawhale.jsonl'

sft_data = 'your local sft_data'
output_sft_data = 'BelleGroup_sft.jsonl'


def split_text(text, chunk_size=512):
    """将文本按指定字符长度切分成块"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def process_pretrain_data(input_path, output_path):
    with open(output_path, 'w', encoding='utf-8') as pretrain:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {input_path}"):
                line = json.loads(line)
                text = line.get('text', '').strip()

                if not text:
                    continue

                chunks = split_text(text)

                for chunk in chunks:
                    pretrain.write(
                        json.dumps({'text': chunk}, ensure_ascii=False) + '\n'
                    )


def convert_message(data):
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]

    for item in data:
        if item.get('from') == 'human':
            message.append({
                'role': 'user',
                'content': item.get('value', '')
            })
        elif item.get('from') == 'assistant':
            message.append({
                'role': 'assistant',
                'content': item.get('value', '')
            })

    return message


def process_sft_data(input_path, output_path):
    with open(output_path, 'w', encoding='utf-8') as sft:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {input_path}", unit="lines"):
                item = json.loads(line)

                conversations = item.get('conversations', [])
                message = convert_message(conversations)

                if len(message) <= 1:
                    continue

                sft.write(
                    json.dumps(message, ensure_ascii=False) + '\n'
                )


process_pretrain_data(pretrain_data, output_pretrain_data)
process_sft_data(sft_data, output_sft_data)
