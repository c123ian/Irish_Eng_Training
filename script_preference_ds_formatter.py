import json
import sys

def process_jsonl(input_file, output_file):
    translations = {}
    system_prompt = "You are an AI assistant. You will be given a sentence to translate:"

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            key = (data['src'], data['direction'].split('_')[0])
            
            if key not in translations:
                translations[key] = []
            translations[key].append(data)

        for key, items in translations.items():
            items.sort(key=lambda x: x['cometkiwi_score'], reverse=True)
            
            question = key[0]
            chosen = items[0]['mt']
            rejected = items[1]['mt'] if len(items) > 1 else items[0]['mt']
            
            if len(items) > 1 and items[0]['cometkiwi_score'] == items[1]['cometkiwi_score']:
                if '_gpt' in items[0]['direction']:
                    chosen, rejected = rejected, chosen

            output = {
                "system": system_prompt,
                "question": question,
                "chosen": chosen,
                "rejected": rejected
            }
            json.dump(output, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_jsonl(input_file, output_file)