import modal
import os
import json

app = modal.App("comet-predict")

image = (
    modal.Image.debian_slim()
    .pip_install("unbabel-comet", "huggingface_hub")
)

# Store the model in a global variable to avoid reloading it
model_cache = {}

@app.function(
    image=image,
    timeout=900,
    cpu=8.0,
    secrets=[modal.Secret.from_name("huggingface-token-2")]
)
def comet_predict(model_name, data, batch_size=8):
    from comet import download_model, load_from_checkpoint
    from huggingface_hub import login
    import torch
    
    # Load the model only if it is not already loaded
    if model_name not in model_cache:
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=huggingface_token, add_to_git_credential=False)
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        model_cache[model_name] = model
    else:
        model = model_cache[model_name]
    
    model_output = model.predict(data, batch_size=batch_size)
    segment_scores = model_output.scores
    system_score = model_output.system_score
    
    return segment_scores, system_score

def preprocess_data(input_data):
    processed_data = []
    for item in input_data:
        processed_data.append({
            "src": item["en"],
            "mt": item["gpt_4_ga"],
            "direction": "en-ga_gpt"
        })
        processed_data.append({
            "src": item["ga"],
            "mt": item["gpt_4_en"],
            "direction": "ga-en_gpt"
        })
        processed_data.append({
            "src": item["en"],
            "mt": item["ga"],
            "direction": "en-ga"
        })
        processed_data.append({
            "src": item["ga"],
            "mt": item["en"],
            "direction": "ga-en"
        })
    return processed_data

@app.local_entrypoint()
def main():
    input_data = []
    try:
        # Open the input file with UTF-8 encoding
        with open('Tatoeba_translated.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    input_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSONDecodeError: {e}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError encountered: {e}")

    processed_data = preprocess_data(input_data)
    
    # Open the output file in append mode
    with open('Tatoeba_translated_RESULT.jsonl', 'a', encoding='utf-8') as f:
        for i, item in enumerate(processed_data):
            try:
                segment_scores, system_score = comet_predict.remote("Unbabel/wmt23-cometkiwi-da-xl", [item])
                
                result = {
                    "src": item["src"],
                    "mt": item["mt"],
                    "direction": item["direction"],
                    "cometkiwi_score": segment_scores[0],
                    "system_score": system_score
                }
                
                # Write the result to the file immediately
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
                
                # Flush the file to ensure it's written to disk
                f.flush()
                
                print(f"Processed item {i+1}/{len(processed_data)}")
            except Exception as e:
                print(f"Error processing item {i+1}: {str(e)}")
                # Optionally, you can log the error or the problematic item for later investigation
                
if __name__ == "__main__":
    modal.run(main)

