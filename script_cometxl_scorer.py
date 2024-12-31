import modal
import os
import json

app = modal.App("comet-predict")

# model stored in volume after running download_cometxxl.py script
MODELS_DIR = "/comet_model"
MODEL_NAME = "Unbabel/wmt23-cometkiwi-da-xxl"

# Time constants
MINUTES = 60
HOURS = 60 * MINUTES

# Create or access the volume (set to True)
volume = modal.Volume.from_name("comet_model", create_if_missing=False)

# Define the Modal image with necessary packages
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # Download models from the Hugging Face Hub
            "hf-transfer",      # Download models faster with Rust
            "unbabel-comet",     # For tokenizer and model handling
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Function to download the model thats stored in the modal.volume (download_cometxxl.py)
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Reload the volume to get the latest state
    volume.reload()

@app.cls(
    image=image,
    timeout=900,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    allow_concurrent_inputs=100,
    secrets=[modal.Secret.from_name("huggingface-token-2")],
    volumes={MODELS_DIR: volume}
)
class CometPredictor:
    @modal.enter()
    def load_model(self):
        from comet import load_from_checkpoint
        import torch
        
        # Set higher precision for better performance
        torch.set_float32_matmul_precision('high')
        
        # Load model once when the class is instantiated
        checkpoint_path = self.find_checkpoint_file(MODELS_DIR)
        if not checkpoint_path:
            raise Exception(f"Could not find checkpoint file in {MODELS_DIR}/checkpoints")
        
        self.model = load_from_checkpoint(checkpoint_path)
    
    @staticmethod
    def find_checkpoint_file(base_dir):
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoint_files:
                return os.path.join(checkpoint_dir, checkpoint_files[0])
        return None
    
    @modal.method()
    def predict(self, data, batch_size=32):
        model_output = self.model.predict(data, batch_size=batch_size)
        return model_output.scores, model_output.system_score

def preprocess_data(input_data):
    processed_data = []
    for item in input_data:
        # Skip items missing the required keys
        if "en" not in item or "ga" not in item or "gpt_4_ga" not in item or "gpt_4_en" not in item:
            continue
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
    BATCH_SIZE = 32  # Process predictions in batches
    
    input_data = []
    try:
        # Open the input file with UTF-8 encoding
        with open('translated.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    input_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSONDecodeError: {e}")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError encountered: {e}")

    processed_data = preprocess_data(input_data)
    total_items = len(processed_data)
    print(f"Total items to process: {total_items}")
    
    # Create predictor instance without 'with' statement
    predictor = CometPredictor()
    
    # Process in batches, but write individually
    for i in range(0, len(processed_data), BATCH_SIZE):
        batch = processed_data[i:i + BATCH_SIZE]
        try:
            # Process entire batch at once
            segment_scores, system_score = predictor.predict.remote(batch, batch_size=BATCH_SIZE)
            
            # Write results individually, but from batched predictions
            with open('translated_gaois_graded.jsonl', 'a', encoding='utf-8') as f:
                for j, item in enumerate(batch):
                    result = {
                        "src": item["src"],
                        "mt": item["mt"],
                        "direction": item["direction"],
                        "cometkiwi_score": segment_scores[j],
                        "system_score": system_score
                    }
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # Keep the flush for safety
                    
                    print(f"Processed item {i+j+1}/{total_items}")
            
        except Exception as e:
            print(f"Error processing batch starting at item {i}: {str(e)}")
            continue  # Skip to next batch if there's an error

    # Commit the changes to the volume
    volume.commit()

if __name__ == "__main__":
    modal.run()  # Automatically use the entrypoint from @app.local_entrypoint
