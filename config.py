from pathlib import Path

def get_config():
    return {
        "batch_size": 16,  # Reduced from 40 to fit in GPU memory
        "num_epochs": 2,
        "learning_rate": 10**-4,
        "seq_len": 310,
        "d_model": 512,
        "h": 8,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt": "it",
        # "model_folder": "weights", # use this in IDE
        "model_folder": "/content/drive/MyDrive/transformer_weights", # use this in colab
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

