"""
Inference Script for Transformer Translation Model
Usage: python inference.py
"""

import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import build_transformer
from config import get_config, get_weights_file_path
from dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding to generate translation.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Encode the source
    encoder_output = model.encode(source, source_mask)

    # Initialize decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Create decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Decode
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project to vocabulary
        prob = model.project(out[:, -1])
        
        # Get the token with max probability
        _, next_word = torch.max(prob, dim=1)

        # Append to decoder input
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        # Stop if EOS token is generated
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)


def translate(sentence, model, tokenizer_src, tokenizer_tgt, config, device):
    """
    Translate a sentence from source language to target language.
    """
    model.eval()
    
    # Tokenize the source sentence
    enc_input_tokens = tokenizer_src.encode(sentence).ids
    
    # Add SOS and EOS tokens
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
    
    # Calculate padding
    num_padding = config['seq_len'] - len(enc_input_tokens) - 2
    
    if num_padding < 0:
        print(f"⚠️  Warning: Sentence too long ({len(enc_input_tokens)} tokens). Truncating to {config['seq_len'] - 2} tokens.")
        enc_input_tokens = enc_input_tokens[:config['seq_len'] - 2]
        num_padding = 0
    
    # Build encoder input
    encoder_input = torch.cat([
        sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        eos_token,
        torch.tensor([pad_token] * num_padding, dtype=torch.int64),
    ]).unsqueeze(0).to(device)  # Add batch dimension
    
    # Build encoder mask
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Perform greedy decoding
    with torch.no_grad():
        model_out = greedy_decode(
            model, encoder_input, encoder_mask,
            tokenizer_src, tokenizer_tgt,
            config['seq_len'], device
        )
    
    # Decode the output tokens to text
    translated_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    return translated_text


def load_model(config, device):
    """
    Load the trained model and tokenizers.
    """
    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists():
        raise FileNotFoundError(f"Source tokenizer not found at {tokenizer_src_path}")
    if not tokenizer_tgt_path.exists():
        raise FileNotFoundError(f"Target tokenizer not found at {tokenizer_tgt_path}")
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    print(f"✅ Loaded tokenizers:")
    print(f"   Source ({config['lang_src']}): {tokenizer_src.get_vocab_size()} tokens")
    print(f"   Target ({config['lang_tgt']}): {tokenizer_tgt.get_vocab_size()} tokens")
    
    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        config['seq_len'],
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['d_model']
    ).to(device)
    
    # Find the latest checkpoint
    model_folder = Path(config['model_folder'])
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found at {model_folder}")
    
    # Get all checkpoint files
    checkpoints = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {model_folder}")
    
    # Sort by epoch number and get the latest
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]
    
    print(f"✅ Loading model from: {latest_checkpoint}")
    state = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Trained for {state['epoch'] + 1} epochs")
    print(f"   Global step: {state['global_step']}")
    
    return model, tokenizer_src, tokenizer_tgt


def main():
    """
    Main inference function.
    """
    print("=" * 80)
    print("Transformer Translation - Inference")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎮 Using device: {device}")
    
    # Load model and tokenizers
    print("\n📦 Loading model and tokenizers...")
    model, tokenizer_src, tokenizer_tgt = load_model(config, device)
    
    print(f"\n🌍 Translation: {config['lang_src'].upper()} → {config['lang_tgt'].upper()}")
    print("=" * 80)
    
    # Interactive translation loop
    print("\nEnter sentences to translate (or 'quit' to exit):")
    print("-" * 80)
    
    while True:
        # Get input from user
        source_text = input(f"\n{config['lang_src'].upper()}: ").strip()
        
        # Check for exit command
        if source_text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty input
        if not source_text:
            continue
        
        # Translate
        try:
            translation = translate(source_text, model, tokenizer_src, tokenizer_tgt, config, device)
            print(f"{config['lang_tgt'].upper()}: {translation}")
        except Exception as e:
            print(f"❌ Error during translation: {e}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
