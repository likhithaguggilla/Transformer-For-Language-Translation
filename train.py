import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# import torch_directml

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path 
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path
import warnings

# def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split = 'train')

    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #s keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, config['lang_src'], config['lang_tgt'], tokenizer_src, tokenizer_tgt, config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, config['lang_src'], config['lang_tgt'], tokenizer_src, tokenizer_tgt, config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source: {max_len_src}")
    print(f"Max length of target: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, config['seq_len'], vocab_tgt_len, config['seq_len'], config['d_model'])
    return model

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1,1).fill_(sos_idx).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:,-1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, loss_fn, num_examples=2):
    model.eval()
    count = 0
    console_width = 80
    
    # Track validation loss
    total_loss = 0
    num_batches = 0
    
    # Store examples for TensorBoard
    source_texts = []
    expected_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in validation_ds:
            num_batches += 1

            encoder_input = batch['enc_input'].to(device)
            decoder_input = batch['dec_input'].to(device)
            encoder_mask = batch['enc_mask'].to(device)
            decoder_mask = batch['dec_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass to calculate loss
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)

            # Calculate loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            # Collect example predictions for display and TensorBoard
            if count < num_examples:
                count += 1
                
                # Get the source and target text
                # DataLoader returns batched data, so we need to index into the batch
                source_text = batch['src_text'][0] if hasattr(batch['src_text'], '__getitem__') else str(batch['src_text'])
                target_text = batch['tgt_text'][0] if hasattr(batch['tgt_text'], '__getitem__') else str(batch['tgt_text'])
                
                # Use greedy decode for prediction
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
                predicted_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
                
                # Store for TensorBoard
                source_texts.append(source_text)
                expected_texts.append(target_text)
                predicted_texts.append(predicted_text)
                
                # Print to console
                print_msg('-' * console_width)
                print_msg(f'VALIDATION EXAMPLE {count}:')
                print_msg(f'  SOURCE:    {source_text}')
                print_msg(f'  EXPECTED:  {target_text}')
                print_msg(f'  PREDICTED: {predicted_text}')
                print_msg('-' * console_width)

    # Calculate average validation loss
    avg_val_loss = total_loss / num_batches
    
    # Log to TensorBoard
    if writer:
        # Log validation loss
        writer.add_scalar('validation_loss', avg_val_loss, global_step)
        
        # Log text examples as a formatted table
        text_table = "| Source | Expected | Predicted |\n|--------|----------|----------|\n"
        for src, exp, pred in zip(source_texts, expected_texts, predicted_texts):
            # Truncate long texts for readability
            src_short = (src[:50] + '...') if len(src) > 50 else src
            exp_short = (exp[:50] + '...') if len(exp) > 50 else exp
            pred_short = (pred[:50] + '...') if len(pred) > 50 else pred
            text_table += f"| {src_short} | {exp_short} | {pred_short} |\n"
        
        writer.add_text('validation_examples', text_table, global_step)
        writer.flush()
    
    print_msg(f'\nValidation Loss: {avg_val_loss:.4f}\n')
    
    return avg_val_loss



def train_model(config):

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # device = torch_directml.device()
    # print(f"Using device: {device}")

    # Create the directories for saving weights and logs
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Load the dataset and the tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Initialize the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Track experiments with tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # Initialize epoch and global step
    initial_epoch = 0
    global_step = 0

    # Load the weights if available
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Training in epochs and batches
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            model.train()

            # Add batch (inputs) to device
            encoder_input = batch['enc_input'].to(device)   # (B, seq_len)
            decoder_input = batch['dec_input'].to(device)   # (B, seq_len)
            encoder_mask = batch['enc_mask'].to(device)     # (B, 1, 1, seq_len)
            decoder_mask = batch['dec_mask'].to(device)     # (B, 1, seq_len, seq_len)

            # Forward Pass
            encoder_output = model.encode(encoder_input, encoder_mask)   # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)   # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)   # (B, seq_len, vocab_size)

            # Add batch (labels) to device
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step) # log the loss
            writer.flush()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
    
        # Run validation at the end of each epoch
        print(f'\n{"="*80}')
        print(f'Running Validation for Epoch {epoch:02d}...')
        print(f'{"="*80}')
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, loss_fn)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
            

if __name__ == '__main__':
    print("Starting training...")
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

