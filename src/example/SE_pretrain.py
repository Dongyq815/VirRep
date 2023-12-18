import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import BertPretrainData
from optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from bertconfig import BertConfig
from semantic_encoder import BertForMaskedLM


def get_opts(args):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-data-file',
                        required=True,
                        type=str)
    
    parser.add_argument('--val-data-file',
                        required=False,
                        type=str)
    
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        type=str)
    
    parser.add_argument('-k', '--kmer-size',
                        required=True,
                        type=int)
    
    parser.add_argument('-l', '--max-seq-len',
                        required=True,
                        type=int)
    
    parser.add_argument('-p', '--mask-prob',
                        required=False,
                        type=float,
                        default=0.15)
    
    parser.add_argument('--num-encoders',
                        required=True,
                        type=int)
    
    parser.add_argument('--dim-model',
                        required=True,
                        type=int)
    
    parser.add_argument('--num-attention-heads',
                        required=True,
                        type=int)
    
    parser.add_argument('--forward-size',
                        required=True,
                        type=int)
    
    parser.add_argument('-e', '--epochs',
                        required=True,
                        type=int)
    
    parser.add_argument('-b', '--per-iter-batch-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--gradient_accumulation_iters',
                        required=True,
                        type=int)
    
    parser.add_argument('--learning-rate',
                        required=True,
                        type=float)
    
    parser.add_argument('--warmup-steps',
                        required=True,
                        type=int)
    
    parser.add_argument('--weight-decay',
                        required=False,
                        type=float,
                        default=0.01)
    
    parser.add_argument('--max-gradient-norm',
                        required=False,
                        type=float,
                        default=1)
    
    parser.add_argument('--beta1',
                        required=False,
                        type=float,
                        default=0.9)
    
    parser.add_argument('--beta2',
                        required=False,
                        type=float,
                        default=0.999)
    
    parser.add_argument('--adamw-eps',
                        required=False,
                        type=float,
                        default=1e-8)
    
    parser.add_argument('--save-steps',
                        required=True,
                        type=int)
    
    parser.add_argument('-d', '--gpu-device',
                        required=False,
                        type=int,
                        default=0)
    
    parser.add_argument('-w', '--num-workers',
                        required=False,
                        type=int,
                        default=0)
    
    if not args:
        parser.print_help(sys.stderr)
        sys.exit()
    
    return parser.parse_args(args)
    

def pretrain(opts, model, dataloader_tr, dataloader_val):
    
    device = 'cuda:'+str(opts.gpu_device)
    model = model.to(device)
    
    max_steps = opts.epochs * len(dataloader_tr) // opts.gradient_accumulation_iters
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 
                       not any(nd in n for nd in no_decay)],
            "weight_decay": opts.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if 
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = optim.AdamW(grouped_parameters, lr=opts.learning_rate,
                            betas=(opts.beta1, opts.beta2), eps=opts.adamw_eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=opts.warmup_steps,
        num_training_steps=max_steps
    )
    
    global_steps, global_batch = 0
    vocab_size = 4**opts.kmer_size+3
    loss_fn = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    
    for current in range(opts.epochs):
        
        loss_tr = 0
        loop = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        
        for batch, (inputs, labels) in loop:
            global_batch += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()

            logits = model(inputs)[0]
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            loss_tr += loss.item()
            loss = loss / opts.gradient_accumulation_iters
            loss.backward()
            
            if global_batch % opts.gradient_accumulation_iters == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opts.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                global_steps += 1
            
                loss_avg = loss_tr / (batch+1)
                perplexity_tr = math.exp(loss_avg)
            
                loop.set_description(f'[{current+1}/{opts.epochs}]')
                loop.set_postfix(loss=loss_avg, ppl=perplexity_tr)
            
            if global_batch % (opts.save_steps*opts.gradient_accumulation_iters) == 0:
                if dataloader_val is not None:
                    model.eval()
                    loss_val = 0
                    
                    for inputs, labels in dataloader_val:
                        inputs, labels = inputs.to(device), labels.to(device)
                        with torch.no_grad():
                            logits = model(inputs)[0]
                            loss_val += loss_fn(logits.view(-1, vocab_size),
                                                labels.view(-1)).item()
                            
                    loss_val = loss_val / len(dataloader_val)
                    perplexity_val = math.exp(loss_val)
                    print('val_loss: %.4f  val_perplexity: %.4f'%(loss_val, perplexity_val))
                    
                model_file = 'epoch%s-step%s.pth'%(current, global_steps)
                model_path = os.path.join(opts.output_dir, model_file)
                torch.save(model.state_dict(), model_path)

        
if __name__ == '__main__':
    
    args = sys.argv[1:]
    opts = get_opts(args)
    
    data_tr = BertPretrainData(opts.train_data_file, opts.kmer_size,
                               opts.max_seq_len, opts.mask_prob)
    dataloader_tr = DataLoader(data_tr, batch_size=opts.per_iter_batch_size,
                               shuffle=True, num_workers=opts.num_workers,
                               pin_memory=True)
    
    if opts.val_data_file is not None:
        data_val = BertPretrainData(opts.val_data_file, opts.kmer_size,
                                    opts.max_seq_len, opts.mask_prob)
        dataloader_val = DataLoader(data_val, batch_size=opts.per_iter_batch_size,
                                    shuffle=True, num_workers=opts.num_workers,
                                    pin_memory=True)
    else:
        dataloader_val = None
    
    vocab_size = 4**opts.kmer_size+3
    config = BertConfig(vocab_size=vocab_size,
                        hidden_size=opts.dim_model,
                        num_hidden_layers=opts.num_encoders,
                        num_attention_heads=opts.num_attention_heads,
                        intermediate_size=opts.forward_size,
                        max_position_embeddings=opts.max_seq_len)
    model = BertForMaskedLM(config)
    pretrain(opts, model, dataloader_tr, dataloader_val)