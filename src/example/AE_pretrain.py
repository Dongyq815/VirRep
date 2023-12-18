import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from alignment_encoder import SkipGram
from dataset import KmerEmbeddingData, batch_stack
from optimization import get_linear_schedule_with_warmup


def get_opts(args):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-data-file',
                        required=True,
                        type=str)
    
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        type=str)
    
    parser.add_argument('-k', '--kmer-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--embedding-dim',
                        required=True,
                        type=int)
    
    parser.add_argument('--window-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--negative-sample',
                        required=True,
                        type=int)
    
    parser.add_argument('-b', '--batch-size',
                        required=True,
                        type=int)
    
    parser.add_argument('-e', '--epochs',
                        required=True,
                        type=int)
    
    parser.add_argument('--learning-rate',
                        required=True,
                        type=float)
    
    parser.add_argument('--weight-decay',
                        required=False,
                        type=float,
                        default=0.0)
    
    parser.add_argument('--warmup-proportion',
                        required=True,
                        type=float)
    
    parser.add_argument('--max-gradient-value',
                        required=False,
                        type=float,
                        default=1)
    
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


def train(opts, model, dataloader):
    
    device = 'cuda:' + str(opts.gpu_device)
    model = model.to(device)
    model.train()
    
    max_steps = opts.epochs * len(dataloader)
    warmup_steps = int(opts.warmup_proportion * max_steps)
    optimizer = optim.SparseAdam(model.parameters, lr=opts.learning_rate,
                                 weight_decay=opts.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    global_steps = 0
    loss_fn = nn.BCEWithLogitsLoss()
    model.zero_grad(set_to_none=True)
    
    for e in range(opts.epochs):
        loss_sum = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for batch, (center_word, target_word, label) in loop:
            center_word = center_word.to(device)
            target_word = target_word.to(device)
            label = label.to(device)
            
            logits = model(center_word, target_word)
            loss = loss_fn(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(),
                                      opts.max_gradient_value)
            optimizer.step()
            scheduler.step()
            
            global_steps += 1
            loss_sum += loss.item()
            avg_loss = loss_sum / global_steps
            
            loop.set_description(f'[{e+1}/{opts.epochs}]')
            loop.set_postfix(loss=avg_loss)
            
            if global_steps % opts.save_steps == 0:
                model_file = 'epoch%s-step%s'%(e+1, global_steps)
                model_path = os.path.join(opts.output_dir, model_file)
                torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    
    args = sys.argv[1:]
    opts = get_opts(args)
    
    corpus = KmerEmbeddingData(opts.train_data_file, opts.kmer_size,
                               opts.window_size, opts.negative_sample)
    dataloader = DataLoader(corpus, batch_size=opts.batch_size,
                            shuffle=True, pin_memory=True,
                            num_workers=opts.num_workers,
                            collate_fn=batch_stack)
    
    model = SkipGram(opts.kmer_size, opts.embedding_dim)
    train(opts, model, dataloader)