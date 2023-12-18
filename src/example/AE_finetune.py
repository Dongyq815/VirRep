import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from alignment_encoder import AEForSimPred, AlnConfig
from dataset import SimPredData
from optimization import get_linear_schedule_with_warmup


def get_opts(args):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--virus-train-fasta',
                        required=True,
                        type=str)
    
    parser.add_argument('--virus-train-response',
                        required=True,
                        type=str)
    
    parser.add_argument('--host-train-fasta',
                        required=True,
                        type=str)
    
    parser.add_argument('--host-train-response',
                        required=True,
                        type=str)
    
    parser.add_argument('--virus-val-fasta',
                        required=False,
                        type=str)
    
    parser.add_argument('--virus-val-response',
                        required=True,
                        type=str)
    
    parser.add_argument('--host-val-fasta',
                        required=False,
                        type=str)
    
    parser.add_argument('--host-val-response',
                        required=True,
                        type=str)
    
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        type=str)
    
    parser.add_argument('-p', '--pretrained-embedding',
                        required=True,
                        type=str)
    
    parser.add_argument('--embedding-freeze',
                        required=False,
                        action='store_true',
                        default=False)
    
    parser.add_argument('-k', '--kmer-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--embedding-dim',
                        required=True,
                        type=int)
    
    parser.add_argument('--hidden-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--lstm-layers',
                        required=True,
                        type=int)
    
    parser.add_argument('--bidirectional',
                        required=False,
                        action='store_true',
                        default=False)
    
    parser.add_argument('-e', '--epochs',
                        required=True,
                        type=int)
    
    parser.add_argument('-b', '--batch-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--learning-rate',
                        required=True,
                        type=float)
    
    parser.add_argument('--embedding-lr',
                        required=True,
                        type=float)
    
    parser.add_argument('--warmup-proportion',
                        required=True,
                        type=float)
    
    parser.add_argument('--weight-decay',
                        required=False,
                        type=float,
                        default=0.01)
    
    parser.add_argument('--patience',
                        required=False,
                        type=float,
                        default=float('inf'))
    
    parser.add_argument('--delta',
                        required=False,
                        type=float,
                        default=0.0)
    
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


def train(opts, model, dataloader_tr, dataloader_val):
    
    device = 'cuda:'+str(opts.gpu_device)
    model = model.to(device)
    
    max_steps = opts.epochs*len(dataloader_tr)
    warmup_steps = int(opts.warmup_proportion * max_steps)
    no_decay = ["bias", "ln"]
    
    embedding_param = [{'params': [p for n, p in model.named_parameters() if
                                   'kmer_embedding' in n],
                        'weight_decay': opts.weight_decay, 'lr': opts.embedding_lr}]
    other_params = [
        {'params': [p for n, p in model.named_parameters() if 
                    any(nd in n for nd in no_decay) and 'kmer_embedding' not in n],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 
                    not any(nd in n for nd in no_decay) and 'kmer_embedding' not in n],
         'weight_decay':opts.weight_decay}]
    
    grouped_params = embedding_param+other_params
    optimizer = optim.AdamW(grouped_params, lr=opts.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    duration, loss_best = 0, 1
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    
    for e in range(opts.epochs):
        if duration >= opts.patience:
            break
        
        loss_tr = 0
        model.train()
        loop = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        
        for batch, (input_fw_tr, input_rc_tr, y_tr) in loop:
            input_fw_tr = input_fw_tr.to(device)
            input_rc_tr = input_rc_tr.to(device)
            y_tr = y_tr.to(device)
            
            y_pred = model(input_fw_tr, input_rc_tr)
            loss = loss_fn(y_pred, y_tr)
            
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                loss_tr += loss.item()
                loss_avg = loss_tr / (batch+1)
            
            loop.set_description(f'[{e+1}/{opts.epochs}]')
            loop.set_postfix(loss=loss_avg)
        
        loss_tr = loss_tr / len(dataloader_tr)
            
        if dataloader_val is None:
            if loss_tr < (loss_best-opts.delta):
                loss_best = loss_tr
                duration = 0
                
                model_file = 'model_weights.pth'
                model_path = os.path.join(opts.output_dir, model_file)
                torch.save(model.state_dict(), model_path)
            else:
                duration += 1
                
            continue
        
        model.eval()
        loss_fn_val = nn.L1Loss()
        loss_val, nsample_val = 0, 0
        
        for input_fw_val, input_rc_val, y_val in dataloader_val:
            input_fw_val = input_fw_val.to(device)
            input_rc_val = input_rc_val.to(device)
            y_val = y_val.to(device)
            
            with torch.no_grad():
                y_pred = model(input_fw_val, input_rc_val)
                loss_val += loss_fn_val(y_pred, y_val).item()
                nsample_val +=  input_fw_val.size(0)
        
        loss_val = loss_val / nsample_val
        if loss_val < (loss_best-opts.delta):
            loss_best = loss_val
            duration = 0
            
            model_file = 'model_weights.pth'
            model_path = os.path.join(opts.output_dir, model_file)
            torch.save(model.state_dict(), model_path)
        else:
            duration += 1
        print('val_loss: %.4f\n'%loss_val)
        
        
def load_pretrained_embedding(opts, model, pretrained_embedding):
     
    dim = pretrained_embedding.shape[1]
    pretrained_embedding = torch.cat([torch.zeros((1, dim)),
                                      torch.tensor(pretrained_embedding)])
    model.embedding.kmer_embedding.from_pretrained(pretrained_embedding,
                                                   freeze=opts.embedding_freeze,
                                                   padding_idx=0)
    
    return model


if __name__ == '__main__':
    
    args = sys.argv[1:]
    opts = get_opts(args)
    
    data_tr = SimPredData(opts.kmer_size,
                          opts.virus_train_fasta,
                          opts.virus_train_response,
                          opts.host_train_fasta,
                          opts.host_train_response)
    dataloader_tr = DataLoader(data_tr, batch_size=opts.batch_size,
                               shuffle=True, pin_memory=True,
                               num_workers=opts.num_workers)
    
    if (opts.virus_val_fasta is not None) and \
       (opts.virus_val_response is not None) and \
       (opts.host_val_fasta is not None) and \
       (opts.host_val_response is not None):
           
        data_val = SimPredData(opts.kmer_size,
                               opts.virus_val_fasta,
                               opts.virus_val_response,
                               opts.host_val_fasta,
                               opts.host_val_response)
        dataloader_val = DataLoader(data_val, batch_size=opts.batch_size,
                                    shuffle=True, pin_memory=True,
                                    num_workers=opts.num_workers)
    else:
        dataloader_val = None
    
    vocab_size = 4**opts.kmer_size+1
    config = AlnConfig(vocab_size, opts.embedding_dim,
                       opts.hidden_size, opts.lstm_layers,
                       opts.bidirectional)
    model = AEForSimPred(config)
    
    pretrained_embedding = np.load(opts.pretrained_embedding)
    model = load_pretrained_embedding(opts, model, pretrained_embedding)
    
    train(opts, model, dataloader_tr, dataloader_val)