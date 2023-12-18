import torch
import torch.nn as nn
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

from bertconfig import BertConfig
from semantic_encoder import BertForSeqCls
from dataset import SeqClsData
from optimization import get_linear_schedule_with_warmup


def get_opts(args):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--virus-train-data',
                        required=True,
                        type=str)
    
    parser.add_argument('--host-train-data',
                        required=True,
                        type=str)
    
    parser.add_argument('--virus-val-data',
                        required=False,
                        type=str)
    
    parser.add_argument('--host-val-data',
                        required=False,
                        type=str)
    
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        type=str)
    
    parser.add_argument('-p', '--pretrained-model',
                        required=True,
                        type=str)
    
    parser.add_argument('-k', '--kmer-size',
                        required=True,
                        type=int)
    
    parser.add_argument('-l', '--max-seq-len',
                        required=True,
                        type=int)
    
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
                        required=False,
                        type=int,
                        default=1)
    
    parser.add_argument('--learning-rate',
                        required=True,
                        type=float)
    
    parser.add_argument('--layer-lr-base',
                        required=True,
                        type=float)
    
    parser.add_argument('--layer-lr-decay',
                        required=True,
                        type=float)
    
    parser.add_argument('--warmup-proportion',
                        required=True,
                        type=float)
    
    parser.add_argument('--weight-decay',
                        required=False,
                        type=float,
                        default=0.01)
    
    parser.add_argument('--max-gradient-norm',
                        required=False,
                        type=float,
                        default=1)
    
    parser.add_argument('--save-steps',
                        required=False,
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


def finetune(opts, model, dataloader_tr, dataloader_val):
    
    device = 'cuda:'+str(opts.gpu_device)
    model = model.to(device)
    
    max_steps = opts.epochs*len(dataloader_tr) // opts.gradient_accumulation_iters
    warmup_steps = int(opts.warmup_proportion * max_steps)
    no_decay = ["bias", "LayerNorm"]
    groups = [f'layer.{i}' for i in range(opts.num_encoders)]
    no_decay_params, decay_params = []
    
    for layer_num in range(len(groups)):
        lr = opts.layer_lr_base*pow(
            opts.layer_lr_decay, opts.num_encoders-1-layer_num)
        no_decay_params.append(
            {'params': [p for n, p in model.named_parameters() if 
                        any(nd in n for nd in no_decay) and groups[layer_num] in n],
             'weight_decay': 0.0, 'lr': lr})
        decay_params.append(
            {'params': [p for n, p in model.named_parameters() if 
                        (not any(nd in n for nd in no_decay)) and groups[layer_num] in n],
             'weight_decay': opts.weight_decay, 'lr': lr})
        
    other_params = [
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and (not any(layer in n for layer in groups))],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 
                    (not any(nd in n for nd in no_decay)) and (not any(layer in n for layer in groups))],
         'weight_decay': opts.weight_decay}]
    
    grouped_params = no_decay_params+decay_params+other_params
    optimizer = optim.AdamW(grouped_params, lr=opts.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    global_steps, global_batch = 0, 0
    loss_fn = nn.BCELoss()
    model.zero_grad(set_to_none=True)
    
    for current in range(opts.epochs):
        loss_tr, correct, trained_sample = 0, 0, 0
        loop = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        
        for batch, (input_fw, input_rc, label) in loop:
            model.train()
            global_batch += 1
            trained_sample += input_fw.size()[0]
            
            input_fw, input_rc = input_fw.to(device), input_rc.to(device)
            label = label.to(device)
            
            probs = model(input_fw, input_rc)
            loss = loss_fn(probs, label)
            loss_tr += loss.item()
            
            if opts.gradient_accumulation_iters > 1:
                loss = loss / opts.gradient_accumulation_iters
            loss.backward()
            
            y_pred = (probs.detach().cpu().numpy()>0.5).astype('float32')
            correct += len(np.where(label.detach().cpu().numpy() == y_pred)[0])
            
            if global_batch % opts.gradient_accumulation_iters == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opts.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                
                global_steps += 1
                loss_avg = loss_tr / (batch+1)
                acc_tr = correct / trained_sample
            
                loop.set_description(f'[{current+1}/{opts.epochs}]')
                loop.set_postfix(loss=loss_avg, accuracy=acc_tr)
            
            if global_batch % (opts.save_steps*opts.gradient_accumulation_iters) == 0:
                if dataloader_val is not None:
                    model.eval()
                    loss_val, correct_val, nsample_val = 0, 0, 0
                    
                    for input_fw_val, input_rc_val, label_val in dataloader_val:
                        input_fw_val = input_fw_val.to(device)
                        input_rc_val = input_rc_val.to(device)
                        label_val = label_val.to(device)
                        
                        with torch.no_grad():
                            probs = model(input_fw_val, input_rc_val)
                            loss_val += loss_fn(probs, label_val).item()*label_val.size(0)
                            
                            y_pred = torch.gt(probs, 0.5).float()
                            correct_val += torch.eq(y_pred, label_val).sum().item()
                            nsample_val += input_fw_val.size(0)
                    
                    loss_val = loss_val / nsample_val
                    acc_val = correct_val / nsample_val
                    print('val_loss: %.4f'%loss_val, 'val_acc: %.4f'%acc_val)
                    
                model_file = 'epoch%s-step%s-model-weight.pth'%(current+1, global_steps)
                model_path = os.path.join(opts.output_dir, model_file)
                torch.save(model.state_dict(), model_path)


def load_pretraied_weight(model, pretrained_path):
    pretrained_weight = torch.load(pretrained_path)
    model_dict = model.state_dict()

    pretrained_dict = {name: param for name, param in pretrained_weight.items() if
                       name in model_dict}
    print(len(pretrained_dict), 'items successfully matched')
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model
 
        
if __name__ == '__main__':
    
    args = sys.argv[1:]
    opts = get_opts(args)
    
    data_tr = SeqClsData(opts.kmer_size, opts.virus_train_data, opts.host_train_data)
    datald_tr = DataLoader(data_tr, batch_size=opts.per_iter_batch_size,
                           shuffle=True, pin_memory=True, num_workers=opts.num_workers)
    
    if (opts.virus_val_data is not None) and (opts.host_val_data is not None):
        data_val = SeqClsData(opts.kmer_size, opts.virus_val_data, opts.host_val_data)
        datald_val = DataLoader(data_val, batch_size=opts.per_iter_batch_size,
                                pin_memory=True, num_workers=opts.num_workers)
    else:
        datald_val = None
    
    vocab_size = 4**opts.kmer_size+3
    config = BertConfig(vocab_size=vocab_size,
                        hidden_size=opts.dim_model,
                        num_hidden_layers=opts.num_encoders,
                        num_attention_heads=opts.num_attention_heads,
                        intermediate_size=opts.forward_size,
                        max_position_embeddings=opts.max_seq_len,
                        num_labels=1)
    model = BertForSeqCls(config)
    model = load_pretraied_weight(model, opts.pretrained_model)

    finetune(opts, model, datald_tr, datald_val)