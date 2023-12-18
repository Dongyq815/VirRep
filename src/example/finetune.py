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

from optimization import get_linear_schedule_with_warmup
from seqcls import SeqCls, SeqclsConfig
from dataset import SeqData


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
    
    parser.add_argument('-ps', '--pretrained-se-model',
                        required=True,
                        type=str)
    
    parser.add_argument('-pa', '--pretrained-ae-model',
                        required=True,
                        type=str)
    
    parser.add_argument('--freeze-pretrain',
                        required=False,
                        action='store_true',
                        defult=False)
    
    parser.add_argument('-k', '--kmer-size',
                        required=True,
                        type=int)
    
    parser.add_argument('-l', '--max-seq-len',
                        required=True,
                        type=int)
    
    parser.add_argument('--seq-split',
                        required=True,
                        type=int)
    
    parser.add_argument('--transformer-encoders',
                        required=True,
                        type=int)
    
    parser.add_argument('--se-dim-model',
                        required=True,
                        type=int)
    
    parser.add_argument('--attention-heads',
                        required=True,
                        type=int)
    
    parser.add_argument('--se-forward-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--ae-embedding-dim',
                        required=True,
                        type=int)
    
    parser.add_argument('--ae-hidden-size',
                        required=True,
                        type=int)
    
    parser.add_argument('--lstm-layers',
                        required=True,
                        type=int)
    
    parser.add_argument('--lstm-bidirectional',
                        required=False,
                        action='store_true',
                        default=False)
    
    parser.add_argument('--cls-hidden-size',
                        required=True,
                        type=int,
                        nargs='+')
    
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
    
    parser.add_argument('--pretrained-lr',
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


def train(opts, model, dataloader_tr, dataloader_val):
    
    device = 'cuda:' + str(opts.gpu_device)
    model = model.to(device)
    
    max_steps = opts.epochs*len(dataloader_tr) // opts.gradient_accumulation_iters
    warmup_steps = int(opts.warmup_proportion * max_steps)
    
    pretrained = ['aln_encoder', 'semantic_encoder']
    no_decay = ['bias', 'LayerNorm']
    
    pretrained_params = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and
                    any(pt in n for pt in pretrained) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': opts.pretrained_lr},
        {'params': [p for n, p in model.named_parameters() if  
                    any(pt in n for pt in pretrained) and (not any(nd in n for nd in no_decay))],
         'weight_decay': opts.weight_decay, 'lr': opts.pretrained_lr}]
    
    cls_params = [
        {'params': [p for n, p in model.named_parameters() if 
                    (not any(pt in n for pt in pretrained)) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if  
                    (not any(pt in n for pt in pretrained)) and (not any(nd in n for nd in no_decay))],
         'weight_decay': opts.weight_decay}]
    
    grouped_params = pretrained_params + cls_params
    optimizer = optim.AdamW(grouped_params, lr=opts.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    global_steps, global_batch = 0, 0
    loss_fn = nn.BCELoss()
    model.zero_grad(set_to_none=True)
    
    for i in range(opts.epochs):
        loss_tr, correct_tr, nsample_tr = 0, 0, 0
        loop = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        
        for batch, (input_fw_smt, input_rc_smt,
                    input_fw_scp, input_rc_scp, label) in loop:
            
            model.train()
            global_batch += 1
            nsample_tr += input_fw_smt.size()[0]
            
            input_fw_smt = input_fw_smt.to(device)
            input_rc_smt = input_rc_smt.to(device)
            input_fw_scp = input_fw_scp.to(device)
            input_rc_scp = input_rc_scp.to(device)
            label = label.to(device)
            
            probs = model(input_fw_smt, input_rc_smt,
                          input_fw_scp, input_rc_scp)
            loss = loss_fn(probs, label)
            loss_tr += loss.item()
            
            if opts.gradient_accumulation_iters > 1:
                loss = loss / opts.gradient_accumulation_iters
            loss.backward()
            
            y_pred = (probs.detach().cpu().numpy()>0.5).astype('float32')
            correct_tr += len(np.where(label.detach().cpu().numpy() == y_pred)[0])
            
            loss_avg = loss_tr / (batch+1)
            acc_tr = correct_tr / nsample_tr
        
            loop.set_description(f'[{i+1}/{opts.epochs}]')
            loop.set_postfix(loss=loss_avg, accuracy=acc_tr)
            
            if global_batch % opts.gradient_accumulation_iters == 0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         opts.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                global_steps += 1
            
            if global_batch % (opts.save_steps*opts.gradient_accumulation_iters) == 0:
                if dataloader_val is not None:
                    model.eval()
                    loss_val, correct_val, nsample_val = 0, 0, 0
                    
                    for code_fw_smt, code_rc_smt, code_fw_scp, \
                        code_rc_scp, y_true in dataloader_val:
                            
                            code_fw_smt = code_fw_smt.to(device)
                            code_rc_smt = code_rc_smt.to(device)
                            code_fw_scp = code_fw_scp.to(device)
                            code_rc_scp = code_rc_scp.to(device)
                            y_true = y_true.to(device)
                            
                            with torch.no_grad():
                                
                                probs = model(code_fw_smt, code_rc_smt, 
                                              code_fw_scp, code_rc_scp)
                                loss_val += loss_fn(probs, y_true).item()*code_fw_smt.size(0)
                                y_pred = torch.gt(probs, 0.5).float()
                                correct_val += torch.eq(y_pred, y_true).sum().item()
                                nsample_val += code_fw_smt.size(0)
                    
                    loss_val = loss_val / nsample_val
                    acc_val = correct_val / nsample_val
                    print('--val_loss: %.4f'%loss_val, '--val_acc: %.4f'%acc_val, '\n')
                
                model_file = 'epoch%s-step%s-model.pth'%(i+1, global_steps)
                model_path = os.path.join(opts.output_dir, model_file)
                torch.save(model.state_dict(), model_path)

        
def load_pretrained_weights(model, se_weight_path,
                            ae_weight_path,
                            freeze_pretrain=False):
    
    se_weight = torch.load(se_weight_path, map_location=torch.device('cpu'))
    ae_weight = torch.load(ae_weight_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    
    se_pretrained = {}
    for name, param in se_weight.items():
        name_new = 'semantic_encoder.'+name
        
        if name_new.find('bert') != -1 and name_new in model_dict:
            se_pretrained[name_new] = param
    print('Semantic Encoder:', len(se_pretrained), 'items sucessfully matched')
    
    ae_pretrained = {}
    for name, param in ae_weight.items():
        name_new = 'aln_encoder.'+name
        
        if name_new in model_dict:
            ae_pretrained[name_new] = param
    print('Alignment Encoder:', len(ae_pretrained), 'items sucessfully matched')
    
    model_dict.update(se_pretrained)
    model_dict.update(ae_pretrained)
    model.load_state_dict(model_dict)
    
    if freeze_pretrain:
        for name, param in model.named_parameters():
            if name in se_pretrained or name in ae_pretrained:
                    param.requires_grad = False
    
    return model     


if __name__ == '__main__':
    
    args = sys.argv[1:]
    opts = get_opts(args)
    
    data_tr = SeqData(opts.kmer_size, opts.max_seq_len,
                      opts.virus_train_data, opts.host_train_data)
    datald_tr = DataLoader(data_tr, batch_size=opts.per_iter_batch_size,
                           shuffle=True, pin_memory=True, num_workers=opts.num_workers)
    
    if (opts.virus_val_data is not None) and (opts.host_val_data is not None):
        data_val = SeqData(opts.kmer_size, opts.max_seq_len,
                           opts.virus_val_data, opts.host_val_data)
        datald_val = DataLoader(data_val, batch_size=opts.per_iter_batch_size,
                                num_workers=opts.num_workers, pin_memory=True)
    
    config = SeqclsConfig(opts.kmer_size,
                          opts.max_seq_len,
                          opts.seq_split,
                          opts.se_dim_model,
                          opts.transformer_encoders,
                          opts.attention_heads,
                          opts.se_forward_size,
                          opts.ae_embedding_dim,
                          opts.ae_hidden_size,
                          opts.lstm_layers,
                          opts.lstm_bidirectional,
                          opts.cls_hidden_size)
    model = SeqCls(config)
    
    model = load_pretrained_weights(model, opts.pretrained_se_model,
                                    opts.pretrained_ae_model)
    train(opts, model, datald_tr, datald_val)