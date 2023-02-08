import torch
from seqcls_xl_BiLSTM import SeqCls_XL_Concat, Seqclsconfig_XL_Concat
from utils_BiLSTM import (readfa, seqfrag, encoder_semantic,
                          encoder_seqcomp, adjust_uncertain_nt)
from Bio.Seq import Seq
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score)
from tqdm import tqdm
import numpy as np
import pandas as pd
import math


def tnr_score(y_true, y_pred):
    neg_true = np.where(y_true == 0)[0]
    neg_pred = np.where(y_pred == 0)[0]
    neg_ins = np.intersect1d(neg_true,neg_pred)
    return len(neg_ins) / len(neg_true)


def predict_seq(model, fapath, minlen=1000, batchsz=128, device=0):
    
    device = 'cuda:' + str(device)
    model = model.to(device)
    model.eval()
    
    seqs, headers = readfa(fapath)
    seqs = adjust_uncertain_nt(seqs)
    
    result = {'header': [], 'length': [], 'score': []}
    letter_set = ['A', 'C', 'G', 'T']
    
    for i in tqdm(range(len(seqs)), unit='seqs', total=len(seqs)):
        seq = seqs[i]
        seqlen = len(seq)
        
        if seqlen < minlen:
            continue
        
        bad = 0
        for base in seq:
            if base not in letter_set:
                bad += 1
        if bad >= 0.2*seqlen:
            continue
        
        frag_fw = seqfrag(seq)
        frag_rc = []
        
        for frag in frag_fw:
            frag_rc.append(str(Seq(frag).reverse_complement()))
        
        seqcode_smt_fw, seqcode_smt_rc = [], []
        seqcode_scp_fw, seqcode_scp_rc = [], []
        for s in range(len(frag_fw)):
            code_smt_fw = encoder_semantic(frag_fw[s])
            code_smt_rc = encoder_semantic(frag_rc[s])
            code_scp_fw = encoder_seqcomp(frag_fw[s])
            code_scp_rc = encoder_seqcomp(frag_rc[s])
            
            seqcode_smt_fw.append(code_smt_fw)
            seqcode_smt_rc.append(code_smt_rc)
            seqcode_scp_fw.append(code_scp_fw)
            seqcode_scp_rc.append(code_scp_rc)
            
        assert len(seqcode_smt_fw) == len(seqcode_smt_rc)
        assert len(seqcode_scp_fw) == len(seqcode_scp_rc)
        assert len(seqcode_smt_fw) == len(seqcode_scp_fw)
        
        nbatch = math.ceil(len(seqcode_smt_fw) / batchsz)
        if nbatch == 0:
            continue
        
        score = 0
        for nb in range(nbatch):
            start = nb*batchsz
            end = start+batchsz
            input_smt_fw = torch.IntTensor(seqcode_smt_fw[start:end])
            input_smt_rc = torch.IntTensor(seqcode_smt_rc[start:end])
            input_scp_fw = torch.IntTensor(seqcode_scp_fw[start:end])
            input_scp_rc = torch.IntTensor(seqcode_scp_rc[start:end])
            
            input_smt_fw = input_smt_fw.to(device)
            input_smt_rc = input_smt_rc.to(device)
            input_scp_fw = input_scp_fw.to(device)
            input_scp_rc = input_scp_rc.to(device)
            
            with torch.no_grad():
                score += model(input_smt_fw, input_smt_rc,
                               input_scp_fw, input_scp_rc).sum().item()
        
        avg_score = score / len(seqcode_smt_fw)
        result['header'].append(headers[i])
        result['length'].append(seqlen)
        result['score'].append(avg_score)
    
    return pd.DataFrame(result)


if __name__ == '__main__':
    
    k = 7
    maxseqlen = 500
    split = 2
    bert_hidden_size = 256
    bert_num_encoders = 8
    bert_num_attention_heads = 8
    bert_forward_size = 512
    skipgram_embed_size = 100
    skipgram_embed_dropout = 0.1
    covpredictor_hidden_size = 256
    covpredictor_num_lstm_layers = 2
    covpredictor_lstm_dropout = 0.15
    covpredictor_lstm_bidirectional = True
    covpredictor_pooler_dropout = 0.1
    linear_units = 1024
    linear_dropout = 0.2
    
    config = Seqclsconfig_XL_Concat(k, maxseqlen, split,
                                    bert_hidden_size,
                                    bert_num_encoders,
                                    bert_num_attention_heads,
                                    bert_forward_size,
                                    skipgram_embed_size,
                                    skipgram_embed_dropout,
                                    covpredictor_hidden_size,
                                    covpredictor_num_lstm_layers,
                                    covpredictor_lstm_dropout,
                                    covpredictor_lstm_bidirectional,
                                    covpredictor_pooler_dropout,
                                    linear_units, linear_dropout)
    model = SeqCls_XL_Concat(config)
    
    weightpath = '../../model/final/XL/concat/BiLSTM/' \
        'epoch3-step33.0k-0.1920-0.9322-0.1878-0.9342/model_weights.pth'
    model.load_state_dict(torch.load(
        weightpath, map_location=torch.device('cpu')))
    
    # fapath = '../../data/metagenome/test/test1/IMGVR/20k-50k/sample30/' \
    #     'IMGVR+IMG-GEM_merged_20k-50k_sample30.fna'
    fapath = '../../data/virus/database/HGAVD/vir_genome.fna'
    prediction = predict_seq(model, fapath, minlen=1000, batchsz=256, device=1)
    
    score = np.array(prediction['score'].tolist())
    y_pred = np.int64(score > 0.5)
    y_true = np.ones(1177)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tnr = tnr_score(y_true, y_pred)
    auc = roc_auc_score(y_true, score)