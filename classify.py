import torch
from utils_BiLSTM import (readfa, seqfrag_v2, encoder_semantic,
                          encoder_seqcomp, adjust_uncertain_nt)
from Bio.Seq import Seq
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score)
import numpy as np
import pandas as pd
import math


def classify(model, fapath, seq_minlen, provirus_minlen, provirus_minfrac,
             threshold=0.5, provirus_off=False, batchsz=128, device=0):
    
    device = 'cuda:' + str(device)
    model = model.to(device)
    model.eval()
    
    seqs_all, headers_all = readfa(fapath)
    seqs_all = adjust_uncertain_nt(seqs_all)
    
    if provirus_off:
        result = {'header':[], 'length':[], 'score':[]}
    else:
        result = {'header':[], 'length':[], 'score':[],
                  'provirus':[], 'start':[], 'end':[]}
    letter_set = ['A', 'C', 'G', 'T']
    
    for i in tqdm(range(len(seqs_all)), unit='seqs'):
        seq = seqs_all[i]
        seqlen = len(seq)
        
        if seqlen < seq_minlen:
            continue
        
        bad = 0
        for base in seq:
            if base not in letter_set:
                bad += 1
        if bad >= 0.2*seqlen:
            continue
        
        fragseq_fw, fragnum_fw = seqfrag_v2(seq)
        fragseq_rc = []
        
        for frag in fragseq_fw:
            fragseq_rc.append(str(Seq(frag).reverse_complement()))
        
        seqcode_smt_fw, seqcode_smt_rc = [], []
        seqcode_scp_fw, seqcode_scp_rc = [], []
        for s in range(len(fragseq_fw)):
            code_smt_fw = encoder_semantic(fragseq_fw[s])
            code_smt_rc = encoder_semantic(fragseq_rc[s])
            code_scp_fw = encoder_seqcomp(fragseq_fw[s])
            code_scp_rc = encoder_seqcomp(fragseq_rc[s])
            
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
        
        score_frags = []
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
                score_frags.append(model(
                    input_smt_fw, input_smt_rc, input_scp_fw,
                    input_scp_rc).cpu().numpy())
        
        score_frags = np.concatenate(score_frags, axis=0)
        score_frags = np.squeeze(score_frags)
        score_entire = np.mean(score_frags)

        if provirus_off:
            result['header'].append(headers_all[i])
            result['length'].append(seqlen)
            result['score'].append(score_entire)
            continue
        
        if score_entire >= threshold:
            result['header'].append(headers_all[i]+'||full')
            result['length'].append(seqlen)
            result['score'].append(score_entire)
            result['provirus'].append('No')
            result['start'].append(1)
            result['end'].append(len(seq))
        else:
            indices_gt = np.where(score_frags >= threshold)[0]
            
            if len(indices_gt) == 0:
                result['header'].append(headers_all[i])
                result['length'].append(seqlen)
                result['score'].append(score_entire)
                result['provirus'].append('No')
                result['start'].append('-')
                result['end'].append('-')
            else:
                flag = False
                idx_first = np.min(indices_gt)
                idx_last = np.max(indices_gt)
                idx_gt= idx_first
                
                while idx_gt <= idx_last:
                    number = [idx_gt]
                    score_sub_sum = score_frags[idx_gt]
                    
                    for pos in range(idx_gt+1, len(score_frags)):
                        
                        score_sub_sum += score_frags[pos]
                        score_sub_mean = score_sub_sum / (len(number)+1)
                        if score_sub_mean >= threshold:
                            number.append(pos)
                        else:
                            break
                    
                    end = number[-1]
                    end_real = fragnum_fw[end]
                    start_real = fragnum_fw[idx_gt]
                    provlen = (end_real-start_real+1)*1000
                    score_prov = sum(score_frags[idx_gt:end+1]) / (end-idx_gt+1)
                    
                    if provlen >= provirus_minlen or provlen >= seqlen*provirus_minfrac:
                        flag = True
                        result['header'].append(headers_all[i]+'||partial')
                        result['length'].append(provlen)
                        result['score'].append(score_prov)
                        result['provirus'].append('Yes')
                        result['start'].append(start_real*1000+1)
                        result['end'].append((end_real+1)*1000)
                    
                    if end >= idx_last:
                        break
                    else:
                        idx_gt = indices_gt[np.where(indices_gt > end)[0][0]]

                if not flag:
                    result['header'].append(headers_all[i])
                    result['length'].append(seqlen)
                    result['score'].append(score_entire)
                    result['provirus'].append('No')
                    result['start'].append('-')
                    result['end'].append('-')
        
    return pd.DataFrame(result)


def merge_neighbor(prediction, threshold=0.5, maxgap=5000, maxfrac=0.1):
    
    pred_cellular = prediction.loc[prediction.score < threshold, :]
    pred_virus = prediction.loc[prediction.score >= threshold, :]
    pred_virus['start'] = pred_virus['start'].astype('int64')
    pred_virus['end'] = pred_virus['end'].astype('int64')
    pred_virus.sort_values(by=['header', 'start'], inplace=True)
    header_virus = pred_virus['header'].tolist()
    
    header_dict = {}
    for header in header_virus:
        if header not in header_dict:
            header_dict[header] = 1
        else:
            header_dict[header] += 1
    
    header_set = list(set(header_virus))
    header_set.sort()
    header_virus = np.array(header_virus)
    header_merge, start_merge, end_merge = [], [], []
    score_merge, provirus_merge, length_merge = [], [], []
    
    for header in header_set:
        freq = header_dict[header]
        if freq > 1:
            header_tmp, start_tmp, end_tmp = [], [], []
            score_tmp, provirus_tmp, length_tmp = [], [], []
            indices = np.where(header_virus == header)[0]
            idx = indices[0]
            
            while idx <= indices[-1]:
                idx_cur = idx
                idx_next = idx_cur+1
                len_sum = pred_virus.iloc[idx_cur, 1]
                
                while idx_next <= indices[-1]:
                    gap = pred_virus.iloc[idx_next, 4]-pred_virus.iloc[idx_cur, 5]-1
                    len_sum += pred_virus.iloc[idx_next, 1]
                    
                    if gap <= maxgap or gap <= maxfrac*len_sum:
                        idx_cur = idx_next
                    else:
                        break
                    idx_next += 1
                
                pos_start = pred_virus.iloc[idx, 4]
                pos_end = pred_virus.iloc[idx_cur, 5]
                header_tmp.append(header+'_'+str(pos_start)+'-'+str(pos_end))
                start_tmp.append(pos_start)
                end_tmp.append(pos_end)
                score_tmp.append(pred_virus.iloc[idx:idx_next, 2].mean())
                len_tig = end_tmp[-1]-start_tmp[-1]+1
                length_tmp.append(len_tig)
                provirus_tmp.append('Yes')
                
                idx = idx_next
            
            header_merge.extend(header_tmp)
            start_merge.extend(start_tmp)
            end_merge.extend(end_tmp)
            score_merge.extend(score_tmp)
            provirus_merge.extend(provirus_tmp)
            length_merge.extend(length_tmp)
                
        else:
            header_merge.append(header)
            idx = np.where(header_virus == header)[0]
            assert len(idx) == 1
            start_merge.append(pred_virus.iloc[idx[0], 4])
            end_merge.append(pred_virus.iloc[idx[0], 5])
            length_merge.append(pred_virus.iloc[idx[0], 1])
            score_merge.append(pred_virus.iloc[idx[0], 2])
            provirus_merge.append(pred_virus.iloc[idx[0], 3])
    
    pred_virus_merge = pd.DataFrame({'header':header_merge,
                                     'length':length_merge,
                                     'score':score_merge,
                                     'provirus':provirus_merge,
                                     'start':start_merge,
                                     'end':end_merge})
    
    pred_all_merge = pd.concat([pred_virus_merge, pred_cellular], axis=0)
            
    return pred_all_merge


def predict(model, fapath, seq_minlen, provirus_minlen, provirus_minfrac,
            threshold=0.5, provirus_off=False, batchsz=128, device=0,
            maxgap=5000, maxfrac=0.1):
    
    prediction = classify(model, fapath, seq_minlen, provirus_minlen,
                          provirus_minfrac, threshold, provirus_off,
                          batchsz, device)
    pred_merge = merge_neighbor(prediction, threshold, maxgap, maxfrac)
    
    return pred_merge