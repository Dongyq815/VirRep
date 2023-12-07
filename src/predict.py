import torch
from torch.utils.data import DataLoader
from utils import readfa, SeqData
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import os


def score_segment(model, fapath, min_seqlen, batchsz=256,
                  num_workers=0, cpu=False, cpu_threads=1,
                  gpu_device=0):
    
    if not cpu:
        device = 'cuda:' + str(gpu_device)
        model = model.to(device)
    else:
        torch.set_num_threads(cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        
    model.eval()
    
    seqs_all, headers_all = readfa(fapath)
    seq_data = SeqData(seqs_all, min_seqlen)
    dataloader = DataLoader(seq_data, batch_size=batchsz, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        
    print('Scoring segments...')
    headers_dict = {idx: headers_all[idx] for idx in range(len(seqs_all))}
    seqlen_dict = {idx: len(seqs_all[idx]) for idx in range(len(seqs_all))}
    scores_dict = {idx: [] for idx in range(len(seqs_all))}
    positions_dict = {idx: [] for idx in range(len(seqs_all))}
    loop = tqdm(dataloader, unit='batch', total=len(dataloader))
    
    for input_fw_smt, input_rc_smt, input_fw_aln, \
        input_rc_aln, positions, indices in loop:
        
        if not cpu:
            input_fw_smt = input_fw_smt.to(device)
            input_rc_smt = input_rc_smt.to(device)
            input_fw_aln = input_fw_aln.to(device)
            input_rc_aln = input_rc_aln.to(device)
        positions = positions.numpy().squeeze().tolist()
        indices = indices.numpy().squeeze().tolist()
        
        with torch.no_grad():
            scores = model(input_fw_smt, input_rc_smt, input_fw_aln,
                           input_rc_aln).cpu().numpy().squeeze().tolist()
        
        assert len(scores) == len(indices)
        assert len(scores) == len(positions)
        
        for s in range(len(scores)):
            scores_dict[indices[s]].append(scores[s])
            positions_dict[indices[s]].append(positions[s])
                
    return headers_dict, seqlen_dict, scores_dict, positions_dict


def calculate_score_mean(score_list, percent=0.8):
    
    scores_sorted = sorted(score_list, reverse=True)
    num2sum = math.ceil(percent * len(scores_sorted))
    score_sum = sum(scores_sorted[0:num2sum])
    
    return score_sum / num2sum


def extend(headers_dict, seqlen_dict, scores_dict,
           positions_dict, minscore, posthoc=False):
    
    result = {'seqname':[], 'seqlen': [], 'hitlen':[],
              'score':[], 'start':[], 'end':[]}
    
    for idx, score_list in scores_dict.items():
        if len(score_list) == 0:
            continue
        
        score_mean = calculate_score_mean(score_list)
        if (score_mean >= minscore) and (not posthoc):
            
            result['seqname'].append(headers_dict[idx]+'||full')
            result['seqlen'].append(seqlen_dict[idx])
            result['hitlen'].append(seqlen_dict[idx])
            result['score'].append(score_mean)
            result['start'].append(1)
            result['end'].append(seqlen_dict[idx])
            
        else:
            positions_gt = np.where(np.array(score_list) >= minscore)[0]
            if len(positions_gt) == 0:
                continue
            
            pos_first, pos_last = np.min(positions_gt), np.max(positions_gt)
            pos_gt = pos_first
            left_boundary = 0
            
            while pos_gt <= pos_last:
                path = [pos_gt]
                score_sum = score_list[pos_gt]
                
                while min(path) > left_boundary or (max(path)+1) < len(score_list):
                    if min(path) == left_boundary:
                        score_sum_left = score_sum
                        score_sum_right = score_sum + score_list[max(path)+1]
                    elif (max(path)+1) == len(score_list):
                        score_sum_left = score_sum + score_list[min(path)-1]
                        score_sum_right = score_sum
                    else:
                        score_sum_left = score_sum + score_list[min(path)-1]
                        score_sum_right = score_sum + score_list[max(path)+1]
                    
                    if score_sum_left > score_sum_right:
                        score_mean = score_sum_left / (len(path)+1)
                        pos = min(path) - 1
                    else:
                        score_mean = score_sum_right / (len(path)+1)
                        pos = max(path) + 1
                    
                    if score_mean >= minscore:
                        path.append(pos)
                    else:
                        break
                
                end_real = positions_dict[idx][max(path)]
                start_real = positions_dict[idx][min(path)]
                prov_len = (end_real-start_real+1)*1000
                score_prov = sum(score_list[min(path):max(path)+1]) / (max(path)-min(path)+1)
                
                result['seqname'].append(headers_dict[idx]+'||partial')
                result['seqlen'].append(seqlen_dict[idx])
                result['hitlen'].append(prov_len)
                result['score'].append(score_prov)
                result['start'].append(start_real*1000+1)
                result['end'].append((end_real+1)*1000)
                
                if max(path) >= pos_last:
                    break
                else:
                    pos_gt = positions_gt[np.where(positions_gt > max(path))[0][0]]
                    left_boundary = max(path)+1
        
    return pd.DataFrame(result)


def merge(prediction, maxgap=5000, maxfrac=0.1,
          provirus_minlen=5000, provirus_minfrac=1):
    
    prediction.sort_values(by=['seqname', 'start'], inplace=True)
    headers_hit = prediction['seqname'].tolist()
    
    freq_dict = {}
    for header in headers_hit:
        if header not in freq_dict:
            freq_dict[header] = 1
        else:
            freq_dict[header] += 1
            
    headers_hit_set = sorted(list(set(headers_hit)))
    headers_hit = np.array(headers_hit)
    header_merge, hitlen_merge, score_merge = [], [], []
    start_merge, end_merge = [], []
    
    for header in headers_hit_set:
        freq = freq_dict[header]
        seqlen = prediction[prediction['seqname'] == header]['seqlen'].tolist()[0]
        
        if freq > 1:
            header_tmp, hitlen_tmp, score_tmp = [], [], []
            start_tmp, end_tmp = [], []
            
            indices = np.where(headers_hit == header)[0]
            idx = indices[0]
            
            while idx <= indices[-1]:
                idx_cur = idx
                idx_next = idx_cur+1
                len_sum = prediction.iloc[idx_cur, 2]
                
                while idx_next <= indices[-1]:
                    gap = prediction.iloc[idx_next, 4]-prediction.iloc[idx_cur, 5]-1
                    len_sum += prediction.iloc[idx_next, 2]
                    
                    if (gap <= maxgap or gap <= maxfrac*len_sum) and (2*gap < len_sum):
                        idx_cur = idx_next
                        idx_next += 1
                    else:
                        break
                    
                pos_start = prediction.iloc[idx, 4]
                pos_end = prediction.iloc[idx_cur, 5]
                hitlen = pos_end-pos_start+1
                
                if hitlen >= provirus_minlen or hitlen >= provirus_minfrac * seqlen:
                    header_tmp.append(header+'_'+str(pos_start)+'-'+str(pos_end))
                    hitlen_tmp.append(pos_end-pos_start+1)
                    score_tmp.append(prediction.iloc[idx:idx_next, 3].mean())
                    start_tmp.append(pos_start)
                    end_tmp.append(pos_end)
                    
                idx = idx_next
            
            header_merge.extend(header_tmp)
            start_merge.extend(start_tmp)
            end_merge.extend(end_tmp)
            score_merge.extend(score_tmp)
            hitlen_merge.extend(hitlen_tmp)
                
        else:
            idx = np.where(headers_hit == header)[0]
            assert len(idx) == 1
            
            hitlen = prediction.iloc[idx[0], 2]
            if hitlen >= provirus_minlen or hitlen >= provirus_minfrac*seqlen:
                header_merge.append(header)
                hitlen_merge.append(prediction.iloc[idx[0], 2])
                score_merge.append(prediction.iloc[idx[0], 3])
                start_merge.append(prediction.iloc[idx[0], 4])
                end_merge.append(prediction.iloc[idx[0], 5])
    
    prediction_merge = pd.DataFrame({'seqname':header_merge,
                                     'hitlen':hitlen_merge,
                                     'score':score_merge,
                                     'start':start_merge,
                                     'end':end_merge})
    
    return prediction_merge


def check_provirus(hitname):
    
    suffix = hitname.split('||')[-1]
    if suffix.find('partial') != -1:
        flag = True
    else:
        flag = False
        
    return flag


def parse_minscore(minscore_dict):
    
    minlen, maxlen = [], []
    for len_range, minscore in minscore_dict.items():
        len_split = len_range.split('-')
        minlen.append(int(len_split[0]))
        
        if len_split[1] == 'Inf':
            maxlen.append('Inf')
        else:
            maxlen.append(int(len_range.split('-')[1]))
    
    return minlen, maxlen


def check_minscore(hitlen, minscore_dict):
    
    minlen, maxlen = parse_minscore(minscore_dict)
    idx = np.where(np.array(minlen) <= hitlen)[0][-1]
    key = str(minlen[idx])+'-'+str(maxlen[idx])
    
    return minscore_dict[key]


def get_idx(header, headers_dict):
    
    headers2idx = {header: idx for idx, header in headers_dict.items()}
    return headers2idx[header]


def posthoc_check(pred_merge, headers_dict, seqlen_dict,
                  scores_dict, positions_dict, minscore_dict):
    
    header_keep, hitlen_keep, score_keep, start_keep, end_keep = [], [], [], [], []
    check_flag = False
    
    for i in range(len(pred_merge)):
        hitname = pred_merge.iloc[i, 0]
        hitlen = pred_merge.iloc[i, 1]
        score = pred_merge.iloc[i, 2]
        start = pred_merge.iloc[i, 3]
        end = pred_merge.iloc[i, 4]
        provirus_flag = check_provirus(hitname)
        minscore = check_minscore(hitlen, minscore_dict)
        
        if provirus_flag and score < minscore:
            check_flag = True
            header = hitname.split('||')[0]
            idx = get_idx(header, headers_dict)
            
            start = int((start-1) / 1000)
            begin = positions_dict[idx].index(start)
            end = int(end / 1000) - 1
            terminal = positions_dict[idx].index(end)
            
            headers_dict_tmp = {idx: headers_dict[idx]}
            seqlen_dict_tmp = {idx: seqlen_dict[idx]}
            scores_dict_tmp = {idx: scores_dict[idx][begin:terminal+1]}
            positions_dict_tmp = {idx: positions_dict[idx][begin:terminal+1]}
            pred_extend = extend(headers_dict_tmp, seqlen_dict_tmp,
                                 scores_dict_tmp, positions_dict_tmp,
                                 minscore, posthoc=True)
            
            pred_merge_tmp = merge(pred_extend)
            header_keep.extend(pred_merge_tmp['seqname'].tolist())
            hitlen_keep.extend(pred_merge_tmp['hitlen'].tolist())
            score_keep.extend(pred_merge_tmp['score'].tolist())
            start_keep.extend(pred_merge_tmp['start'].tolist())
            end_keep.extend(pred_merge_tmp['end'].tolist())
            
        else:
            header_keep.append(hitname)
            hitlen_keep.append(hitlen)
            score_keep.append(score)
            start_keep.append(start)
            end_keep.append(end)
            
    pred_df =  pd.DataFrame({'seqname': header_keep, 'hitlen': hitlen_keep,
                             'score': score_keep, 'start': start_keep,
                             'end': end_keep})
    
    if check_flag:
        return posthoc_check(pred_df, headers_dict, seqlen_dict,
                             scores_dict, positions_dict, minscore_dict)
    else:
        return pred_df
    


def predict(model, fapath,
            min_seqlen=1000,
            baseline=0.5,
            batchsz=256,
            num_workers=0,
            cpu=False,
            cpu_threads=1,
            gpu_device=0,
            provirus_off=False,
            maxgap=5000,
            maxfrac=0.1,
            provirus_minlen=5000,
            provirus_minfrac=1,
            minscore_dict={'1000-Inf': 0.5}):
    
    headers_dict, seqlen_dict, scores_dict, positions_dict = score_segment(
        model, fapath, min_seqlen, batchsz, num_workers, cpu, cpu_threads, gpu_device)
        
    if provirus_off:
        pred_df = {'seqname': [], 'seqlen': [], 'score': []}
    
        for idx, scores in scores_dict.items():
            if len(scores) > 0:
                
                pred_df['seqname'].append(headers_dict[idx])
                pred_df['seqlen'].append(seqlen_dict[idx])
                pred_df['score'].append(sum(scores) / len(scores))
            
        pred_df = pd.DataFrame(pred_df)
        
    else:
        extension = extend(headers_dict, seqlen_dict, scores_dict,
                           positions_dict, baseline)
        pred_merge = merge(extension, maxgap, maxfrac,
                           provirus_minlen, provirus_minfrac)
        pred_df = posthoc_check(pred_merge, headers_dict, seqlen_dict,
                                scores_dict, positions_dict, minscore_dict)
    
    return pred_df