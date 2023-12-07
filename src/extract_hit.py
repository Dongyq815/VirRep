from predict import check_minscore
from utils import readfa, writefa


def split_suffix(seqname_list):
    
    prefix, suffix = [], []
    for name in seqname_list:
        prefix.append(name.split('||')[0])
        suffix.append(name.split('||')[1])
    
    return prefix, suffix


def extract_seq(prediction, fapath, minscore_dict, minlen, outpath):
    
    seqs_all, headers_all = readfa(fapath)
    header2idx_dict = {headers_all[idx]: idx for idx in range(len(headers_all))}
    seqs_hit, headers_hit = [], []
    
    for i in range(len(prediction)):
        if prediction['seqlen'][i] < minlen:
            continue
        
        minscore = check_minscore(prediction['seqlen'][i], minscore_dict)
        if prediction['score'][i] >= minscore:
            
            idx = header2idx_dict[prediction['seqname'][i]]
            seqs_hit.append(seqs_all[idx])
            headers_hit.append(prediction['seqname'][i])
            
    writefa(headers_hit, seqs_hit, outpath)


def extract_seq_prov(prediction, fapath, minlen, outpath):
    
    hitname = prediction['seqname'].tolist()
    header_hit, suffix = split_suffix(hitname)
    
    start = prediction['start'].tolist()
    end = prediction['end'].tolist()
    
    seqs, headers = readfa(fapath)
    lookup_dict = {headers[idx]: idx for idx in range(len(headers))}
    seqnames_hit, seqs_hit = [], []
    
    for i in range(len(header_hit)):
        idx = lookup_dict[header_hit[i]]
        seq = seqs[idx][(start[i]-1):end[i]]
        
        if len(seq) >= minlen:
            seqnames_hit.append(header_hit[i]+'||'+suffix[i])
            seqs_hit.append(seq)
    
    writefa(seqnames_hit, seqs_hit, outpath)