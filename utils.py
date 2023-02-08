from functools import reduce
from tqdm import tqdm
import math


id2token_semantic = ['[UNK]', '[CLS]', '[MASK]']
kmerset = reduce(lambda x, y: [i+j for i in x for j in y], [['A', 'C', 'G', 'T']]*7)
id2token_semantic.extend(kmerset)
token2id_semantic = {token:idx for idx, token in enumerate(id2token_semantic)}

id2token_seqcomp = ['[UNK]']
id2token_seqcomp.extend(kmerset)
token2id_seqcomp = {token:idx for idx, token in enumerate(id2token_seqcomp)}


def readfa(filename):
    
    file = open(filename, 'r')
    seq = ''
    sequences = []
    names = []
    
    for line in file:
        if line.startswith(">"):
            header = line[1:-1]
            names.append(header)
            if seq != '':
                sequences.append(seq)
            seq = ''
            continue
        if line.startswith("\n"):
            continue
        seq += line[:-1]
    
    sequences.append(seq)
    file.close()
    return sequences, names


def seqfrag(seq):
    
    letter_set = ['A', 'C', 'G', 'T']
    nfrag = math.ceil(len(seq) / 1000)
    fragment = []
    
    for i in range(nfrag):
        if (i+1) < nfrag:
            start = i*1000
        elif (i+1) == nfrag:
            start = len(seq)-1000
        end = start+1000
        
        frag = seq[start:end]
        
        bad1, bad2 = 0, 0
        for pos in range(len(frag)):
            if frag[pos] not in letter_set:
                if pos < 500:
                    bad1 += 1
                else:
                    bad2 += 1
        
        if bad1 <= 50 and bad2 <= 50:
            fragment.append(frag)
    
    return fragment

#### 0726
def seqfrag_v2(seq):
    letter_set = ['A', 'C', 'G', 'T']
    nfrag = math.ceil(len(seq) / 1000)
    frag_seq, frag_number = [], []
    
    for i in range(nfrag):
        if (i+1) < nfrag:
            start = i*1000
        elif (i+1) == nfrag:
            start = len(seq)-1000
        end = start+1000
        
        frag = seq[start:end]
        
        bad1, bad2 = 0, 0
        for pos in range(len(frag)):
            if frag[pos] not in letter_set:
                if pos < 500:
                    bad1 += 1
                else:
                    bad2 += 1
        
        if bad1 <= 50 and bad2 <= 50:
            frag_seq.append(frag)
            frag_number.append(i)
    
    return frag_seq, frag_number


def encoder_semantic(seq):
    
    seqcode = []
    nfrag = math.ceil(len(seq) / 500)
    
    for i in range(nfrag):
        start = i*500
        end = start+500
        seqfrag = seq[start:end]
        seqcode.extend(encoder_semantic_frag(seqfrag))
    
    return seqcode


def encoder_semantic_frag(seqfrag):
    
    fragcode = [1]
    pos = 0
    
    while (pos+7) <= len(seqfrag):
        token = seqfrag[pos:pos+7]
        fragcode.append(token2id_semantic.get(token, 0))
        pos += 1
        
    return fragcode


def encoder_seqcomp(seq):
    seqcode = []
    nfrag = math.ceil(len(seq) / 500)
    
    for i in range(nfrag):
        start = i*500
        end = start+500
        seqfrag = seq[start:end]
        seqcode.extend(encoder_seqcomp_frag(seqfrag))
        
    return seqcode


def encoder_seqcomp_frag(seqfrag):
    
    letter_set = ['A', 'C', 'G', 'T']
    fragcode = []
    pos = 0
    
    while (pos+7) <= len(seqfrag):
        token = seqfrag[pos:pos+7]
        bad = 0
        
        for base in token:
            if base not in letter_set:
                bad += 1
                break
            
        if bad == 0:
            fragcode.append(token2id_seqcomp[token])
        else:
            fragcode.append(0)
        
        pos += 1
    
    return fragcode


def adjust_seq(seq):
        
    seq = seq.upper()
    seqlist = list(seq)
    
    for i in range(len(seqlist)):
        if seqlist[i] == 'R':
            seqlist[i] = 'A'
            
        elif seqlist[i] == 'Y':
            seqlist[i] = 'T'
            
        elif seqlist[i] == 'M':
            seqlist[i] = 'C'
            
        elif seqlist[i] == 'K':
            seqlist[i] = 'G'
            
        elif seqlist[i] == 'S':
            seqlist[i] = 'C'
            
        elif seqlist[i] == 'W':
            seqlist[i] = 'A'
        
        elif seqlist[i] == 'H':
            seqlist[i] = 'A'
            
        elif seqlist[i] == 'B':
            seqlist[i] = 'C'
            
        elif seqlist[i] == 'V':
            seqlist[i] = 'G'
            
        elif seqlist[i] == 'D':
            seqlist[i] = 'T'
                
    return ''.join(seqlist)


def adjust_uncertain_nt(seq_list):
        
    seqs_new = []
    for i in tqdm(range(len(seq_list)), unit='seqs'):
        seqs_new.append(adjust_seq(seq_list[i]))
        
    return seqs_new