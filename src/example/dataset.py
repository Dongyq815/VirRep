import torch
from torch.utils.data import Dataset
from Bio.Seq import Seq
from functools import reduce
from copy import deepcopy
import numpy as np
import pandas as pd
import math
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import readfa


class BertPretrainData(Dataset):
    
    def __init__(self, fapath, k, maxseqlen, mask_prob=0.15):
        
        self.k = k
        self.maxseqlen = maxseqlen
        self.vocab_size = 4**k+3
        self.id2token = ['[UNK]', '[CLS]', '[MASK]']
        self.id2token.extend(reduce(lambda x,y: [i+j for i in x for j in y],
                                    [['A', 'C', 'G', 'T']]*k))
        self.token2id = {token:idx for idx, token in enumerate(self.id2token)}
        self.seqs, _ = readfa(fapath)
        
        if k % 2 == 0:
            left_size = (k-1) // 2
            right_size = (k-1) // 2 + 1
        else:
            left_size = int((k-1) / 2)
            right_size = left_size
        self.mask_range = list(range(-left_size, 0))
        self.mask_range.extend(list(range(1, right_size+1)))
        self.mask_prob = mask_prob

    def seqencode(self, seq):
        seq = seq.upper()
        tokens, pos = [1], 0
        
        while (pos+self.k) <= len(seq):
            tokens.append(self.token2id.get(seq[pos:pos+self.k], 0))
            pos += 1
        
        return torch.IntTensor(tokens)
    
    def __getitem__(self, index):
        seq_tokens = self.seqencode(self.seqs[index])
        inputs, labels = self.mask_token(seq_tokens)
        
        return inputs, labels.to(torch.int64)
    
    def mask_token(self, seq_tokens):
        labels = seq_tokens.clone()
        
        prob_mat = torch.full(labels.shape, self.mask_prob / self.k)
        prob_mat[0] = 0
        
        masks = torch.bernoulli(prob_mat).bool()
        masked_centers = set(torch.where(masks == 1)[0].tolist())
        masked_indices = deepcopy(masked_centers)
        end = labels.size()[0]-1
        
        for center in masked_centers:
            for mask_number in self.mask_range:
                index = center+mask_number
                if index <= end and index >= 1:
                    masked_indices.add(index)
        masked_indices = list(masked_indices)
        masks[masked_indices] = True
        
        labels[~masks] = -100
        
        idx_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masks
        seq_tokens[idx_replaced] = self.token2id['[MASK]']
        
        idx_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() \
            & masks & ~idx_replaced
        random_words = torch.randint(self.vocab_size, labels.shape,
                                     dtype=torch.int32)
        seq_tokens[idx_random] = random_words[idx_random]
        
        return seq_tokens, labels
    
    def __len__(self):
        return len(self.seqs)
    

class SeqClsData(Dataset):
    
    def __init__(self, k, virpath, hostpath):
        
        self.k = k
        self.virseqs_fw, _ = readfa(virpath)
        self.hostseqs_fw, _ = readfa(hostpath)
        
        self.id2token = ['[UNK]', '[CLS]', '[MASK]']
        self.id2token.extend(reduce(lambda x,y: [i+j for i in x for j in y],
                             [['A', 'C', 'G', 'T']]*k))
        self.token2id = {token:idx for idx, token in enumerate(self.id2token)}
        
    def __getitem__(self, index):
        
        if index < len(self.virseqs_fw):
            seq_fw = self.virseqs_fw[index]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            seqcode_fw = self.encoder(seq_fw)
            seqcode_rc = self.encoder(seq_rc)
            label = torch.tensor([1], dtype=torch.float32)
        else:
            idx_new = index-len(self.virseqs_fw)
            seq_fw = self.hostseqs_fw[idx_new]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            seqcode_fw = self.encoder(seq_fw)
            seqcode_rc = self.encoder(seq_rc)
            label = torch.tensor([0], dtype=torch.float32)
        
        return seqcode_fw, seqcode_rc, label
    
    def encoder(self, seq):
        seq = seq.upper()
        seqcode, pos = [1], 0
        
        while (pos+self.k) <= len(seq):
            token = seq[pos:pos+self.k]
            seqcode.append(self.token2id.get(token, 0))
            pos += 1
        
        return torch.IntTensor(seqcode)
    
    def __len__(self):
        return len(self.virseqs_fw)+len(self.hostseqs_fw)
    
    
class KmerEmbeddingData(Dataset):
    
    def __init__(self, fapath, k, windowsz, neg_num):
        
        self.k = k
        self.kmer_size = 4**k
        self.window_size = windowsz
        self.neg_num = neg_num
        
        self.id2token = reduce(lambda x,y: [i+j for i in x for j in y],
                               [['A', 'C', 'G', 'T']]*self.k)
        self.token2id = {kmer:idx for idx, kmer in enumerate(self.id2token)}
        
        print('building dataset for training...')
        self.build_dataset(fapath)
        self.kmer_freqs = self.compute_kmer_freq()
        
    def build_dataset(self, fapath):
        
        self.vocab = {}
        for kmer in self.id2token:
            self.vocab[kmer] = 0
            
        seqs, headers = readfa(fapath)
        self.text_encoded = []
        count = 0
        
        for seq in seqs:
            count += 1
            seq = seq.upper()
            
            for pos in range(len(seq)-self.k+1):
                kmer = seq[pos:pos+self.k]
                flag = self.check_kmer(kmer, ['A', 'C', 'G', 'T'])
                
                if flag:
                    self.text_encoded.append(self.token2id[kmer])
                    if pos == (len(seq)-self.k):
                        self.text_encoded.append(self.kmer_size)
                    self.vocab[kmer] += 1
                else:
                    self.text_encoded.append(self.kmer_size)
                    
    def check_kmer(self, kmer, letter_set):
        flag = True
        
        for letter in kmer:
            if letter not in letter_set:
                flag = False
                break
            
        return flag
    
    def compute_kmer_freq(self):
            
        kmer_counts = np.array([count for count in self.vocab.values()])
        kmer_freqs = kmer_counts / np.sum(kmer_counts)
        kmer_freqs = kmer_freqs ** (3./4.)
        kmer_freqs = kmer_freqs / np.sum(kmer_freqs)
        
        return kmer_freqs
    
    def __getitem__(self, index):
        if self.text_encoded[index] == self.kmer_size:
            return []
        
        center_word = self.text_encoded[index]
        pos_word = self.get_positive_word(index)
        neg_word = self.get_negative_word(pos_word)
        
        word_pair = []
        for target in pos_word:
            word_pair.append((center_word, target, 1))
            
        for target in neg_word:
            word_pair.append((center_word, target, 0))
        
        return word_pair
    
    def get_positive_word(self, center_index):
        
        left_idx = []
        for i in range(1, self.window_size+1):
            curr_index = center_index-i
            
            if curr_index < 0:
                break
            if self.text_encoded[curr_index] == self.kmer_size:
                break
            left_idx.append(curr_index)
            
        right_idx = []
        for i in range(1, self.window_size+1):
            curr_index = center_index+i
            
            if curr_index >= len(self.text_encoded):
                break
            if self.text_encoded[curr_index] == self.kmer_size:
                break
            right_idx.append(curr_index)
        
        pos_idx = left_idx+right_idx
        return [self.text_encoded[idx] for idx in pos_idx]
    
    def get_negative_word(self, positive_word):
        neg_candidate = np.random.choice(range(self.kmer_size),
                                         self.neg_num*len(positive_word),
                                         replace=False,
                                         p=self.kmer_freqs)
        
        neg_word = []
        for word in neg_candidate:
            if word not in positive_word:
                neg_word.append(word)
        
        return neg_word
    
    def __len__(self):
        return len(self.text_encoded)
    

def batch_stack(word_pair_list):
    center_word = []
    target_word = []
    label = []
    
    for word_pair in word_pair_list:
        for tupple in word_pair:
            center_word.append(tupple[0])
            target_word.append(tupple[1])
            label.append(tupple[2])
    
    center_word = torch.IntTensor(center_word).reshape((-1, 1))
    target_word = torch.IntTensor(target_word).reshape((-1, 1))
    label = torch.tensor(label, dtype=torch.float32).reshape((-1, 1))
    
    return center_word, target_word, label


class SimPredData(Dataset):
    
    def __init__(self, k, virpath, virpath_response,
                 hostpath, hostpath_response):
        
        self.virseq_fw, _ = readfa(virpath)
        self.virus_response = pd.read_table(virpath_response)
        
        self.hostseq_fw, _ = readfa(hostpath)
        self.host_response = pd.read_table(hostpath_response)
        
        self.idx2kmer = ['[UNK]']
        self.idx2kmer.extend(reduce(
            lambda x,y: [i+j for i in x for j in y], [['A', 'C', 'G', 'T']]*self.k))
        self.kmer2idx = {kmer:idx for idx, kmer in enumerate(self.idx2kmer)}
        
    def __getitem__(self, index):
        
        if index < len(self.virseq_fw):
            seq_fw = self.virseq_fw[index]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            
            seqcode_fw = self.encoder(seq_fw, '+')
            seqcode_rc = self.encoder(seq_rc, '-')
            response = self.virus_response.iloc[index, 1]
            
        else:
            index = index-len(self.virseq_fw)
            seq_fw = self.hostseq_fw[index]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            
            seqcode_fw = self.encoder(seq_fw, '+')
            seqcode_rc = self.encoder(seq_rc, '-')
            response = self.host_response.iloc[index, 1]
        
        seqcode_fw = torch.IntTensor(seqcode_fw)
        seqcode_rc = torch.IntTensor(seqcode_rc)
        response = torch.tensor([response], dtype=torch.float32)
        
        return seqcode_fw, seqcode_rc, response
    
    def encoder(self, seq, strand):
        
        seq = seq.upper()
        letter_set = ['A', 'C', 'G', 'T']
        seqcode, pos = [], 0
        
        while (pos+7) <= len(seq):
            kmer = seq[pos: pos+7]
            ambiguous = 0
            
            for base in kmer:
               if base not in letter_set:
                   ambiguous += 1
            if ambiguous >= 3:
                seqcode.append(0)
            elif ambiguous > 0 and ambiguous < 3:
                kmer = self.kmercor(kmer, strand)
                seqcode.append(self.kmer2idx.get(kmer, 0))
            elif ambiguous == 0:
                seqcode.append(self.kmer2idx[kmer])
            
            pos += 1
        
        return seqcode
    
    def kmercor(self, kmer, strand):
        
        kmer = kmer.upper()
        kmerlist = list(kmer)
        substitution_fw = {'R': 'A', 'Y': 'C', 'M': 'A', 'K': 'G', 'S': 'C',
                           'W': 'T', 'H': 'A', 'B': 'C', 'V': 'G', 'D': 'T'}
        substitution_rc = {'R': 'T', 'Y': 'G', 'M': 'T', 'K': 'C', 'S': 'G',
                           'W': 'A', 'H': 'T', 'B': 'G', 'V': 'C', 'D': 'A'}
        
        if strand == '+':
            for i in range(len(kmerlist)):
                kmerlist[i] = substitution_fw[kmerlist[i]]
        
        elif strand == '-':
            for i in range(len(kmerlist)):
                kmerlist[i] = substitution_rc[kmerlist[i]]
                    
        return ''.join(kmerlist)
    
    def __len__(self):
        return len(self.virseq_fw)+len(self.hostseq_fw)
    
    
class SeqData(Dataset):
    
    def __init__(self, k, max_seqlen, virpath, hostpath):
        
        self.k = k
        self.max_seqlen = max_seqlen
        self.kmerset = reduce(lambda x,y: [i+j for i in x for j in y],
                              [['A', 'C', 'G', 'T']]*k)
        
        self.id2token_semantic = ['[UNK]', '[CLS]', '[MASK]']
        self.id2token_semantic.extend(self.kmerset)
        self.token2id_semantic = {
            token:idx for idx, token in enumerate(self.id2token_semantic)
            }
        
        self.id2token_seqcomp = ['[UNK]']
        self.id2token_seqcomp.extend(self.kmerset)
        self.token2id_seqcomp = {
            token:idx for idx, token in enumerate(self.id2token_seqcomp)
            }
        
        self.virseq_fw, _ = readfa(virpath)
        self.hostseq_fw, _ = readfa(hostpath)
        
    def __getitem__(self, index):
        
        if index < len(self.virseq_fw):
            seq_fw = self.virseq_fw[index]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            label = torch.tensor([1], dtype=torch.float32)    
        else:
            idx_new = index-len(self.virseq_fw)
            seq_fw = self.hostseq_fw[idx_new]
            seq_rc = str(Seq(seq_fw).reverse_complement())
            label = torch.tensor([0], dtype=torch.float32)
        
        seqcode_fw_semantic = self.encoder_semantic(seq_fw)
        seqcode_rc_semantic = self.encoder_semantic(seq_rc)
        
        seqcode_fw_seqcomp = self.encoder_seqcomp(seq_fw, '+')
        seqcode_rc_seqcomp = self.encoder_seqcomp(seq_rc, '-')
        
        return seqcode_fw_semantic, seqcode_rc_semantic, \
               seqcode_fw_seqcomp, seqcode_rc_seqcomp, label
               
    def encoder_semantic(self, seq):
        
        seq = seq.upper()
        seqcode = []
        nfrag = math.ceil(len(seq) / self.max_seqlen)
        
        for i in range(nfrag):
            if (i+1) == nfrag:
                start = len(seq)-self.max_seqlen
                end = len(seq)
            else:
                start = i*self.max_seqlen
                end = start+self.max_seqlen
                
            seqfrag = seq[start:end]
            seqcode.extend(self.encoder_semantic_frag(seqfrag))
            
        return torch.IntTensor(seqcode)
    
    def encoder_semantic_frag(self, seqfrag):
        
        fragcode, pos = [1], 0
        
        while (pos+self.k) <= len(seqfrag):
            token = seqfrag[pos:pos+self.k]
            fragcode.append(self.token2id_semantic.get(token, 0))
            pos += 1
        
        return fragcode
    
    def encoder_seqcomp(self, seq, strand):
        
        seq = seq.upper()
        seqcode = []
        nfrag = math.ceil(len(seq) / self.max_seqlen)
        
        for i in range(nfrag):
            if (i+1) == nfrag:
                start = len(seq)-self.max_seqlen
                end = len(seq)
            else:
                start = i*self.max_seqlen
                end = start+self.max_seqlen
                
            seqfrag = seq[start:end]
            seqcode.extend(self.encoder_seqcomp_frag(seqfrag, strand))
        
        return torch.IntTensor(seqcode)
    
    def encoder_seqcomp_frag(self, seqfrag, strand):
        
        letter_set = ['A', 'C', 'G', 'T']
        fragcode, pos = [], 0
        
        while (pos+self.k) <= len(seqfrag):
            token = seqfrag[pos:pos+self.k]
            ambiguous = 0
            
            for base in token:
                if base not in letter_set:
                    ambiguous += 1
                
            if ambiguous == 0:
                fragcode.append(self.token2id_seqcomp[token])
            elif ambiguous > 0 and ambiguous < 3:
                token = self.kmercor(token, strand)
                fragcode.append(self.token2id_seqcomp.get(token, 0))
            elif ambiguous >= 3:
                fragcode.append(0)
                    
            pos += 1
                
        return fragcode
    
    def kmercor(self, kmer, strand):
        
        kmer = kmer.upper()
        kmerlist = list(kmer)
        substitution_fw = {'R': 'A', 'Y': 'C', 'M': 'A', 'K': 'G', 'S': 'C',
                           'W': 'T', 'H': 'A', 'B': 'C', 'V': 'G', 'D': 'T'}
        substitution_rc = {'R': 'T', 'Y': 'G', 'M': 'T', 'K': 'C', 'S': 'G',
                           'W': 'A', 'H': 'T', 'B': 'G', 'V': 'C', 'D': 'A'}
        
        if strand == '+':
            for i in range(len(kmerlist)):
                kmerlist[i] = substitution_fw[kmerlist[i]]
        
        elif strand == '-':
            for i in range(len(kmerlist)):
                kmerlist[i] = substitution_rc[kmerlist[i]]
                    
        return ''.join(kmerlist)
    
    def __len__(self):
        return len(self.virseq_fw)+len(self.hostseq_fw)