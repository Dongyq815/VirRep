import torch
from torch.utils.data import Dataset
from Bio.Seq import Seq
from functools import reduce
from tqdm import tqdm
import math


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


def writefa(headers, seqs, outpath):
    
    outfile = open(outpath, 'w')
    for i in range(len(headers)):
        outfile.write('>'+headers[i]+'\n')
        outfile.write(seqs[i]+'\n')
        
    outfile.close()


def adjust_seq(seq):
    
    seq = seq.upper()
    seq = seq.replace('R', 'A')
    seq = seq.replace('Y', 'T')
    seq = seq.replace('M', 'C')
    seq = seq.replace('K', 'G')
    seq = seq.replace('S', 'C')
    seq = seq.replace('W', 'A')
    seq = seq.replace('H', 'A')
    seq = seq.replace('B', 'C')
    seq = seq.replace('V', 'G')
    seq = seq.replace('D', 'T')
    
    return seq


def adjust_uncertain_nt(seq_list):
        
    seqs_new = []
    for i in range(len(seq_list)):
        seqs_new.append(adjust_seq(seq_list[i]))
        
    return seqs_new


def segment_seq(seq):
    
    letter_set = ['A', 'C', 'G', 'T']
    nseg = math.ceil(len(seq) / 1000)
    segments, positions = [], []
    
    for i in range(nseg):
        
        if (i+1) < nseg:
            start = i*1000
        elif (i+1) == nseg:
            start = len(seq)-1000
        end = start+1000
        segment = seq[start:end]
        
        bad1, bad2 = 0, 0
        for pos in range(len(segment)):
            if segment[pos] not in letter_set:
                if pos < 500:
                    bad1 += 1
                else:
                    bad2 += 1
        
        if bad1 <= 50 and bad2 <= 50:
            segments.append(segment)
            positions.append(i)
    
    return segments, positions


def segmentation(seqs, min_seqlen):
    
    seqs = adjust_uncertain_nt(seqs)
    letter_set = ['A', 'C', 'G', 'T']
    segment_list, position_list, idx_list = [], [], []
    
    for i in tqdm(range(len(seqs)), unit='seqs', total=len(seqs)):
        
        if len(seqs[i]) < min_seqlen:
            continue
        
        bad = 0
        for base in seqs[i]:
            if base not in letter_set:
                bad += 1
        if bad >= 0.2*len(seqs[i]):
            continue
        
        segments, positions = segment_seq(seqs[i])
        segment_list.extend(segments)
        position_list.extend(positions)
        idx_list.extend([i for _ in range(len(segments))])
        
    return segment_list, position_list, idx_list


def rev_seq(seq):
    return str(Seq(seq).reverse_complement())


class SeqData(Dataset):
    
    def __init__(self, seqs, min_seqlen):
        
        self.kmerset = reduce(lambda x,y: [i+j for i in x for j in y],
                              [['A', 'C', 'G', 'T']]*7)
        
        self.id2token_semantic = ['[UNK]', '[CLS]', '[MASK]']
        self.id2token_semantic.extend(self.kmerset)
        self.token2id_semantic = {
            token:idx for idx, token in enumerate(self.id2token_semantic)
        }
        
        self.id2token_aln = ['[UNK]']
        self.id2token_aln.extend(self.kmerset)
        self.token2id_aln = {
            token:idx for idx, token in enumerate(self.id2token_aln)
        }
        
        print('Segment seqs...')
        self.segment_list, self.position_list, self.idx_list = segmentation(seqs, min_seqlen)
        assert len(self.segment_list) == len(self.position_list)
        assert len(self.segment_list) == len(self.idx_list)
            
    def __getitem__(self, index):
        
        segment_fw = self.segment_list[index]
        segment_rc = rev_seq(segment_fw)
        
        input_fw_semantic = self.encode_semantic(segment_fw)
        input_rc_semantic = self.encode_semantic(segment_rc)
        
        input_fw_aln = self.encode_aln(segment_fw)
        input_rc_aln = self.encode_aln(segment_rc)
        
        return input_fw_semantic, input_rc_semantic, \
               input_fw_aln, input_rc_aln, \
               torch.IntTensor([self.position_list[index]]), \
               torch.IntTensor([self.idx_list[index]])
    
    def encode_semantic(self, segment):
        
        input_ids = self.encode_semantic_piece(segment[0:500]) + \
                    self.encode_semantic_piece(segment[500:1000])
        return torch.IntTensor(input_ids)
    
    def encode_semantic_piece(self, piece):
        
        piece_code = [1]
        pos = 0
        
        while (pos+7) <= len(piece):
            token = piece[pos:(pos+7)]
            piece_code.append(self.token2id_semantic.get(token, 0))
            pos += 1
        
        return piece_code
    
    def encode_aln(self, segment):
        
        input_ids = self.encode_aln_piece(segment[0:500]) + \
                    self.encode_aln_piece(segment[500:1000])
        return torch.IntTensor(input_ids)
    
    def encode_aln_piece(self, piece):
        
        piece_code = []
        pos = 0
        
        while (pos+7) <= len(piece):
            token = piece[pos:pos+7]
            piece_code.append(self.token2id_aln.get(token, 0))
            pos += 1
        
        return piece_code
    
    def __len__(self):
        return len(self.segment_list)