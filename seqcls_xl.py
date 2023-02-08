import torch
from torch import nn
from transformers import BertConfig
from SemanticEncoder2 import SemanticEncoder
from SeqcompEncoder_BiLSTM import SeqcompositionEncoder, SeqcompConfig
import numpy as np


class SeqCls_XL_Concat(nn.Module):
    
    def __init__(self, config):
        super(SeqCls_XL_Concat, self).__init__()
        
        self.config = config
        
        self.bert_vocab_size = 4**config.k+3
        self.bert_maxnum_token = config.maxseqlen-config.k+2
        self.bertconfig = BertConfig(
            vocab_size=self.bert_vocab_size,
            hidden_size=config.bert_hidden_size,
            num_hidden_layers=config.bert_num_encoders,
            num_attention_heads=config.bert_num_attention_heads,
            intermediate_size=config.bert_forward_size,
            max_position_embeddings=config.maxseqlen)
        self.semantic_encoder = SemanticEncoder(self.bertconfig)
        
        self.skipgram_vocab_size = 4**config.k+1
        self.seqcomp_maxnum_token = config.maxseqlen-config.k+1
        self.seqcompconfig = SeqcompConfig(
            vocab_size=self.skipgram_vocab_size,
            embed_size=config.skipgram_embed_size,
            embed_dropout=config.skipgram_embed_dropout,
            hidden_size=config.covpredictor_hidden_size,
            num_lstm_layers=config.covpredictor_num_lstm_layers,
            lstm_dropout=config.covpredictor_lstm_dropout,
            bidirectional=config.covpredictor_lstm_bidirectional,
            pooler_dropout=config.covpredictor_pooler_dropout)
        self.seqcomp_encoder = SeqcompositionEncoder(self.seqcompconfig)
        
        self.concat_features = config.bert_hidden_size + \
            config.covpredictor_hidden_size
            
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            'linear', nn.Linear(2*self.concat_features, config.linear_units))
        self.classifier.add_module(
            'layernorm', nn.LayerNorm(config.linear_units))
        self.classifier.add_module('relu', nn.ReLU())
        self.classifier.add_module('dropout', nn.Dropout(config.linear_dropout))
        self.classifier.add_module(
            'logits', nn.Linear(config.linear_units, 1))
        self.classifier.add_module('sigmoid', nn.Sigmoid())
        
        
    def forward(self, input_fw_semantic, input_rc_semantic,
                input_fw_seqcomp, input_rc_seqcomp):
        
        """
        input_xx: (batch, seqlen)
        """
        batchsz = input_fw_semantic.size(0)
        input_fw_semantic = input_fw_semantic.view(self.config.split*batchsz,
                                                   self.bert_maxnum_token)
        input_rc_semantic = input_rc_semantic.view(self.config.split*batchsz,
                                                   self.bert_maxnum_token)
        bertembed_fw, bertembed_rc = self.semantic_encoder(
            input_fw_semantic, input_rc_semantic)
        
        input_fw_seqcomp = input_fw_seqcomp.view(self.config.split*batchsz,
                                                 self.seqcomp_maxnum_token)
        input_rc_seqcomp = input_rc_seqcomp.view(self.config.split*batchsz,
                                                 self.seqcomp_maxnum_token)
        seqembed_fw, seqembed_rc = self.seqcomp_encoder(
            input_fw_seqcomp, input_rc_seqcomp)
        
        ##### [batch*config.split, concat_features]
        embed_fw = torch.cat([bertembed_fw, seqembed_fw], dim=1)
        ##### [batch, config.split*concat_features]
        embed_fw = embed_fw.view(batchsz, self.config.split*self.concat_features)
        
        embed_rc = torch.cat([bertembed_rc, seqembed_rc], dim=1)
        embed_rc = embed_rc.view(batchsz, self.config.split*self.concat_features)
        
        probs_fw = self.classifier(embed_fw)
        probs_rc = self.classifier(embed_rc)
        probs = (probs_fw+probs_rc) / 2
        
        return probs
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
class Seqclsconfig_XL_Concat():
    
    def __init__(self, k, maxseqlen, split, bert_hidden_size,
                 bert_num_encoders, bert_num_attention_heads,
                 bert_forward_size, skipgram_embed_size,
                 skipgram_embed_dropout, covpredictor_hidden_size,
                 covpredictor_num_lstm_layers, covpredictor_lstm_dropout,
                 covpredictor_lstm_bidirectional, covpredictor_pooler_dropout,
                 linear_units, linear_dropout):
        
        self.k = k
        self.maxseqlen = maxseqlen
        self.split = split
        self.bert_hidden_size = bert_hidden_size
        self.bert_num_encoders = bert_num_encoders
        self.bert_num_attention_heads = bert_num_attention_heads
        self.bert_forward_size = bert_forward_size
        self.skipgram_embed_size = skipgram_embed_size
        self.skipgram_embed_dropout = skipgram_embed_dropout
        self.covpredictor_hidden_size = covpredictor_hidden_size
        self.covpredictor_num_lstm_layers = covpredictor_num_lstm_layers
        self.covpredictor_lstm_dropout = covpredictor_lstm_dropout
        self.covpredictor_lstm_bidirectional = covpredictor_lstm_bidirectional
        self.covpredictor_pooler_dropout = covpredictor_pooler_dropout
        self.linear_units = linear_units
        self.linear_dropout = linear_dropout
        
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
    
    names = []
    for n, p in model.named_parameters():
        names.append(n)
    
    outpath = 'SeqCls_XL_BiLSTM_param-names.npy'
    np.save(outpath, names)