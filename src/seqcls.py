import torch
from torch import nn
from bertconfig import BertConfig
from semantic_encoder import SemanticEncoder
from alignment_encoder import AlignmentEncoder, AlnConfig


class SeqCls(nn.Module):
    
    def __init__(self, config):
        super(SeqCls, self).__init__()
        
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
        
        self.aln_vocab_size = 4**config.k+1
        self.aln_maxnum_token = config.maxseqlen-config.k+1
        self.alnconfig = AlnConfig(
            vocab_size=self.aln_vocab_size,
            embed_size=config.alnenc_embed_size,
            dropout_rate=config.dropout_rate,
            hidden_size=config.alnenc_hidden_size,
            num_lstm_layers=config.alnenc_num_lstm_layers,
            bidirectional=config.alnenc_lstm_bidirectional)
        self.aln_encoder = AlignmentEncoder(self.alnconfig)
        
        self.concat_features = config.bert_hidden_size + config.alnenc_hidden_size
            
        self.classifier = nn.Sequential()
        for i in range(len(config.fc_units)):
            if (i+1) == 1:
                self.classifier.add_module('linear'+str(i+1), nn.Linear(
                    2*self.concat_features, config.fc_units[i]))
            else:
                self.classifier.add_module('linear'+str(i+1), nn.Linear(
                    config.fc_units[i-1], config.fc_units[i]))
                
            self.classifier.add_module(
                'LayerNorm'+str(i+1), nn.LayerNorm(config.fc_units[i]))
            self.classifier.add_module('relu', nn.ReLU())
            self.classifier.add_module(
                'dropout', nn.Dropout(config.dropout_rate))
            
        self.classifier.add_module(
            'logits', nn.Linear(config.fc_units[-1], 1))
        self.classifier.add_module('sigmoid', nn.Sigmoid())
        
        
    def forward(self, input_fw_semantic, input_rc_semantic,
                input_fw_aln, input_rc_aln):
        
        batchsz = input_fw_semantic.size(0)
        input_fw_semantic = input_fw_semantic.view(self.config.split*batchsz,
                                                   self.bert_maxnum_token)
        input_rc_semantic = input_rc_semantic.view(self.config.split*batchsz,
                                                   self.bert_maxnum_token)
        bertembed_fw, bertembed_rc = self.semantic_encoder(
            input_fw_semantic, input_rc_semantic)
        
        input_fw_aln = input_fw_aln.view(self.config.split*batchsz,
                                           self.aln_maxnum_token)
        input_rc_aln = input_rc_aln.view(self.config.split*batchsz,
                                           self.aln_maxnum_token)
        alnembed_fw, alnembed_rc = self.aln_encoder(
            input_fw_aln, input_rc_aln)
        
        embed_fw = torch.cat([bertembed_fw, alnembed_fw], dim=1)
        embed_fw = embed_fw.view(batchsz, self.config.split*self.concat_features)
        
        embed_rc = torch.cat([bertembed_rc, alnembed_rc], dim=1)
        embed_rc = embed_rc.view(batchsz, self.config.split*self.concat_features)
        
        probs_fw = self.classifier(embed_fw)
        probs_rc = self.classifier(embed_rc)
        probs = (probs_fw+probs_rc) / 2
        
        return probs
    
    
class SeqclsConfig():
    
    def __init__(self, k, maxseqlen, split, bert_hidden_size,
                 bert_num_encoders, bert_num_attention_heads,
                 bert_forward_size, alnenc_embed_size,
                 alnenc_hidden_size, alnenc_num_lstm_layers,
                 alnenc_lstm_bidirectional, fc_units, dropout_rate=0.1):
        
        self.k = k
        self.maxseqlen = maxseqlen
        self.split = split
        self.bert_hidden_size = bert_hidden_size
        self.bert_num_encoders = bert_num_encoders
        self.bert_num_attention_heads = bert_num_attention_heads
        self.bert_forward_size = bert_forward_size
        self.alnenc_embed_size = alnenc_embed_size
        self.alnenc_hidden_size = alnenc_hidden_size
        self.alnenc_num_lstm_layers = alnenc_num_lstm_layers
        self.alnenc_lstm_bidirectional = alnenc_lstm_bidirectional
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate