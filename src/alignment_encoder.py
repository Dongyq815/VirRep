from torch import nn


class AlignmentEncoder(nn.Module):
    
    def __init__(self, config):
        super(AlignmentEncoder, self).__init__()
        
        self.config = config
        direction = 2 if config.bidirectional else 1
        
        self.embedding = nn.Sequential()
        self.embedding.add_module(
            'kmer_embedding',
            nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        )
        self.embedding.add_module('ln', nn.LayerNorm(config.embed_size))
        self.embedding.add_module('dropout', nn.Dropout(config.dropout_rate))
        
        self.encoder = nn.LSTM(input_size=config.embed_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_lstm_layers,
                               batch_first=True,
                               dropout=config.dropout_rate,
                               bidirectional=config.bidirectional)
        
        self.ln_encoder = nn.Sequential()
        self.ln_encoder.add_module('ln', nn.LayerNorm(
            direction*config.hidden_size))
        self.ln_encoder.add_module('dropout', nn.Dropout(config.dropout_rate))
        
        self.pooler = nn.Sequential()
        self.pooler.add_module(
            'linear',
            nn.Linear(direction*config.hidden_size, config.hidden_size))
        self.pooler.add_module('ln', nn.LayerNorm(config.hidden_size))
        self.pooler.add_module('Tanh', nn.Tanh())
        self.pooler.add_module('dropout', nn.Dropout(config.dropout_rate))
        
        
    def forward(self, input_fw, input_rc):
        # [batch, seqlen, embed_dim]
        embed_fw = self.embedding(input_fw)
        embed_rc = self.embedding(input_rc)
        
        # out_xx: [batch, seqlen, direction*hidden_size]
        # hn_xx/cn_xx: [direction*num_lstm, batch, hidden_size]
        out_fw, (hn_fw, cn_fw) = self.encoder(embed_fw)
        out_rc, (hn_rc, cn_rc) = self.encoder(embed_rc)
        
        # [batch, direction*hidden_size]
        seqembed_fw = out_fw.mean(dim=1)
        seqembed_rc = out_rc.mean(dim=1)
        
        seqembed_fw = self.ln_encoder(seqembed_fw)
        seqembed_rc = self.ln_encoder(seqembed_rc)
        
        pooled_output_fw = self.pooler(seqembed_fw)
        pooled_output_rc = self.pooler(seqembed_rc)
        
        return pooled_output_fw, pooled_output_rc
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class AlnConfig():
    
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_lstm_layers, bidirectional, dropout_rate):
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional