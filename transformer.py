import torch
import torch.nn as nn
import math


# Part 1 - Encoder

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.w_q(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        x, attn = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.w_o(x), attn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(0.3), ###
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        dropout_rate = 0.3 ###
        self.dropout = nn.Dropout(p=dropout_rate)  ###
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attn(x, mask)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head) for _ in range(n_layer)])
        self.scale = math.sqrt(d_model) ###
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        #x = self.token_embedding(x) + self.position_embedding(pos)
        x = self.token_embedding(x) * self.scale + self.position_embedding(pos) ###
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x)
            attentions.append(attn)
        
        return x, attentions
    
    def get_pooled_output(self, x):
        return x.mean(dim=1)

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        encoded, attentions = self.encoder(x)
        pooled = self.encoder.get_pooled_output(encoded)
        return self.classifier(pooled), attentions
    
    
# Part 2 - Decoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff, max_seq_len, return_attentions=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff) for _ in range(n_layer)])
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.return_attentions = return_attentions
        self.vocab_size = vocab_size
        
    def forward(self, x, targets=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(pos)
        
        # Mask for future tokens (batch_size, 1, seq_len, seq_len)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)
        
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        
        output = self.final_layer(x)
        
        if targets is not None:
            # Flatten the output and targets to calculate CrossEntropyLoss
            output = output.view(-1, output.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            targets = targets.view(-1)  # Shape: (batch_size * seq_len)
            loss = self.loss_fn(output, targets)  # Compute the cross-entropy loss
            return output, loss
        
        return output  
        
        if self.return_attentions:
            return output, attentions 
        else:
            return output