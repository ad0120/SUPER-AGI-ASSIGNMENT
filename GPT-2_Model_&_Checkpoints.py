import torch
import torch.nn as nn

# Define the GPT-2 model class
class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, max_sequence_len=1024):
        super(GPT2, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_embeddings = nn.Embedding(max_sequence_len, d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # Token embeddings
        token_embedded = self.token_embeddings(input_ids)
        
        # Positional encoding
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        positional_embedded = self.positional_embeddings(positions)

        # Sum token and positional embeddings
        embeddings = token_embedded + positional_embedded

        # Transformer Encoder
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)

        # Generate logits for output predictions
        logits = self.decoder(embeddings)

        return logits

# Define the Encoder layer with multi-head self-attention and feed-forward network
class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=12):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-head self-attention
        attention_output = self.multi_head_attention(x, x, x)
        attention_output = self.layer_norm1(x + attention_output)

        # Feed-forward network
        feed_forward_output = self.feed_forward(attention_output)
        output = self.layer_norm2(attention_output + feed_forward_output)

        return output

# Define the Multi-head Self-Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        # Define linear transformations for query, key, and value
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear transformations for query, key, and value
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Reshape to split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return context
