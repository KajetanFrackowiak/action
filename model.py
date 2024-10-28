from transformers import AutoConfig
from flax import linen as nn
import jax.numpy as jnp
from jax.nn import softmax, gelu
from typing import Any

# Load model configurations
model_ckpt = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_ckpt)


class ScaledDotProductAttention(nn.Module):
    def __call__(self, query, key, value, mask=None):
        dim_k = query.shape[-1]
        scores = jnp.einsum("...qd,...kd->...qk", query, key) / jnp.sqrt(dim_k)
        if mask is not None:
            scores = jnp.where(mask == 0, -jnp.inf, scores)
        weights = softmax(scores, axis=-1)
        return jnp.einsum("...qk,...kd->...qd", weights, value)


class AttentionHead(nn.Module):
    embed_dim: int
    head_dim: int

    def setup(self):
        self.q = nn.Dense(self.head_dim)
        self.k = nn.Dense(self.head_dim)
        self.v = nn.Dense(self.head_dim)
        self.attention = ScaledDotProductAttention()

    def __call__(self, hidden_state):
        query = self.q(hidden_state)
        key = self.k(hidden_state)
        value = self.v(hidden_state)
        return self.attention(query, key, value)


class MultiHeadAttention(nn.Module):
    config: Any

    def setup(self):
        embed_dim = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.output_linear = nn.Dense(embed_dim)

    def __call__(self, hidden_state):
        attn_outputs = jnp.concatenate(
            [head(hidden_state) for head in self.heads], axis=-1
        )
        return self.output_linear(attn_outputs)


class FeedForward(nn.Module):
    config: Any

    def setup(self):
        self.linear_1 = nn.Dense(self.config.intermediate_size)
        self.linear_2 = nn.Dense(self.config.hidden_size)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, x, train=True):
        x = gelu(self.linear_1(x))
        x = self.dropout(self.linear_2(x), deterministic=not train)
        return x


class TransformerEncoderLayer(nn.Module):
    config: Any

    def setup(self):
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.attention = MultiHeadAttention(self.config)
        self.feed_forward = FeedForward(self.config)

    def __call__(self, x, train=True):
        attn_output = self.attention(self.layer_norm_1(x))
        x = x + attn_output
        x = x + self.feed_forward(self.layer_norm_2(x), train=train)
        return x


class Embeddings(nn.Module):
    config: Any

    def setup(self):
        self.token_embeddings = nn.Embed(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, train=True):
        seq_length = input_ids.shape[1]
        position_ids = jnp.arange(seq_length)[None, :]
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=not train)
        return embeddings


class TransformerEncoder(nn.Module):
    config: Any

    def setup(self):
        self.embeddings = Embeddings(self.config)
        self.layers = [
            TransformerEncoderLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        ]

    def __call__(self, x, train=True):
        x = self.embeddings(x, train=train)
        for layer in self.layers:
            x = layer(x, train=train)
        return x


class TransformerForSequenceClassification(nn.Module):
    config: Any

    def setup(self):
        self.encoder = TransformerEncoder(self.config)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels)

    def __call__(self, x, train=True):
        x = x.astype(jnp.int32)
        x = self.encoder(x, train=train)[:, 0, :]
        x = self.dropout(x, deterministic=not train)
        return self.classifier(x)
