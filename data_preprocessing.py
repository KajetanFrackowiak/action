import jax
import jax.numpy as jnp
from flax import linen as nn
import pandas as pd
from transformers import AutoTokenizer

label_mapping = {"positive": 1, "negative": 0}
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


class IMDbJAXDataset:
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_length = max_length
        self.indices = jnp.arange(len(dataframe))  # For indexing

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["review"]
        label = row["sentiment"]

        label = label_mapping[label]

        # Tokenize the input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_tensors="jax",  # Use JAX tensors here
        )

        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": jnp.array(label, dtype=jnp.int32),  # Convert label to JAX array
        }

    def shuffle(self):
        self.indices = jax.random.permutation(jax.random.PRNGKey(0), self.indices)

    def get_batch(self, batch_size):
        # Generate a batch of data based on the shuffled indices
        batch_indices = self.indices[:batch_size]
        return [self[i.item()] for i in batch_indices]


def load_data(file_path, tokenizer=tokenizer, batch_size=32):
    """Load the dataset and create batches using JAX."""
    df = pd.read_csv(file_path)
    return IMDbJAXDataset(df, tokenizer)


if __name__ == "__main__":
    train_dataset = load_data("data/IMDB Dataset.csv")
    print(f"Loaded dataset with {len(train_dataset)} samples.")

    train_dataset.shuffle()
    batch = train_dataset.get_batch(5)
    print(batch)
