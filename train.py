import wandb
import jax
import jax.numpy as jnp
import random
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
from transformers import AutoConfig
from model import TransformerForSequenceClassification
from data_preprocessing import load_data

wandb.init(project="sentiment-analisis")

model_ckpt = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_ckpt)

rng = jax.random.PRNGKey(0)


def create_train_state(rng, learning_rate, model):
    """Create a train state."""
    params = model.init(rng, jnp.ones((1, 128)))
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def compute_loss(logits, labels):
    """Computes the loss."""
    labels_one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot).mean()


@jax.jit
def train_step(state, batch, rng):
    """Single training step."""

    def loss_fn(params, rng):
        rng, dropout_rng = jax.random.split(rng)
        logits = state.apply_fn(
            {"params": params["params"]},
            batch["input_ids"],
            train=True,
            rngs={"dropout": dropout_rng},
        )
        loss = compute_loss(logits, batch["labels"])
        return loss, logits

    grad_fn = jax.value_and_grad(lambda params: loss_fn(params, rng), has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss, logits


def compute_accuracy(logits, labels):
    """Computes accuracy based on the logits and true labels"""
    predictions = jnp.argmax(logits, -1)
    correct_predictions = jnp.sum(predictions == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy


def train_model(dataset, batch_size=32, num_epochs=3, rng=rng):
    state = create_train_state(
        rng, learning_rate=1e-3, model=TransformerForSequenceClassification(config)
    )
    for epoch in range(num_epochs):
        dataset_list = list(dataset)
        random.shuffle(dataset_list)

        for i in tqdm(
            range(0, len(dataset_list), batch_size), desc="Batches", leave=False
        ):
            batch = dataset_list[i : i + batch_size]
            batch_input_ids = jnp.stack([item["input_ids"] for item in batch])
            batch_attention_mask = jnp.stack([item["attention_mask"] for item in batch])
            batch_labels = jnp.stack([item["labels"] for item in batch])

            # Prepare batch
            jax_batch = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
            }

            # Call the training step
            state, loss, logits = train_step(state, jax_batch, rng)
            accuracy = compute_accuracy(logits, jax_batch["labels"])

            wandb.log({"loss": loss, "accuracy": accuracy})
            tqdm.write(
                f"Epoch {epoch + 1}, Step {i // batch_size + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )

        print(f"Epoch {epoch+1} completed. Loss: {loss}")


if __name__ == "__main__":
    train_dataset = load_data("data/IMDB Dataset.csv")

    try:
        train_model(train_dataset)
    finally:
        wandb.finish()
