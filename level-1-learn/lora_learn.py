"""LoRA from scratch, NumPy only. Level 1, Learn, finetune-lab (P10).

A frozen base linear layer plus a low rank adapter. Only the adapter
parameters A and B train. The base weights never change.

Task, defined exactly:
- Inputs are synthetic vectors in R^16 drawn from a standard normal
  (data seed 7): 512 train examples and 128 test examples.
- The label is the argmax of x @ W_target, with 4 classes.
- W_target = W0 + scaling * (A_true @ B_true), where W0 is the frozen
  base (seed 3, scale 0.5), A_true has shape (16, 2), B_true has shape
  (2, 4), the true adapter factors use seed 9 and scale 0.6, and
  scaling = alpha / rank = 2.0.
- Because the true change from the base is rank 2 by construction, a
  rank 2 adapter can in principle express it. That is a teaching
  construction, not a claim about real tasks, where the needed change
  is not known to be low rank.
"""
import hashlib

import numpy as np

IN_DIM = 16
NUM_CLASSES = 4
RANK = 2
ALPHA = 4.0
SCALING = ALPHA / RANK
N_TRAIN = 512
N_TEST = 128
DATA_SEED = 7
BASE_SEED = 3
TARGET_SEED = 9
ADAPTER_SEED = 5
LEARNING_RATE = 0.5
EPOCHS = 200


def lora_parameter_count(in_dim=IN_DIM, num_classes=NUM_CLASSES, rank=RANK):
    """Trainable adapter parameters: A is (in_dim, rank), B is (rank, classes)."""
    return in_dim * rank + rank * num_classes


def full_parameter_count(in_dim=IN_DIM, num_classes=NUM_CLASSES):
    """Parameters in the full weight matrix of the same layer."""
    return in_dim * num_classes


def make_task():
    """Build the synthetic task and the frozen base, deterministically."""
    rng_data = np.random.default_rng(DATA_SEED)
    x_train = rng_data.standard_normal((N_TRAIN, IN_DIM))
    x_test = rng_data.standard_normal((N_TEST, IN_DIM))

    w_base = np.random.default_rng(BASE_SEED).standard_normal(
        (IN_DIM, NUM_CLASSES)) * 0.5

    rng_target = np.random.default_rng(TARGET_SEED)
    a_true = rng_target.standard_normal((IN_DIM, RANK)) * 0.6
    b_true = rng_target.standard_normal((RANK, NUM_CLASSES)) * 0.6
    w_target = w_base + SCALING * (a_true @ b_true)

    y_train = np.argmax(x_train @ w_target, axis=1)
    y_test = np.argmax(x_test @ w_target, axis=1)
    return x_train, y_train, x_test, y_test, w_base


def init_adapter():
    """A starts small and random, B starts at zero.

    With B at zero the adapter adds nothing at step 0, so the untrained
    model is exactly the base model. The gradient for B is non-zero at
    step 0, so training can start; the gradient for A becomes non-zero
    once B has moved.
    """
    a = np.random.default_rng(ADAPTER_SEED).standard_normal(
        (IN_DIM, RANK)) * 0.1
    b = np.zeros((RANK, NUM_CLASSES))
    return a, b


def effective_weights(w_base, a, b):
    return w_base + SCALING * (a @ b)


def evaluate(x, y, w_base, a, b):
    """Mean cross entropy loss and accuracy for the adapted model."""
    logits = x @ effective_weights(w_base, a, b)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(y)
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    accuracy = float((logits.argmax(axis=1) == y).mean())
    return loss, accuracy, probs


def train(x_train, y_train, w_base, a, b, epochs=EPOCHS, lr=LEARNING_RATE):
    """Full batch gradient descent on A and B only. W_base is not updated."""
    a = a.copy()
    b = b.copy()
    n = len(y_train)
    for _ in range(epochs):
        _, _, probs = evaluate(x_train, y_train, w_base, a, b)
        d_logits = probs.copy()
        d_logits[np.arange(n), y_train] -= 1.0
        d_logits /= n
        d_w = x_train.T @ d_logits
        d_delta = SCALING * d_w
        d_a = d_delta @ b.T
        d_b = a.T @ d_delta
        a = a - lr * d_a
        b = b - lr * d_b
    return a, b


def weights_hash(weights):
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


def run_lora():
    x_train, y_train, x_test, y_test, w_base = make_task()
    base_hash_before = weights_hash(w_base)
    a_init, b_init = init_adapter()

    train_loss_before, train_acc_before, _ = evaluate(
        x_train, y_train, w_base, a_init, b_init)
    test_loss_before, test_acc_before, _ = evaluate(
        x_test, y_test, w_base, a_init, b_init)

    a_trained, b_trained = train(x_train, y_train, w_base, a_init, b_init)

    train_loss_after, train_acc_after, _ = evaluate(
        x_train, y_train, w_base, a_trained, b_trained)
    test_loss_after, test_acc_after, _ = evaluate(
        x_test, y_test, w_base, a_trained, b_trained)

    lora_params = lora_parameter_count()
    full_params = full_parameter_count()
    return {
        "task": {
            "in_dim": IN_DIM,
            "num_classes": NUM_CLASSES,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "data_seed": DATA_SEED,
            "base_seed": BASE_SEED,
            "target_seed": TARGET_SEED,
            "adapter_seed": ADAPTER_SEED,
            "label_rule": "argmax of x @ W_target, W_target = W0 + scaling * (A_true @ B_true)",
        },
        "lora": {
            "rank": RANK,
            "alpha": ALPHA,
            "scaling": SCALING,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "trainable_parameters": lora_params,
            "full_finetune_parameters": full_params,
            "trainable_fraction_of_full": round(lora_params / full_params, 4),
            "base_weights_hash_before_training": base_hash_before,
            "base_weights_hash_after_training": weights_hash(w_base),
            "base_weights_unchanged": bool(base_hash_before == weights_hash(w_base)),
        },
        "before_training": {
            "train_loss": round(train_loss_before, 4),
            "train_accuracy": round(train_acc_before, 4),
            "test_loss": round(test_loss_before, 4),
            "test_accuracy": round(test_acc_before, 4),
            "test_correct": int(round(test_acc_before * N_TEST)),
        },
        "after_training": {
            "train_loss": round(train_loss_after, 4),
            "train_accuracy": round(train_acc_after, 4),
            "test_loss": round(test_loss_after, 4),
            "test_accuracy": round(test_acc_after, 4),
            "test_correct": int(round(test_acc_after * N_TEST)),
        },
    }
