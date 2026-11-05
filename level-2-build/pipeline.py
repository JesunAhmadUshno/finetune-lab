"""Full fine-tune pipeline, NumPy only. Level 2, Build, finetune-lab (P10).

Extends Level 1 Learn in four named ways:
1. Data prep stage: a deterministic train, validation, and held-out
   test split with a batch iterator and a data card (counts, class
   balance, seeds). Learn had train and test only, no validation.
2. SFT-style LoRA training: minibatch gradient descent (batch 64)
   over shuffled epochs, training only the LoRA adapter A and B on
   a frozen base, with per-epoch train and validation logging.
   This is SFT-style in the teaching sense only: the examples are
   synthetic vectors with class labels standing in for instruction
   and response pairs. No text, tokens, or real LLM are involved.
3. Checkpoint save and load: the adapter, its config, and the base
   hash are saved to a .npz checkpoint plus a JSON sidecar, and can
   be loaded back with hash verification.
4. Eval harness: before and after accuracy, loss, per-class
   accuracy, and a confusion matrix on the held-out test set, plus
   validation numbers. The harness is the scoring surface for both
   the SFT track and the reporting in results_build.json.

Task construction is identical to Learn so numbers are comparable:
inputs in R^16, data seed 7, base seed 3, target seed 9, adapter
seed 5, rank 2, alpha 4.0, scaling 2.0. The split here draws 512
train, 128 validation, and 128 test examples from one stream in
that order, so the first 512 and next 128 match the Learn train
and test draws only in stream position, not in role: validation
here is the slice Learn called test, and test here is a fresh
held-out slice Learn never used. That is stated plainly so no one
compares the wrong slices.
"""
import hashlib
import json
import os

import numpy as np

IN_DIM = 16
NUM_CLASSES = 4
RANK = 2
ALPHA = 4.0
SCALING = ALPHA / RANK
N_TRAIN = 512
N_VAL = 128
N_TEST = 128
DATA_SEED = 7
BASE_SEED = 3
TARGET_SEED = 9
ADAPTER_SEED = 5
SHUFFLE_SEED = 11
BATCH_SIZE = 64
LEARNING_RATE = 0.5
EPOCHS = 60


def lora_parameter_count(in_dim=IN_DIM, num_classes=NUM_CLASSES, rank=RANK):
    return in_dim * rank + rank * num_classes


def full_parameter_count(in_dim=IN_DIM, num_classes=NUM_CLASSES):
    return in_dim * num_classes


def weights_hash(weights):
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


def make_dataset():
    """Data prep: build the task and split it deterministically."""
    rng_data = np.random.default_rng(DATA_SEED)
    total = N_TRAIN + N_VAL + N_TEST
    x_all = rng_data.standard_normal((total, IN_DIM))

    w_base = np.random.default_rng(BASE_SEED).standard_normal(
        (IN_DIM, NUM_CLASSES)) * 0.5
    rng_target = np.random.default_rng(TARGET_SEED)
    a_true = rng_target.standard_normal((IN_DIM, RANK)) * 0.6
    b_true = rng_target.standard_normal((RANK, NUM_CLASSES)) * 0.6
    w_target = w_base + SCALING * (a_true @ b_true)
    y_all = np.argmax(x_all @ w_target, axis=1)

    x_train, y_train = x_all[:N_TRAIN], y_all[:N_TRAIN]
    x_val, y_val = x_all[N_TRAIN:N_TRAIN + N_VAL], y_all[N_TRAIN:N_TRAIN + N_VAL]
    x_test, y_test = x_all[N_TRAIN + N_VAL:], y_all[N_TRAIN + N_VAL:]
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "w_base": w_base,
    }


def data_card(dataset):
    """Counts and class balance per split, for the data prep record."""
    card = {}
    for name in ("train", "val", "test"):
        y = dataset[f"y_{name}"]
        counts = [int((y == c).sum()) for c in range(NUM_CLASSES)]
        card[name] = {"n": int(len(y)), "class_counts": counts}
    card["seeds"] = {"data": DATA_SEED, "base": BASE_SEED,
                     "target": TARGET_SEED, "adapter": ADAPTER_SEED,
                     "shuffle": SHUFFLE_SEED}
    card["in_dim"] = IN_DIM
    card["num_classes"] = NUM_CLASSES
    return card


def batch_indices(n, batch_size=BATCH_SIZE, epoch=0, shuffle_seed=SHUFFLE_SEED):
    """Deterministic shuffled minibatch index lists for one epoch."""
    rng = np.random.default_rng(shuffle_seed + epoch)
    order = rng.permutation(n)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def init_adapter():
    a = np.random.default_rng(ADAPTER_SEED).standard_normal(
        (IN_DIM, RANK)) * 0.1
    b = np.zeros((RANK, NUM_CLASSES))
    return a, b


def effective_weights(w_base, a, b):
    return w_base + SCALING * (a @ b)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(x, y, w_base, a, b):
    """Eval harness core: loss, accuracy, per-class accuracy, confusion."""
    logits = x @ effective_weights(w_base, a, b)
    probs = softmax(logits)
    n = len(y)
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    pred = logits.argmax(axis=1)
    accuracy = float((pred == y).mean())
    per_class = []
    for c in range(NUM_CLASSES):
        mask = y == c
        per_class.append(round(float((pred[mask] == y[mask]).mean()), 4)
                         if mask.any() else 0.0)
    confusion = [[int(((y == r) & (pred == c)).sum())
                  for c in range(NUM_CLASSES)] for r in range(NUM_CLASSES)]
    return {"loss": round(loss, 4), "accuracy": round(accuracy, 4),
            "correct": int((pred == y).sum()), "n": int(n),
            "per_class_accuracy": per_class, "confusion": confusion}


def train_sft(dataset, epochs=EPOCHS, lr=LEARNING_RATE,
              batch_size=BATCH_SIZE):
    """Minibatch SFT-style LoRA training. Only A and B update."""
    w_base = dataset["w_base"]
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    a, b = init_adapter()
    a, b = a.copy(), b.copy()
    history = []
    n = len(y_train)
    for epoch in range(epochs):
        for idx in batch_indices(n, batch_size, epoch=epoch):
            xb, yb = x_train[idx], y_train[idx]
            m = len(yb)
            probs = softmax(xb @ effective_weights(w_base, a, b))
            d_logits = probs.copy()
            d_logits[np.arange(m), yb] -= 1.0
            d_logits /= m
            d_w = xb.T @ d_logits
            d_delta = SCALING * d_w
            d_a = d_delta @ b.T
            d_b = a.T @ d_delta
            a = a - lr * d_a
            b = b - lr * d_b
        if epoch % 10 == 9 or epoch == 0:
            tr = evaluate(x_train, y_train, w_base, a, b)
            va = evaluate(dataset["x_val"], dataset["y_val"], w_base, a, b)
            history.append({"epoch": epoch + 1,
                            "train_loss": tr["loss"],
                            "train_accuracy": tr["accuracy"],
                            "val_loss": va["loss"],
                            "val_accuracy": va["accuracy"]})
    return a, b, history


def save_checkpoint(path_npz, a, b, w_base, extra=None):
    """Save adapter checkpoint (.npz) plus a JSON sidecar with hashes."""
    np.savez(path_npz, a=a, b=b)
    sidecar = {
        "format": "finetune-lab-adapter-v1",
        "rank": RANK, "alpha": ALPHA, "scaling": SCALING,
        "in_dim": IN_DIM, "num_classes": NUM_CLASSES,
        "a_hash": weights_hash(a), "b_hash": weights_hash(b),
        "base_weights_hash": weights_hash(w_base),
        "trainable_parameters": lora_parameter_count(),
    }
    if extra:
        sidecar.update(extra)
    sidecar_path = path_npz + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    return sidecar


def load_checkpoint(path_npz, w_base=None):
    """Load a checkpoint and verify hashes against its sidecar."""
    with open(path_npz + ".json", encoding="utf-8") as f:
        sidecar = json.load(f)
    data = np.load(path_npz)
    a, b = data["a"], data["b"]
    if weights_hash(a) != sidecar["a_hash"]:
        raise ValueError("checkpoint A hash mismatch")
    if weights_hash(b) != sidecar["b_hash"]:
        raise ValueError("checkpoint B hash mismatch")
    if w_base is not None and weights_hash(w_base) != sidecar["base_weights_hash"]:
        raise ValueError("base weights hash mismatch")
    return a, b, sidecar


def run_pipeline(checkpoint_path=None):
    dataset = make_dataset()
    w_base = dataset["w_base"]
    base_hash_before = weights_hash(w_base)
    a_init, b_init = init_adapter()

    before_test = evaluate(dataset["x_test"], dataset["y_test"], w_base, a_init, b_init)
    before_val = evaluate(dataset["x_val"], dataset["y_val"], w_base, a_init, b_init)
    before_train = evaluate(dataset["x_train"], dataset["y_train"], w_base, a_init, b_init)

    a_trained, b_trained, history = train_sft(dataset)

    after_test = evaluate(dataset["x_test"], dataset["y_test"], w_base, a_trained, b_trained)
    after_val = evaluate(dataset["x_val"], dataset["y_val"], w_base, a_trained, b_trained)
    after_train = evaluate(dataset["x_train"], dataset["y_train"], w_base, a_trained, b_trained)

    ckpt_info = None
    if checkpoint_path is not None:
        sidecar = save_checkpoint(checkpoint_path, a_trained, b_trained, w_base)
        a_loaded, b_loaded, _ = load_checkpoint(checkpoint_path, w_base=w_base)
        reloaded = evaluate(dataset["x_test"], dataset["y_test"], w_base, a_loaded, b_loaded)
        ckpt_info = {
            "sidecar": sidecar,
            "round_trip_test_accuracy": reloaded["accuracy"],
            "round_trip_test_loss": reloaded["loss"],
            "round_trip_matches_after": bool(
                reloaded["accuracy"] == after_test["accuracy"]
                and reloaded["loss"] == after_test["loss"]),
        }

    return {
        "data_card": data_card(dataset),
        "config": {"rank": RANK, "alpha": ALPHA, "scaling": SCALING,
                   "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                   "epochs": EPOCHS,
                   "trainable_parameters": lora_parameter_count(),
                   "full_finetune_parameters": full_parameter_count(),
                   "base_weights_hash_before": base_hash_before,
                   "base_weights_hash_after": weights_hash(w_base),
                   "base_weights_unchanged": bool(base_hash_before == weights_hash(w_base))},
        "before_training": {"train": before_train, "val": before_val, "test": before_test},
        "after_training": {"train": after_train, "val": after_val, "test": after_test},
        "history": history,
        "checkpoint": ckpt_info,
    }
