"""QLoRA memory accounting study. Level 3, Deep, P10.

This is accounting, not measurement. No GPU memory is measured
anywhere in this module. Every byte count is computed from a
stated parameter count and a stated bytes-per-parameter rule:

- Full fine-tune storage: all model parameters at 2 bytes each
  (fp16 weights). Optimizer state, gradients, activations, and
  master fp32 copies are excluded, and that exclusion is stated
  here and in the README so the full number is a weight-storage
  floor, not a training-memory total.
- LoRA adapter: the adapter parameters only, at 2 bytes each
  (fp16). The adapter adapts 2 matrices per layer (the query and
  value projections, the standard teaching choice), each matrix
  of shape hidden by hidden replaced by two low-rank factors of
  shape hidden by rank and rank by hidden, so one adapted matrix
  costs 2 * rank * hidden parameters and one layer costs
  2 matrices * 2 * rank * hidden parameters.
- LoRA total: full fp16 base plus the adapter, which is what a
  LoRA run must hold for the weights.
- QLoRA total: the base at 0.5 bytes per parameter (4-bit
  quantization, 4 bits is half a byte) plus the same fp16
  adapter. Quantization scales and zero points are excluded;
  at these model sizes they are a small fraction of the base
  and excluding them is stated, not hidden.

Model rows are stated illustrative sizes (parameter count,
layers, hidden width), not claims about the exact architecture
of any named released model. Rank is 16 for every row.
"""
import numpy as np

RANK = 16
MATRICES_PER_LAYER = 2
BYTES_FP16 = 2
BYTES_4BIT = 0.5

MODELS = [
    {"name": "small-125M", "parameters": 125_000_000,
     "layers": 12, "hidden": 768},
    {"name": "medium-1B", "parameters": 1_000_000_000,
     "layers": 24, "hidden": 2048},
    {"name": "large-7B", "parameters": 7_000_000_000,
     "layers": 32, "hidden": 4096},
    {"name": "xl-13B", "parameters": 13_000_000_000,
     "layers": 40, "hidden": 5120},
]


def lora_parameter_count(layers, hidden, rank=RANK,
                         matrices_per_layer=MATRICES_PER_LAYER):
    """Adapter parameters: layers * matrices * (rank*hidden + hidden*rank)."""
    per_matrix = rank * hidden + hidden * rank
    return int(layers * matrices_per_layer * per_matrix)


def memory_row(model, rank=RANK):
    """One accounting row, all bytes computed from the stated rules."""
    params = int(model["parameters"])
    lora_params = lora_parameter_count(model["layers"], model["hidden"],
                                       rank=rank)
    full_bytes = int(params * BYTES_FP16)
    adapter_bytes = int(lora_params * BYTES_FP16)
    lora_total_bytes = int(full_bytes + adapter_bytes)
    qlora_base_bytes = int(params * BYTES_4BIT)
    qlora_total_bytes = int(qlora_base_bytes + adapter_bytes)
    return {
        "name": model["name"],
        "parameters": params,
        "layers": int(model["layers"]),
        "hidden": int(model["hidden"]),
        "rank": int(rank),
        "lora_parameters": lora_params,
        "full_finetune_bytes": full_bytes,
        "lora_adapter_bytes": adapter_bytes,
        "lora_total_bytes": lora_total_bytes,
        "qlora_base_4bit_bytes": qlora_base_bytes,
        "qlora_total_bytes": qlora_total_bytes,
        "qlora_total_fraction_of_full": round(qlora_total_bytes / full_bytes, 4),
        "adapter_fraction_of_full": round(adapter_bytes / full_bytes, 4),
    }


def memory_study(models=None, rank=RANK):
    rows = [memory_row(m, rank=rank) for m in (models or MODELS)]
    return {
        "method": "accounting from parameter counts and bytes per "
                  "parameter, not measured GPU memory",
        "assumptions": {
            "bytes_fp16": BYTES_FP16,
            "bytes_4bit": BYTES_4BIT,
            "rank": int(rank),
            "matrices_per_layer": MATRICES_PER_LAYER,
            "adapted_matrices": "query and value projections",
            "excluded": "optimizer state, gradients, activations, "
                        "fp32 master copies, quantization scales "
                        "and zero points",
        },
        "rows": rows,
    }
