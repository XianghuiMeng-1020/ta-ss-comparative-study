"""
Machine unlearning module for P2 naive-student simulation.

Implements the LoRA + KL forgetting procedure from Jiajia et al. (2026) and extends
it to Qwen2.5-3B for cross-family replication.

Modules:
  unlearn.py      — single training run (one model, one ratio, one seed)
  relearn.py      — coach-agent re-exposure loop
  train_runner.py — orchestrates all ratios × seeds × models
  evaluate.py     — accuracy / F1 / human-likeness evaluation for P2 outputs
"""
