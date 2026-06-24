#!/usr/bin/env python
"""
Convert a best model checkpoint to a resume checkpoint.
Allows resuming training from an interrupted run started with old code.

Usage:
  python convert_best_to_resume.py --data_name _DA_SleepEEG_256_00 --view2 logsig --view3 xf --epochs_pretrain 200 --seed 0
"""

import os
import sys
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert best checkpoint to resume checkpoint for interrupted training."
    )
    parser.add_argument("--data_name", required=True, help="Dataset name (e.g. _DA_SleepEEG_256_00)")
    parser.add_argument("--view2", required=True, help="Second view (e.g. logsig, dx)")
    parser.add_argument("--view3", required=True, help="Third view (e.g. xf)")
    parser.add_argument("--epochs_pretrain", type=int, required=True, help="Total pretrain epochs")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--num_feature", type=int, default=1, help="Number of input features (must match original training)")

    args = parser.parse_args()

    # Paths
    best_ckpt_path = (
        f"model_pretrain/{args.data_name}/"
        f"{args.data_name}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}.pth"
    )
    resume_ckpt_path = (
        f"out_pretrain/.resume_{args.data_name}_v2{args.view2}_v3{args.view3}"
        f"_ep{args.epochs_pretrain}_{args.seed}.pth"
    )

    print(f"Using num_feature={args.num_feature}")

    if not os.path.exists(best_ckpt_path):
        print(f"Error: Best checkpoint not found: {best_ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading best checkpoint: {best_ckpt_path}")
    best_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

    # Extract epoch and other state
    epoch = best_state.get("epoch")
    if epoch is None:
        print("Error: checkpoint does not contain 'epoch' key", file=sys.stderr)
        sys.exit(1)

    print(f"Best checkpoint was saved at epoch {epoch}")

    # Build resume checkpoint
    resume_state = {
        "epoch": epoch,
        "encoder_state_dict": best_state["encoder_state_dict"],
        "optimizer_state_dict": best_state["optimizer_state_dict"],
        "scheduler_state_dict": best_state["scheduler_state_dict"],
        "loss_list": best_state["loss_list"],
        "best_valid_loss": best_state["best_valid_loss"],
        "early_stop_counter": 0,  # Reset to 0; resume will continue from best model
        "num_feature": args.num_feature,  # Store for resume validation
    }

    os.makedirs(os.path.dirname(resume_ckpt_path), exist_ok=True)
    torch.save(resume_state, resume_ckpt_path)
    print(f"Wrote resume checkpoint: {resume_ckpt_path}")
    print(f"Next run will resume from epoch {epoch + 1}")
