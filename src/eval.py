"""
Leave-one-session-out (LOSO) or leave-one-subject-out evaluation for SpeechNet.

python eval.py \
    --x_path ./processed_raw/X_silent.npy \
    --y_path ./processed_raw/y_silent.npy \
    --meta_path ./processed_raw/meta_silent.npz \
    --window_ms 1400 \
    --out_dir ./runs/8ch \
    --pretrained ./runs/8ch/vocalized_pretrained.pt \
    --num_channels 8 \
    --width_factor 2.0 \
    --n_res_blocks 4 \
    --use_se \
    --channel_attention \
    --dropout 0.5 \
    --channel_dropout 0.4 \
    --mixup_alpha 0.0 \
    --lr 5e-5 \
    --weight_decay 2e-2 \
    --epochs 200 \
    --batch_size 32 \
    --plot_confusion \
    --split_mode inter_session \
    --channel_indices 2,3,5,6,7,8,9,10
"""

import argparse, json, os
import numpy as np
import torch

import train as train_module

def build_fold_args(base_args, split_mode, held_out_id, fold_out_dir):
    fold_args = argparse.Namespace(**vars(base_args))
    fold_args.split_mode = split_mode
    fold_args.out_dir = fold_out_dir

    if split_mode == "inter_session":
        fold_args.held_out_session = held_out_id
        fold_args.held_out_subject = None
    elif split_mode == "inter_subject":
        fold_args.held_out_subject = held_out_id
        fold_args.held_out_session = None
    else:
        raise ValueError(f"Unsupported split_mode for LOSO: {split_mode}")

    return fold_args


def main():
    parser = train_module.build_argparser()
    parser.description = "Leave-one-session/subject-out evaluation wrapper around train.py"

    if not any(action.dest == 'channel_indices' for action in parser._actions):
        parser.add_argument("--channel_indices", type=str, default=None,
                            help="comma-separated channel indices to use, e.g. '2,3,5,6,7,8,9,10'")
    args = parser.parse_args()

    if args.meta_path is None:
        raise ValueError("--meta_path is required for LOSO (need session/subject ids)")

    meta = np.load(args.meta_path)
    split_mode = args.split_mode

    if split_mode == "inter_session":
        ids = sorted(set(meta["session"].tolist()))
        id_name = "session"
    elif split_mode == "inter_subject":
        ids = sorted(set(meta["subject"].tolist()))
        id_name = "subject"
    else:
        raise ValueError(f"LOSO evaluation only supports inter_session or inter_subject; got {split_mode}")

    print("=" * 70)
    print(f"Leave-one-{id_name}-out evaluation over {id_name}s: {ids}")
    print("=" * 70)

    os.makedirs(args.out_dir, exist_ok=True)
    fold_results = []

    for held_out_id in ids:
        fold_out_dir = os.path.join(args.out_dir, f"held_out_{id_name}_{held_out_id}")
        fold_args = build_fold_args(args, split_mode, held_out_id, fold_out_dir)

        print()
        print(f"### Fold: held-out {id_name} = {held_out_id} ###")

        _, val_acc = train_module.train(fold_args)

        ckpt_path = os.path.join(fold_out_dir, "vocalized_pretrained.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        fold_results.append({
            id_name: held_out_id,
            "val_balanced_accuracy": float(val_acc),
            "val_macro_f1": float(ckpt["val_macro_f1"]),
        })

    accs = np.array([r["val_balanced_accuracy"] for r in fold_results])
    f1s = np.array([r["val_macro_f1"] for r in fold_results])

    print()
    print("=" * 70)
    print("LOSO SUMMARY")
    print("=" * 70)
    for r in fold_results:
        print(f"  {id_name} {r[id_name]}: balanced acc = {r['val_balanced_accuracy']*100:5.1f}% | macro F1 = {r['val_macro_f1']*100:5.1f}%")
    print("-" * 50)
    print(f"  mean balanced acc = {accs.mean()*100:.1f}% +/- {accs.std()*100:.1f}%  (n={len(accs)} {id_name}s)")
    print(f"  mean macro F1     = {f1s.mean()*100:.1f}% +/- {f1s.std()*100:.1f}%  (n={len(f1s)} {id_name}s)")
    print("\nFor reference, the SilentWear paper reports 71.1+/-8.3% (vocalized) and 59.3+/-2.2% (silent).")

    summary = {
        "folds": fold_results,
        "mean_balanced_accuracy": float(accs.mean()),
        "std_balanced_accuracy": float(accs.std()),
        "mean_macro_f1": float(f1s.mean()),
        "std_macro_f1": float(f1s.std()),
        "split_mode": split_mode,
        "ids": ids,
    }
    with open(os.path.join(args.out_dir, "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {os.path.join(args.out_dir, 'loso_summary.json')}")


if __name__ == "__main__":
    main()