import argparse

def build_argparser():
    p = argparse.ArgumentParser(description="Train SpeechNet on SilentWear EMG data")
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--x_path", type=str, required=True)
    p.add_argument("--y_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/exp")
    p.add_argument("--window_ms", type=int, default=1400, choices=[400,600,800,1000,1200,1400])
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--meta_path", type=str, default=None)
    p.add_argument("--split_mode", type=str, default="random",
                   choices=["random", "inter_session", "inter_subject"])
    p.add_argument("--held_out_session", type=int, default=None)
    p.add_argument("--held_out_subject", type=str, default=None)
    p.add_argument("--apply_preprocessing", action="store_true")

    # Channel selection
    p.add_argument("--channel_indices", type=str, default=None,
                   help="comma-separated channel indices, e.g. '2,3,5,6,7,8,9,10'")

    # Pretraining
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--freeze_backbone", action="store_true")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Model options
    p.add_argument("--width_factor", type=float, default=1.0)
    p.add_argument("--n_res_blocks", type=int, default=2)
    p.add_argument("--use_se", action="store_true")
    p.add_argument("--channel_attention", action="store_true")
    p.add_argument("--channel_dropout", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batchnorm", action="store_true", default=True)

    # MixUp
    p.add_argument("--mixup_alpha", type=float, default=0.2)

    # Plotting
    p.add_argument("--plot_confusion", action="store_true")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    
    return p