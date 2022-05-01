import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.dataloader_v2 import MODISDataset
from modules.LSTM_benchmark import LSTM
from modules.transformer_map_pred import Transformer

MODELS = {"Transformer": Transformer, "LSTM": LSTM}


def main(config_path):
    try:
        print("[INFO] Loading config from: {}".format(config_path))
        config = json.load(open(config_path))
    except:
        raise FileNotFoundError("Config {} does not exist.".format(config_path))

    if config["seed"]["fixed"]:
        seed = config["seed"]["value"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model_class = MODELS[config["model"]["class"]]
    loss_class = getattr(nn, config["loss"])

    has_pos_info = "enc" in config["name"]
    token_is_zero = "ulast" not in config["name"]
    use_diff = config["dataset"]["hyperparams"]["use_diff"]

    model = model_class(**config["model"]["hyperparams"])

    if "device" in config:
        model.device = "cuda:5"
    device = model.device
    model.to(model.device)

    resume_from_checkpoint = os.path.join(
        "model_checkpoints/", config["resume_from_checkpoint"], "best/best.pt"
    )
    checkpoint = torch.load(resume_from_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = loss_class()

    train_dataset = MODISDataset(**config["dataset"]["hyperparams"])
    collate_fn = train_dataset.collate

    test_dataset = MODISDataset(**config["dataset"]["hyperparams"], mode="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test
    t_losses = []
    for i, (t_input, t_pred, t_encodings) in enumerate(test_dataloader):
        t_input, t_pred, t_encodings = (
            t_input.permute(1, 0, 2),
            t_pred.permute(1, 0, 2),
            t_encodings.permute(1, 0).unsqueeze(dim=-1),
        )

        t_input, t_pred, t_encodings = (
            t_input.to(device),
            t_pred.to(device),
            t_encodings.to(device),
        )

        if model_class == Transformer:
            if has_pos_info:
                t_output = model(t_input, pos_info=t_encodings, token_is_zero=token_is_zero)
            else:
                t_output = model(t_input, pos_info=None, token_is_zero=token_is_zero)
        else:
            t_output = model(t_input)

        if use_diff:
            t_output = torch.cumsum(t_output, dim=0)
            t_pred = torch.cumsum(t_pred, dim=0)

        t_loss = criterion(t_output, t_pred)
        t_loss_val = t_loss.item()
        t_losses.append(t_loss_val)
    
    test_loss = np.mean(t_losses)

    print("[INFO] Test Loss: {}".format(test_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_name", type=str, default="transformer.json")
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config_name)
    main(config_path)
