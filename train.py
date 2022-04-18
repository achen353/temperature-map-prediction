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

    EPOCH = config["epoch"]
    model_class = MODELS[config["model"]["class"]]
    optimizer_class = getattr(optim, config["optimizer"]["name"])
    scheduler_class = getattr(lr_scheduler, config["scheduler"]["name"])
    loss_class = getattr(nn, config["loss"])

    model = model_class(**config["model"]["hyperparams"])
    device = model.device

    model.to(model.device)

    criterion = loss_class()
    optimizer = optimizer_class(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = scheduler_class(
        optimizer=optimizer, **config["scheduler"]["hyperparams"]
    )

    train_dataset = MODISDataset(**config["dataset"]["hyperparams"])
    collate_fn = train_dataset.collate

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    valid_dataset = MODISDataset(**config["dataset"]["hyperparams"], mode="validation")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(comment=config["name"])

    for epoch in range(EPOCH):
        losses = []
        for i, (input, pred, encodings) in enumerate(train_dataloader):
            # Make sure batch_size is in the second dim
            input, pred = input.permute(1, 0, 2), pred.permute(1, 0, 2)

            # Place tensors on GPU
            input, pred, encodings = (
                input.to(device),
                pred.to(device),
                encodings.to(device),
            )

            optimizer.zero_grad()
            output = model(input, pos_info=encodings)

            loss = criterion(output, pred)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

        scheduler.step()

        writer.add_scalar("loss/train_epoch", np.mean(losses), epoch)

        model.eval()
        with torch.no_grad():
            # TODO: Add evaluation metric (@Andrew)
            v_losses = []
            for i, (v_input, v_pred, v_encodings) in enumerate(valid_dataloader):
                v_input, v_pred = v_input.permute(1, 0, 2), v_pred.permute(1, 0, 2)
                v_input, v_pred = v_input.to(device), v_pred.to(device)
                v_output = model(v_input)
                v_loss = criterion(v_output, v_pred)
                v_loss_val = v_loss.item()
                v_losses.append(v_loss_val)
            writer.add_scalar("loss/valid_epoch", np.mean(v_losses), epoch)

        model.train()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(losses),
            },
            "./model_checkpoints/model_" + config["name"] + ".pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_name", type=str, default="transformer.json")
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config_name)
    main(config_path)
