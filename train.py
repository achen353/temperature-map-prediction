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


def main(config_path, trial_num=0):
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

    RUN_NAME = "{}_{}".format(config["name"], trial_num)
    EPOCH = config["epoch"]
    model_class = MODELS[config["model"]["class"]]
    optimizer_class = getattr(optim, config["optimizer"]["name"])
    scheduler_class = getattr(lr_scheduler, config["scheduler"]["name"])
    loss_class = getattr(nn, config["loss"])

    has_pos_info = "enc" in config["name"]
    token_is_zero = "ulast" not in config["name"]
    use_diff = config["dataset"]["hyperparam"]["use_diff"]

    model = model_class(**config["model"]["hyperparams"])

    if "device" in config:
        model.device = config["device"]
    device = model.device
    model.to(model.device)

    criterion = loss_class()
    optimizer = optimizer_class(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = scheduler_class(
        optimizer=optimizer, **config["scheduler"]["hyperparams"]
    )

    if not os.path.exists("./model_checkpoints/{}".format(RUN_NAME)):
        os.makedirs("./model_checkpoints/{}/all".format(RUN_NAME))
        os.makedirs("./model_checkpoints/{}/best".format(RUN_NAME))

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

    test_dataset = MODISDataset(**config["dataset"]["hyperparams"], mode="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(comment=config["name"])

    n_iter = 0
    best_valid_loss, best_test_loss = float("inf"), float("inf")

    for epoch in range(EPOCH):
        losses = []
        print("[INFO] Epoch: {}".format(epoch + 1))
        for i, (input, pred, encodings) in enumerate(train_dataloader):
            model.train()
            n_iter += 1

            # Make sure batch_size is in the second dim
            input, pred, encodings = (
                input.permute(1, 0, 2),
                pred.permute(1, 0, 2),
                encodings.permute(1, 0).unsqueeze(dim=-1),
            )

            # Place tensors on GPU
            input, pred, encodings = (
                input.to(device),
                pred.to(device),
                encodings.to(device),
            )

            optimizer.zero_grad()

            if model_class == Transformer:
                if has_pos_info:
                    output = model(
                        input, pos_info=encodings, token_is_zero=token_is_zero
                    )
                else:
                    output = model(input, pos_info=None, token_is_zero=token_is_zero)
            else:
                output = model(input)

            loss = criterion(output, pred)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            writer.add_scalar("loss/train_iter", loss_val, n_iter)

            model.eval()
            with torch.no_grad():
                # Validation
                v_losses = []
                for i, (v_input, v_pred, v_encodings) in enumerate(valid_dataloader):
                    v_input, v_pred, v_encodings = (
                        v_input.permute(1, 0, 2),
                        v_pred.permute(1, 0, 2),
                        v_encodings.permute(1, 0).unsqueeze(dim=-1),
                    )

                    v_input, v_pred, v_encodings = (
                        v_input.to(device),
                        v_pred.to(device),
                        v_encodings.to(device),
                    )

                    if model_class == Transformer:
                        if has_pos_info:
                            v_output = model(
                                v_input,
                                pos_info=v_encodings,
                                token_is_zero=token_is_zero,
                            )
                        else:
                            v_output = model(
                                v_input, pos_info=None, token_is_zero=token_is_zero
                            )
                    else:
                        v_output = model(v_input)

                    if use_diff:
                        v_output = torch.cumsum(v_output, dim=0)
                        v_pred = torch.cumsum(v_pred, dim=0)

                    v_loss = criterion(v_output, v_pred)
                    v_loss_val = v_loss.item()
                    v_losses.append(v_loss_val)

                valid_loss = np.mean(v_losses)

                writer.add_scalar("loss/valid_iter", valid_loss, n_iter)

                if valid_loss < best_valid_loss:
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
                                t_output = model(
                                    t_input,
                                    pos_info=t_encodings,
                                    token_is_zero=token_is_zero,
                                )
                            else:
                                t_output = model(
                                    t_input, pos_info=None, token_is_zero=token_is_zero
                                )
                        else:
                            t_output = model(t_input)

                        if use_diff:
                            t_output = torch.cumsum(t_output, dim=0)
                            t_pred = torch.cumsum(t_pred, dim=0)

                        t_loss = criterion(t_output, t_pred)
                        t_loss_val = t_loss.item()
                        t_losses.append(t_loss_val)

                    test_loss = np.mean(t_losses)

                    writer.add_scalar("loss/test_iter", test_loss, n_iter)

                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": n_iter,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "valid_loss": valid_loss,
                            "test_loss": test_loss,
                        },
                        "./model_checkpoints/{}/best/best.pt".format(RUN_NAME),
                    )
                    best_valid_loss = valid_loss
                    best_test_loss = test_loss

        train_loss = np.mean(losses)
        writer.add_scalar("loss/train_epoch", train_loss, epoch)

        scheduler.step()

    with open("results.csv", mode="a") as f:
        f.write(
            "{},{},{:.4f},{:.4f}\n".format(
                RUN_NAME, trial_num, best_valid_loss, best_test_loss
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_name", type=str, default="transformer.json")
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config_name)
    main(config_path)
