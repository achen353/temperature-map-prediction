import os

from train import main

CONFIGS = [
    "lstm",
    "transformer",
    "transformer_diff",
    "transformer_enc",
    "transformer_ulast",
    "transformer_diff_ulast",
    "transformer_enc_diff",
    "transformer_enc_ulast",
    "transformer_enc_diff_ulast",
]

for config in CONFIGS:
    config_path = os.path.join("configs", config + ".json")
    for n_trial in range(5):
        main(config_path, n_trial)
