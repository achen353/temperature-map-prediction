#!/usr/bin/env/python3


"""
About
-----
Script for the Dataloader of the MODIS images

Classes
-------
MODISDataset

Functions
---------
__init__
_init_dataset
_sort_images_by_date
_prepare_single_sequence
_prepare_train_test_validation_data
_create_dir
__getitem__
__len__

Variables
---------
"""

# Meta-data.
__author__ = "Sourish Gunesh Dhekane"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "2.1"
__maintainer__ = "Sourish Gunesh Dhekane"
__email__ = "sourish.dhekane@gatech.edu"
__status__ = ""

# Dependencies.
import calendar
import os
import random
import re
import shutil
from datetime import datetime
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Main.
if __name__ == "__main__":
    pass


class MODISDataset(Dataset):
    """
    Class for loading MODIS Images
    """

    def __init__(
        self,
        patch_dim: int = 16,
        input_seq_len: int = 10,
        prediction_seq_len: int = 3,
        number_training_seq: int = 10,
        number_validation_seq: int = 2,
        number_test_seq: int = 2,
        mode: str = "training",
        modis_img_path: str = "./datasets/MOD11A2",
    ):
        self.patch_dim = patch_dim
        self.input_seq_len = input_seq_len
        self.prediction_seq_len = prediction_seq_len
        self.number_training_seq = number_training_seq
        self.number_validation_seq = number_validation_seq
        self.number_test_seq = number_test_seq
        self.mode = mode

        self.modis_image_path = modis_img_path

        self.unique_datapoints_set = set()
        self.nan_threshold = 0.2
        self.training_data_seq_id = None
        self.test_data_seq_id = None
        self.validation_data_seq_id = None

        self.h5_path = "dataset.h5"

        self._init_dataset()

    def _init_dataset(self):
        """
        Initialize the dataset and store them as `dataset.h5`
        """
        if not os.path.exists(self.h5_path):
            print("[INFO] Creating dataset...")

            # Split the data and prepare them
            self._prepare_train_test_validation_data()

            # Start writing to h5py file
            h5file = h5py.File(self.h5_path, "w")

            for data_type in ["training", "test", "validation"]:
                seq_ids = getattr(self, "{}_data_seq_id".format(data_type))
                data = getattr(self, "{}_data".format(data_type))

                input_modis_sequences, pred_modis_sequences, input_seq_encodings = (
                    [],
                    [],
                    [],
                )

                for i, data_point in enumerate(data):
                    inp_seq, pred_seq = data_point
                    seq_id = seq_ids[i]

                    # Construct the actual input image sequence of shape:
                    # [input_seq_len, patch_dim, patch_dim]
                    input_modis_seq = np.zeros(
                        (self.input_seq_len, self.patch_dim, self.patch_dim)
                    )
                    # At the same time, construct the encoding of each sequence
                    # of shape [input_seq_len] based on `img_str`
                    input_seq_enc = np.zeros(self.input_seq_len)

                    for j, img_str in enumerate(inp_seq):
                        # Read the image
                        image_path = os.path.join(
                            self.modis_image_path, img_str + ".tif"
                        )
                        modis_img = cv2.imread(image_path, -1)
                        # Perform cropping
                        modis_img = modis_img[
                            int(seq_id[1]) : int(seq_id[1]) + self.patch_dim,
                            int(seq_id[2]) : int(seq_id[2]) + self.patch_dim,
                        ]
                        # Aggregate the cropped patch
                        input_modis_seq[j, :, :] = modis_img

                        # Calculate the encoding
                        if self.start_year is None:
                            raise NameError(
                                "Instance variable `start_year` is not defined."
                            )
                        time_tuple = datetime.strptime(img_str, "%Y_%m_%d").timetuple()
                        total_yday = 366 if calendar.isleap(time_tuple.tm_year) else 365
                        enc = round(
                            time_tuple.tm_year
                            - self.start_year
                            + time_tuple.tm_yday / total_yday,
                            3,
                        )
                        # Aggregate the encoding
                        input_seq_enc[j] = enc

                    # Construct the actual prediction image sequence of shape:
                    # [pred_seq_len, patch_dim, patch_dim]
                    pred_modis_seq = np.zeros(
                        (self.prediction_seq_len, self.patch_dim, self.patch_dim)
                    )
                    for j, img_str in enumerate(pred_seq):
                        # Read the image
                        image_path = os.path.join(
                            self.modis_image_path, img_str + ".tif"
                        )
                        modis_img = cv2.imread(image_path, -1)
                        # Perform cropping
                        modis_img = modis_img[
                            int(seq_id[1]) : int(seq_id[1]) + self.patch_dim,
                            int(seq_id[2]) : int(seq_id[2]) + self.patch_dim,
                        ]
                        pred_modis_seq[j, :, :] = modis_img

                    input_modis_sequences.append(input_modis_seq)
                    pred_modis_sequences.append(pred_modis_seq)
                    input_seq_encodings.append(input_seq_enc)

                input_modis_sequences = np.stack(input_modis_sequences)
                pred_modis_sequences = np.stack(pred_modis_sequences)
                input_seq_encodings = np.stack(input_seq_encodings)

                data_group = h5file.create_group(data_type)
                data_group.create_dataset("input", data=input_modis_sequences)
                data_group.create_dataset("pred", data=pred_modis_sequences)
                data_group.create_dataset("encodings", data=input_seq_encodings)

            h5file.close()

            print("[INFO] Saving created dataset to: {}".format(self.h5_path))

        print("[INFO] Loading dataset from: {}".format(self.h5_path))

        self.dataset = h5py.File(
            self.h5_path, "r"
        )  # self.dataset is only a HDF5 pointer

        print("[INFO] Dataset loaded.")

    def _sort_images_by_date(self) -> List[str]:
        """
        Collect image names in a list, sort that list, and return it

        Returns:
        -   list[str]: a list of image names in a sorted order
        """
        pattern = re.compile(r"^\d{4}_\d{2}_\d{2}.tif$")

        # Create list of images
        image_list = [
            filename.split(".")[0]
            for filename in os.listdir(self.modis_image_path)
            if pattern.search(filename)
        ]
        # Sort the list based on dates
        image_list.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d"))

        # Get the starting year of the dataset
        self.start_year = (
            datetime.strptime(image_list[0], "%Y_%m_%d").timetuple().tm_year
        )

        return image_list

    def _prepare_single_sequence(
        self, image_list: List[str]
    ) -> Tuple[Tuple[List[str], List[str]], Tuple[int, int, int]]:
        """
        Prepare one data point of the sequence-to-sequence prediction problem

        Args:
        -   new_image_list: the sorted list of ALL the image names in the dataset as per their date
        Returns:
        -   data_sequence: Tuple[List[str], List[str]]: a tuple of lists- first list being the input sequence of image
                          names and second list being the output sequence of image names
        -   sequence_identifier: Tuple[int, int, int] where the first element represents the image ID, second number
                                 represents the x-coordinate of the top-left point of the selected path, and the
                                third number represents the y-coordinate of the top-left point of the selected path
        """
        input_seq, pred_seq, seq_identifiers = None, None, None
        no_of_imgs = len(image_list)

        while True:
            # Random selection of sequence starting date
            seq_starting_point = random.randint(
                0, no_of_imgs - (self.input_seq_len + self.prediction_seq_len + 1)
            )

            # Obtain the dates of the input sequence
            input_seq = image_list[
                seq_starting_point : seq_starting_point + self.input_seq_len
            ]

            # Obtain the dates of the corresponding prediction sequence
            pred_seq = image_list[
                seq_starting_point
                + self.input_seq_len : seq_starting_point
                + self.input_seq_len
                + self.prediction_seq_len
            ]

            # Read the first image of the sequence to get its dimensions
            image_path = os.path.join(self.modis_image_path, input_seq[0] + ".tif")
            modis_img = cv2.imread(image_path, -1)
            modis_img_length = np.array(modis_img).shape[0]
            modis_img_breadth = np.array(modis_img).shape[1]

            # Randomly select a patch (subregion) for the sequence
            starting_point_x = random.randint(
                0, modis_img_length - (self.patch_dim + 1)
            )
            starting_point_y = random.randint(
                0, modis_img_breadth - (self.patch_dim + 1)
            )

            # Check NaN percentage (only check the first image of the sequence)
            modis_img = modis_img[
                starting_point_x : starting_point_x + self.patch_dim,
                starting_point_y : starting_point_y + self.patch_dim,
            ]
            nan_ratio = np.sum(np.isnan(modis_img)) / modis_img.size

            # If the sequence is unique and has few enough Nan, return it.
            # Otherwise, repeat until we generate one that satisfies
            seq_identifiers = (seq_starting_point, starting_point_x, starting_point_y)
            if (
                seq_identifiers not in self.unique_datapoints_set
                and nan_ratio < self.nan_threshold
            ):
                self.unique_datapoints_set.add(seq_identifiers)
                break

        return ((input_seq, pred_seq), seq_identifiers)

    def _prepare_train_test_validation_data(self) -> None:
        """
        Prepare training, test, and validation datasets for the sequence-to-sequence prediction problem
        """
        image_list = self._sort_images_by_date()

        # Initialize a set to store unique datapoints (sequences)
        self.unique_datapoints_set = set()

        # Create unique sequences for the three splits (training, test, validation)
        for data_type in ["training", "test", "validation"]:
            no_seq = getattr(self, "number_{}_seq".format(data_type))
            data_seqs, data_seq_id = [], []

            for _ in range(no_seq):
                seq, seq_id = self._prepare_single_sequence(image_list)
                data_seqs.append(seq)
                data_seq_id.append(seq_id)

            # Store the `data_seqs` and `data_seq_id` as instance variables
            # e.g. self.training_data = data_seqs
            #      self.training_data_seq_id = data_seq_id
            #      self.training_data_encodings = encodings
            setattr(self, "{}_data".format(data_type), data_seqs)
            setattr(self, "{}_data_seq_id".format(data_type), data_seq_id)

    def _create_dir(self, dir_name: str):
        """
        Delete (if already exists) and Create a directory with the argument as its name,

        Args:
        -   dir_name: name of the directory
        """
        if os.path.exists(dir_name):
            # If the directory exists, prune all the subfolders and files within
            shutil.rmtree(dir_name)
        # Create the directory
        os.makedirs(dir_name)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array, np.array]:
        """
        Fetches the data point (input seq of images, output seq of images, encoding of input_seq) at a given index

        Args:
            idx (int): index
        Returns:
            tuple: (image_seq, image_seq, seq_encoding)
        """
        try:
            data_split = self.dataset[self.mode]
        except:
            raise ValueError(
                "{} not a valid mode. Expecting `training`, `test`, or `validation`.".format(
                    self.mode
                )
            )

        input = data_split["input"][idx]
        pred = data_split["pred"][idx]
        encoding = data_split["encodings"][idx]

        return input, pred, encoding

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: length of the dataset
        """
        try:
            data_split = self.dataset[self.mode]
        except:
            raise ValueError(
                "{} not a valid mode. Expecting `training`, `test`, or `validation`.".format(
                    self.mode
                )
            )

        return len(data_split["input"])

    def collate(self, batch):
        """
        Collate function used to combine data into batch tensor
        """
        batch_input, batch_pred, batch_encoding = [], [], []
        for input, pred, encoding in batch:
            # Flatten the image
            input = input.reshape((input.shape[0], -1))
            pred = pred.reshape((pred.shape[0], -1))

            # Fill in NaN values with average of non-NaN pixels in the patch
            # 1. Obtain global non-NaN mean of each patch
            input_mean = np.nanmean(input, axis=1)
            # 2. If any global mean is NaN, fill with 0
            input_mean = np.nan_to_num(input_mean)
            # 3. Fill NaNs in the patch with the global non-NaN mean
            for r in range(input.shape[0]):
                input[r, :] = np.nan_to_num(input[r, :], nan=input_mean[r])

            # Fill in NaN values with average of non-NaN pixels in the patch
            # 1. Obtain global non-NaN mean of each patch
            pred_mean = np.nanmean(pred, axis=1)
            # 2. If any global mean is NaN, fill with 0
            pred_mean = np.nan_to_num(pred_mean)
            # 3. Fill NaNs in the patch with the global non-NaN mean
            for r in range(pred.shape[0]):
                pred[r, :] = np.nan_to_num(pred[r, :], nan=pred_mean[r])

            batch_input.append(input)
            batch_pred.append(pred)
            batch_encoding.append(encoding)

        batch_input = np.stack(batch_input)
        batch_pred = np.stack(batch_pred)
        batch_encoding = np.stack(batch_encoding)

        batch_input = torch.from_numpy(batch_input).float()
        batch_pred = torch.from_numpy(batch_pred).float()
        batch_encoding = torch.from_numpy(batch_encoding).float()

        return batch_input, batch_pred, batch_encoding
