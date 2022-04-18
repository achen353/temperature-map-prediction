#!/usr/bin/env/python3


"""
About
-----
This python file contains the class for Dataloader of MODIS satellite images for the sequence-to-sequence prediction task.

Classes
-------
MODISDataLoader

Functions
---------
__init__
sort_images_by_date
prepare_data_sequence
prepare_training_data
prepare_validation_data
__getitem__
__len__

Variables
---------
"""

# Meta-data.
__author__ = "Sourish Gunesh Dhekane"
__copyright__ = "FreeToUse"
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Sourish Gunesh Dhekane"
__email__ = "sourish.dhekane@gatech.edu"
__status__ = ""

# Dependencies.
import glob
import random
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
import torch.utils.data as data

# Main.
if __name__ == "__main__":
    pass


class MODISDataLoader(data.Dataset):
    """
    Class for loading MODIS Images
    """

    def __init__(
        self,
        patch_dim: int = 50,
        input_seq_len: int = 5,
        prediction_seq_len: int = 1,
        number_training_seq: int = 10,
        number_validation_seq: int = 2,
        mode: str = "train",
        modis_img_path: str = "./datasets/MOD11A2",
    ):
        self.patch_dim = patch_dim
        self.input_seq_len = input_seq_len
        self.prediction_seq_len = prediction_seq_len
        self.number_training_seq = number_training_seq
        self.number_validation_seq = number_validation_seq
        self.mode = mode
        self.modis_image_path = modis_img_path

        self.image_list = self.sort_images_by_date()
        self.training_data = self.prepare_training_data(self.image_list)
        self.validation_data = self.prepare_validation_data(self.image_list)

    def sort_images_by_date(self) -> List[str]:
        """
        Collect image names in a list, sort that list, and return it

        Returns:
        -   list[str]: a list of image names in a sorted order
        """
        image_list = glob.glob("./MOD11A1_Dataset/*.tif")  # Create list of images
        new_image_list = [
            x.replace("./", "").replace("MOD11A1_Dataset/", "").replace(".tif", "")
            for x in image_list
        ]  # Extract filenames
        new_image_list = [
            x for x in new_image_list if "(1)" not in x
        ]  # Remove duplicates
        new_image_list.sort(
            key=lambda x: datetime.strptime(x, "%Y_%m_%d")
        )  # Sort the list based on dates

        return new_image_list

    def prepare_data_sequence(self, new_image_list: str) -> Tuple[List[str], List[str]]:
        """
        Prepare one data point of the sequence-to-sequence prediction problem

        Args:
        -   new_image_list: the sorted list of ALL the image names in the dataset as per their date
        Returns:
        -   data_sequence: Tuple[List[str], List[str]]: a tuple of lists- first list being the input sequence of image
                          names and second list being the output sequence of image names
        """
        no_of_imgs = len(new_image_list)
        seq_starting_point = random.randint(
            0, no_of_imgs - (self.input_seq_len + self.prediction_seq_len + 1)
        )
        input_seq = new_image_list[
            seq_starting_point : seq_starting_point + self.input_seq_len
        ]
        pred_seq = new_image_list[
            seq_starting_point
            + self.input_seq_len : seq_starting_point
            + self.input_seq_len
            + self.prediction_seq_len
        ]

        return (input_seq, pred_seq)

    def prepare_training_data(
        self, new_image_list: str
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Prepare training dataset for the sequence-to-sequence prediction problem

        Args:
        -   new_image_list: the sorted list of ALL the image names in the dataset as per their date
        Returns:
        -   training_dataset: List[Tuple[List[str], List[str]]]: A list of tuples of lists- Each tuple represents a single data point
                              A tuple contains 2 lists: first list being the input sequence of image names
                              and second list being the output sequence of image names
        """
        training_data = [
            self.prepare_data_sequence(new_image_list)
            for _ in range(self.number_training_seq)
        ]

        return training_data

    def prepare_validation_data(
        self, new_image_list: str
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Prepare Validation dataset for the sequence-to-sequence prediction problem

        Args:
        -   new_image_list: the sorted list of ALL the image names in the dataset as per their date
        Returns:
        -   validation_dataset: List[Tuple[List[str], List[str]]]: A list of tuples of lists- Each tuple represents a single data point
                              A tuple contains 2 lists: first list being the input sequence of image names
                              and second list being the output sequence of image names
        """
        validation_data = [
            self.prepare_data_sequence(new_image_list)
            for _ in range(self.number_validation_seq)
        ]

        return validation_data

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """
        Fetches the data point (input seq of images, output seq of images) at a given index

        Args:
            index (int): Index
        Returns:
            tuple: (image_seq, image_seq)
        """
        if self.mode == "train":
            inp_seq, pred_seq = self.training_data[index]
        else:
            inp_seq, pred_seq = self.validation_data[index]

        path = self.modis_image_path + "/" + inp_seq[0] + ".tif"
        modis_img = cv2.imread(path, -1)
        modis_img_length = np.array(modis_img).shape[0]
        modis_img_breadth = np.array(modis_img).shape[1]
        starting_point_x = random.randint(0, modis_img_length - (self.patch_dim + 1))
        starting_point_y = random.randint(0, modis_img_breadth - (self.patch_dim + 1))

        input_modis_seq = np.zeros((self.input_seq_len, self.patch_dim, self.patch_dim))
        counter = 0
        for img_str in inp_seq:
            path = self.modis_image_path + "/" + img_str + ".tif"
            modis_img = cv2.imread(path, -1)
            modis_img = modis_img[
                starting_point_x : starting_point_x + self.patch_dim,
                starting_point_y : starting_point_y + self.patch_dim,
            ]
            input_modis_seq[counter, :, :] = modis_img
            counter = counter + 1

        pred_modis_seq = np.zeros(
            (self.prediction_seq_len, self.patch_dim, self.patch_dim)
        )
        counter = 0
        for img_str in pred_seq:
            path = self.modis_image_path + "/" + img_str + ".tif"
            modis_img = cv2.imread(path, -1)
            modis_img = modis_img[
                starting_point_x : starting_point_x + self.patch_dim,
                starting_point_y : starting_point_y + self.patch_dim,
            ]
            pred_modis_seq[counter, :, :] = modis_img
            counter = counter + 1

        return (input_modis_seq, pred_modis_seq)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: length of the dataset
        """
        if self.mode == "train":
            l = len(self.training_data)
        else:
            l = len(self.validation_data)
        return l
