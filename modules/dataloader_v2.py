#!/usr/bin/env/python3


"""
About
-----
Script for the Dataloader of the MODIS images

Classes
-------
MODISDataLoader

Functions
---------
__init__
sort_images_by_date
prepare_data_sequence
prepare_train_test_validation_data
create_dir
save_data_into_dir
save_train_test_validation_data
prepare_encodings
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
__version__ = "2.0"
__maintainer__ = "Sourish Gunesh Dhekane"
__email__ = "sourish.dhekane@gatech.edu"
__status__ = ""

import glob
# Dependencies.
import os
import random
import shutil
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
        number_test_seq: int = 2,
        mode: str = "train",
        modis_img_path: str = "./MOD11A1_Dataset",
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
        self.training_data_seq_id = None
        self.test_data_seq_id = None
        self.validation_data_seq_id = None
        self.training_data_encodings = None
        self.test_data_encodings = None
        self.validation_data_encodings = None

        self.image_list = self.sort_images_by_date()
        (
            self.training_data,
            self.test_data,
            self.validation_data,
        ) = self.prepare_train_test_validation_data(self.image_list)
        self.save_train_test_validation_data(
            self.training_data, self.test_data, self.validation_data
        )
        self.prepare_encodings()

    def sort_images_by_date(self) -> List[str]:
        """
        Collect image names in a list, sort that list, and return it

        Returns:
        -   list[str]: a list of image names in a sorted order
        """
        new_image_list = [
            os.path.basename(x).replace(".tif", "")
            for x in glob.glob(self.modis_image_path + "/*.tif")
        ]  # Create list of images
        new_image_list = [
            x for x in new_image_list if "(1)" not in x
        ]  # Remove duplicates
        new_image_list.sort(
            key=lambda x: datetime.strptime(x, "%Y_%m_%d")
        )  # Sort the list based on dates

        return new_image_list

    def prepare_data_sequence(
        self, new_image_list: List[str]
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
        input_seq = None
        pred_seq = None
        seq_identifiers = None

        no_of_imgs = len(new_image_list)
        flag = 0

        while flag == 0:
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

            path = self.modis_image_path + "/" + input_seq[0] + ".tif"
            modis_img = cv2.imread(path, -1)
            modis_img_length = np.array(modis_img).shape[0]
            modis_img_breadth = np.array(modis_img).shape[1]
            starting_point_x = random.randint(
                0, modis_img_length - (self.patch_dim + 1)
            )
            starting_point_y = random.randint(
                0, modis_img_breadth - (self.patch_dim + 1)
            )

            seq_identifiers = (seq_starting_point, starting_point_x, starting_point_y)
            if seq_identifiers not in self.unique_datapoints_set:
                self.unique_datapoints_set.add(seq_identifiers)
                flag = 1

        return ((input_seq, pred_seq), seq_identifiers)

    def prepare_train_test_validation_data(
        self, new_image_list: List[str]
    ) -> Tuple[
        List[Tuple[List[str], List[str]]],
        List[Tuple[List[str], List[str]]],
        List[Tuple[List[str], List[str]]],
    ]:
        """
        Prepare training dataset for the sequence-to-sequence prediction problem

        Args:
        -   new_image_list: the sorted list of ALL the image names in the dataset as per their date
        Returns:
            Tuple(training_data, test_data, validation_data)
        -   training_dataset: List[Tuple[List[str], List[str]]]: A list of tuples of lists- Each tuple represents a single data point
                              A tuple contains 2 lists: first list being the input sequence of image names
                              and second list being the output sequence of image names
        -   test_dataset: List[Tuple[List[str], List[str]]]: A list of tuples of lists- Each tuple represents a single data point
                              A tuple contains 2 lists: first list being the input sequence of image names
                              and second list being the output sequence of image names
        -   validation_dataset: List[Tuple[List[str], List[str]]]: A list of tuples of lists- Each tuple represents a single data point
                              A tuple contains 2 lists: first list being the input sequence of image names
                              and second list being the output sequence of image names
        """
        self.unique_datapoints_set = set()
        training_data = []
        training_data_seq_id = []
        for _ in range(self.number_training_seq):
            data_seq, data_seq_id = self.prepare_data_sequence(new_image_list)
            training_data.append(data_seq)
            training_data_seq_id.append(data_seq_id)

        test_data = []
        test_data_seq_id = []
        for _ in range(self.number_test_seq):
            data_seq, data_seq_id = self.prepare_data_sequence(new_image_list)
            test_data.append(data_seq)
            test_data_seq_id.append(data_seq_id)

        validation_data = []
        validation_data_seq_id = []
        for _ in range(self.number_validation_seq):
            data_seq, data_seq_id = self.prepare_data_sequence(new_image_list)
            validation_data.append(data_seq)
            validation_data_seq_id.append(data_seq_id)

        self.training_data_seq_id = training_data_seq_id
        self.test_data_seq_id = test_data_seq_id
        self.validation_data_seq_id = validation_data_seq_id

        return training_data, test_data, validation_data

    def create_dir(self, dir_name: str):
        """
        Delete (if already exists) and Create a directory with the argument as its name,

        Args:
        -   dir_name: name of the directory
        """
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)

    def save_data_into_dir(
        self, data: List[Tuple[List[str], List[str]]], data_type: str
    ):
        """
        Save the data into the directory structure as described in the docstring of save_train_test_validation_data()

        Args:
        -   data: training/test/validation data as output by prepare_train_test_validation_data()
        """

        data_point_name = None

        for i, data_point in enumerate(data):
            inp_seq = data_point[0]
            pred_seq = data_point[1]
            if data_type == "training":
                data_point_name = self.training_data_seq_id[i]
            elif data_type == "test":
                data_point_name = self.test_data_seq_id[i]
            elif data_type == "validation":
                data_point_name = self.validation_data_seq_id[i]
            data_point_name_str = (
                str(data_point_name[0])
                + "_"
                + str(data_point_name[1])
                + "_"
                + str(data_point_name[2])
            )
            if data_type == "training":
                self.create_dir("./training_dataset/" + str(i))
            elif data_type == "test":
                self.create_dir("./test_dataset/" + str(i))
            elif data_type == "validation":
                self.create_dir("./validation_dataset/" + str(i))

            input_modis_seq = np.zeros(
                (self.input_seq_len, self.patch_dim, self.patch_dim)
            )
            for j, img_str in enumerate(inp_seq):
                path = self.modis_image_path + "/" + img_str + ".tif"
                modis_img = cv2.imread(path, -1)
                modis_img = modis_img[
                    int(data_point_name[1]) : int(data_point_name[1]) + self.patch_dim,
                    int(data_point_name[2]) : int(data_point_name[2]) + self.patch_dim,
                ]
                input_modis_seq[j, :, :] = modis_img

            pred_modis_seq = np.zeros(
                (self.prediction_seq_len, self.patch_dim, self.patch_dim)
            )
            for j, img_str in enumerate(pred_seq):
                path = self.modis_image_path + "/" + img_str + ".tif"
                modis_img = cv2.imread(path, -1)
                modis_img = modis_img[
                    int(data_point_name[1]) : int(data_point_name[1]) + self.patch_dim,
                    int(data_point_name[2]) : int(data_point_name[2]) + self.patch_dim,
                ]
                pred_modis_seq[j, :, :] = modis_img

            if data_type == "training":
                np.save(
                    os.path.join(
                        "./training_dataset/" + str(i),
                        "input_" + data_point_name_str + ".npy",
                    ),
                    input_modis_seq,
                )
                np.save(
                    os.path.join(
                        "./training_dataset/" + str(i),
                        "pred_" + data_point_name_str + ".npy",
                    ),
                    pred_modis_seq,
                )
            elif data_type == "test":
                np.save(
                    os.path.join(
                        "./test_dataset/" + str(i),
                        "input_" + data_point_name_str + ".npy",
                    ),
                    input_modis_seq,
                )
                np.save(
                    os.path.join(
                        "./test_dataset/" + str(i),
                        "pred_" + data_point_name_str + ".npy",
                    ),
                    pred_modis_seq,
                )
            elif data_type == "validation":
                np.save(
                    os.path.join(
                        "./validation_dataset/" + str(i),
                        "input_" + data_point_name_str + ".npy",
                    ),
                    input_modis_seq,
                )
                np.save(
                    os.path.join(
                        "./validation_dataset/" + str(i),
                        "pred_" + data_point_name_str + ".npy",
                    ),
                    pred_modis_seq,
                )

    def save_train_test_validation_data(
        self,
        training_data: List[Tuple[List[str], List[str]]],
        test_data: List[Tuple[List[str], List[str]]],
        validation_data: List[Tuple[List[str], List[str]]],
    ):
        """
        Save the Training, Test, and Validation data into respective directories-
        The directory structure is as follows:
        training_dataset -> 0,1,2,...,n (where n is number of training data sequences) -> for i: input_<image-ID> and
        pred_<image-ID>, which represent the input and prediction image sequences.
        """

        self.create_dir("training_dataset")
        self.create_dir("test_dataset")
        self.create_dir("validation_dataset")

        self.save_data_into_dir(training_data, "training")
        self.save_data_into_dir(test_data, "test")
        self.save_data_into_dir(validation_data, "validation")

    def prepare_encodings(self):
        """
        Prepare encodings necessary for the transformer model based on the timestamp of the image
        """

        training_encodings = [float(x[0] / 365) for x in self.training_data_seq_id]
        test_encodings = [float(x[0] / 365) for x in self.test_data_seq_id]
        validation_encodings = [float(x[0] / 365) for x in self.validation_data_seq_id]

        self.training_data_encodings = training_encodings
        self.test_data_encodings = test_encodings
        self.validation_data_encodings = validation_encodings

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """
        Fetches the data point (input seq of images, output seq of images) at a given index

        Args:
            index (int): Index
        Returns:
            tuple: (image_seq, image_seq)
        """
        inp_seq = None
        pred_seq = None
        starting_point_x = None
        starting_point_y = None

        if self.mode == "train":
            inp_seq, pred_seq = self.training_data[index]
        elif self.mode == "validation":
            inp_seq, pred_seq = self.validation_data[index]
        elif self.mode == "test":
            inp_seq, pred_seq = self.test_data[index]

        if self.mode == "train":
            starting_point_x = self.training_data_seq_id[index][1]
            starting_point_y = self.training_data_seq_id[index][2]
        elif self.mode == "validation":
            starting_point_x = self.validation_data_seq_id[index][1]
            starting_point_y = self.validation_data_seq_id[index][2]
        elif self.mode == "test":
            starting_point_x = self.test_data_seq_id[index][1]
            starting_point_y = self.test_data_seq_id[index][2]

        input_modis_seq = np.zeros((self.input_seq_len, self.patch_dim, self.patch_dim))
        for i, img_str in enumerate(inp_seq):
            path = self.modis_image_path + "/" + img_str + ".tif"
            modis_img = cv2.imread(path, -1)
            modis_img = modis_img[
                starting_point_x : starting_point_x + self.patch_dim,
                starting_point_y : starting_point_y + self.patch_dim,
            ]
            input_modis_seq[i, :, :] = modis_img

        pred_modis_seq = np.zeros(
            (self.prediction_seq_len, self.patch_dim, self.patch_dim)
        )
        for i, img_str in enumerate(pred_seq):
            path = self.modis_image_path + "/" + img_str + ".tif"
            modis_img = cv2.imread(path, -1)
            modis_img = modis_img[
                starting_point_x : starting_point_x + self.patch_dim,
                starting_point_y : starting_point_y + self.patch_dim,
            ]
            pred_modis_seq[i, :, :] = modis_img

        return (input_modis_seq, pred_modis_seq)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: length of the dataset
        """
        l = None

        if self.mode == "train":
            l = len(self.training_data)
        elif self.mode == "validation":
            l = len(self.validation_data)
        elif self.mode == "test":
            l = len(self.test_data)

        return l
