#!/bin/python3
# -*- coding: utf-8 -*-
from colorama import Fore, Style
import model_learning as ml
import os
from colorama import Fore, Style
import argparse

"""Create argparse for more power over CLI
"""


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Small persian chatbot based on deep learning."
    )
    parser.add_argument(
        "-p", "--path", type=str, help="path to your subtitles in a directory"
    )
    parser.add_argument(
        "-r",
        "--removingchars",
        type=str,
        help="your own custom removing chars from subtitle",
    )
    parser.add_argument(
        "--hiddenlayers", type=int, help="hidden layers for neural network"
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        help="epoch for neural network",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        help="batch size for neural network",
    )
    parser.add_argument(
        "-m",
        "--metric",
        help="Show logs for training powered by tflearn",
        action="store_true",
    )
    parser.add_argument("-q", "--quite", help="Unix silence rule", action="store_true")
    return parser.parse_args()


"""Loading Training Dataset by using translated subtitles
"""


def load_sub(path, deleted_char, quite):
    data = []

    # for all files inside PATH
    for i in os.listdir(path):
        file_path = path + i
        with open(file_path, "r", encoding="utf-8") as file:
            if not quite:
                print("Opening file: ", end="")
                print(Fore.GREEN + file_path)
            content = file.read()
            # Delete timestamp and other stuff
            for ch in deleted_char:
                content = content.replace(ch, "")
            lines = content.split("\n")
        print(Style.RESET_ALL, end="")  # Reset the style of CLI
        for j in lines:
            if (
                j.isspace() or j.startswith("\n") or j == ""
            ):  # DO NOT Add whitespaces to our training data
                continue
            data.append(j)
    if not quite:
        print(
            Fore.BLUE + str(len(data)) + " Sentences loaded"
        )  # Log how many files were loaded
        print(Fore.RED + "Ready to start learning")
    return data


if __name__ == "__main__":
    args = init_argparse()

    # Subtitle loading Consts
    quite = args.quite  # If passed to argparser it'll be True. Else it'll be False
    path = "DATA/"
    deleted_char = "0123456789:-_.><?~!@\"';,؟\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # Model Training Consts
    hidden_layers = 5
    e = 100  # epoch
    b = 10  # batch
    m = args.metric

    # Load params from argparser
    if args.path is not None:
        path = args.path
    if args.removingchars is not None:
        deleted_char = args.removingchars
    if args.epoch is not None:
        e = args.epoch
    if args.batch is not None:
        b = args.batch
    if args.hiddenlayers is not None:
        hidden_layers = args.hiddenlayers

    data = load_sub(path, deleted_char, quite)
    re = ml.get_axis(data, q=quite)
    model = ml.train(re, HIDDEN_LAYERS=hidden_layers, epoch=e, batch=b, metric=m)
