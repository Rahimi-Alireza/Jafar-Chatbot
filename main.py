#!/bin/python3
# -*- coding: utf-8 -*-
from ast import Str
from colorama import Fore, Style

# import model_learning
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


if __name__ == "__main__":
    args = init_argparse()

    quite = args.quite  # If passed to argparser it'll be True. Else it'll be False
    path = "DATA/"
    deleted_char = "0123456789:-_.><?~!@\"';,؟\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    if args.path is not None:
        path = args.path
    if args.removingchars is not None:
        deleted_char = args.removingchars

    load_sub(path, deleted_char, quite)
