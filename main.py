#!/bin/python3
# -*- coding: utf-8 -*-
from colorama import Fore, Style
from model_learning import *
import os
from colorama import Fore, Style
import argparse
import numpy as np
import intent
import bot
#import multiprocessing


def init_argparse():
    """Create argparse for more power over CLI
    and more customizability
    Return: 
        args : needed to determine values passed by options
    """
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




def load_sub(path, deleted_char, quite):

    """Loading Training Dataset by using translated subtitles
    Params:
        path (str): directory path
        deleted_char (str): characters that should be removed from our input data
        quite (bool): Unix rule of silence
    Return:
        data : an array containing all of sentences in all files
    """

    data = []
    
    # for all files inside PATH
    for i in os.listdir(path):
        file_path = path + i
        try: #UTF8
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

        except: #UTF16
            with open(file_path, "r", encoding="utf16") as file:
                content = file.read()
    
            
        # Delete timestamp and other stuff
        for ch in deleted_char:
            content = content.replace(ch, "")
        lines = content.split("\n")

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
def chat(inp, words, pat_y):
    """ use model prediction to predict the intent
    of input then response with one of the responses from
    the data . IT CAN NOT PRODUCE SENTENCES . it only 
    can understand what have been said.
    Params: 
        inp (string): user message
        words (array): an array containing all of words in the whole data
        pat_y (array): an array containing responses to pat_x
    """
    # Model will return possibilty of intent tags
    re = model.predict([convert2bag(inp, words)])
    # Return the highest possibilty sentence
    #sentence = pat_y[np.argmax(re)]
    return re



if __name__ == "__main__":
    args = init_argparse()

    # Subtitle loading Consts
    quite = args.quite  # If passed to argparser it'll be True. Else it'll be False
    path = "DATA/"
    deleted_char = "0123456789:-_.><?~!@\"';,؟\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # Model Training Consts
    hidden_layers = 5
    e = 1000  # epoch
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

    #data = load_sub(path, deleted_char, quite)
    #cores = multiprocessing.cpu_count()
    #with multiprocessing.Pool(cores) as p:
    #    re = p.map(get_axis,data)
    #re = get_axis(data, q=quite)
    #chatbot_model = train(re, HIDDEN_LAYERS=hidden_layers, epoch=e, batch=b, metric=m)
    re = prepare_assistant()
    words, training, output, fourth = re

    assitant_model = train(re,HIDDEN_LAYERS=hidden_layers, epoch=e, batch=b, metric=m)

    labels = fourth
    print(chat_assistant(input(), assitant_model, words, labels))

    #bot.run(words, pat_y)
