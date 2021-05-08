#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from colorama import Fore, Style
import tflearn
import tensorflow as tf
import pickle
from tqdm import tqdm
import json
import random

def tokenize(sentence):
    words = sentence.split(" ")
    zamir_malekiat = ["ت", "م", "ش", "تان", "شان", "مان"]
    neshane_jam = ["ان", "ات", "ها", "های"]
    #pasvand = ["بی", "با", "ن", "ب"]
    #pishvand = ["ی"]
    pasvand = []
    pishvand = []
    useless_words = ["رو", "را", "به", "با", "از", "در", "برای"]
    result_words = []
    for w in words:
        useless = False
        for j in useless_words:  # Check if it's useless
            if w == j:
                useless = True
                break
        if useless:
            continue

        # Words in the rest are not USELESS

        # final_word
        fw = w
        for zamir in zamir_malekiat:
            if fw.endswith(zamir):
                fw = fw.replace(zamir, "")
                break

        for neshane in neshane_jam:
            if fw.endswith(neshane):
                fw = fw.replace(neshane, "")
                break

        for pas in pasvand:
            if fw.endswith(pas):
                fw = fw.replace(pas, "")
                # No break , We can have 2 or more pasvands

        for pish in pishvand:
            if fw.startswith(pish):
                fw = fw.replace(pish, "")

        result_words.append(fw)

    return result_words


def convert2bag(sentence, words, already_tokenized=False):
    """Convert the sentence to a bag of words
    Containing 0 or 1 . Each 0 or 1 means
    If a word is in a sentence or not
    So model can predict by the training data

    Param:
        sentence (str): string that is wanted to be converted
        words (array): ALL words of ALL sentences
        already_tokenized (bool): Sometimes this method is called with
            Already tokenized array and it'll prompt a error if we try to tokenize it again"""
    bag = [0 for _ in words]

    if already_tokenized:
        sentence_words = sentence
    else:
        sentence_words = tokenize(sentence)

    for se in sentence_words:
        for i, word in enumerate(words):
            if word == se:
                bag[i] = 1

    return np.array(bag)


def open_proccesed():
    """Initilize words, training, output set from previosly saved
    to a pickle file hardcoded in program named data.pickle
    Return:
        a tuple containing words, training, output"""
    try:
        with open("data.picke", "rb") as file:  # read binary
            words, training, output, pat_y = pickle.load(file)
    except:  # No file saved as data.pickle
        return None

    return (words, training, output, pat_y)


def save_proccessed(tup):
    """Save words, training, output set from currently proccessed data
    if the data gets too much we don't want to do it again
    Params:
        tup (tuple): a tuple containing words, training, output
    """
    with open("data.picke", "wb") as file:  # Write in binary
        pickle.dump(tup, file)


def get_axis(data, q=False):
    """Create x and y axis based on data
    Data should be provided by main.py
    and based on subtitles translations
    data

    Params:
        data (array): containing sentences given from main.py after proccessing the subs
    Return:
        a tuple containing words, training, output
    """
    print(Fore.CYAN + "Opening pickle file ...")
    opened_pickle = open_proccesed()
    if opened_pickle is not None:
        if not q:
            print(Fore.GREEN + "Pickle file loaded . ignoring proccessing")
        return opened_pickle

    if not q:
        print(Fore.RED + "Pickle file doesn't exist")
        print(Style.RESET_ALL + "Proccessing to create training set")

    # Machine learning is a curve you need X and Y
    # In this Case we are CREATING BAGS so it will fit our X
    # Y can be either bag or an intent tag (like previos versions of program)
    # We are using subtitles so we can't use intent tags becuase can't relate a sentence
    # To a specific topic. so in the end We have like len(words)=20000 or maybe greater
    # and every bags is like this and finding relatness between this bags of 20000 words
    # is too hard .SO PLEASE CONTAIN MORE DATA FOR MORE MORE MORE ACCRATE RESULT
    pat_x = []
    pat_y = []
    words = []

    # In this program we suppose that each sentence in subtitles is a response to another
    # So if we have third sentence like A, B, C coming after each other we'll have Question and Answer
    # In program like this:
    # X = [A, B, C]
    # Y = [B, C, none]
    # It's not very accurate model but it can gives better result with much less data provided
    for i, d in enumerate(data):
        # Split the root sentence to smaller words
        ws = tokenize(d)
        # Add all smaller ws to words list
        # Words(array) contains ALL words in ALL sentences
        words.extend(ws)
        pat_x.append(ws)  # Last index = i

        # If it's the first sentence we don't want to add it to our Y
        # OR our question, answer will be like this
        # [A, B, C]
        # [A, B, C]
        if i == 0:
            continue

        pat_y.append(ws)  # Last index = i-1

    pat_y.append("")  # To make pat_x and pat_y have same index

    if not q:
        print(Fore.GREEN + str(len(words)) + " words loaded ...")
    # Remove duplicates
    words = sorted(list(set(words)))

    train = []  # X
    output = []  # Y

    # Y preparations
    for y in tqdm(pat_y):
        bag_y = convert2bag(y, words, already_tokenized=True)
        output.append(bag_y)

    print(Fore.RED)
    # X preparation
    for x in tqdm(pat_x):
        bag_x = convert2bag(x, words, already_tokenized=True)
        train.append(bag_x)
    if not q:
        print(Fore.CYAN + str(len(train)) + " training set loaded")

    # Numpy array is required for training
    # Also it's more efficent in Speed, Memory, Performance
    train = np.array(train)
    output = np.array(output)

    re = (words, train, output, pat_y)

    if not q:
        print(Fore.CYAN + "Saving to pickle file for second run")
    save_proccessed(re)  # Save for the second run optimization
    return re

def prepare_assistant(q=False):
    
    with open("intent.json", encoding="UTF-8") as file:
        data = json.load(file)
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Remove duplicates
    words = sorted(list(set(words)))
    labels = sorted(labels)

    train = []  # X
    output = []  # Y

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in tqdm(enumerate(docs_x)):
        bag = []

        wrds = [w for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        train.append(bag)
        output.append(output_row)

    if not q:
        print(Fore.CYAN + str(len(train)) + " training set loaded")

    train = np.array(train)
    output = np.array(output)

    re = (words, train, output, labels)
    return re

def train(re, HIDDEN_LAYERS=5, epoch=100, batch=10, metric=False):
    """Training based on train,output we had. It'll try to make connection between
    bag_x and bag_y .Remember that this model just understands what you said FOR NOW
    and returns a sentence from data and CAN NOT CREATE SENTENCES BY ITSELF
    Params:
        re (tuple): a tuple containing words, training, output
        HIDDEN_LAYERS (int): a number that specify hidden layers for our model
        epoch (int) : a number shows epoch for model
        batch (int) : a number shows batch size for model
        metric (boll) : shows log or not
    Return:
        model : the model trained or opened
    """

    # TODO CREATE SENTENCES BY ITSELF
    words, train, output, pat_y = re
    # Reset all previous data
    tf.compat.v1.reset_default_graph()

    # Declare the input shape ,All train elements have same len
    net = tflearn.input_data(shape=[None, len(train[0])])

    # Loop through for HIDDEN_LAYERS
    for i in range(HIDDEN_LAYERS):
        net = tflearn.fully_connected(net, 8)  # Each represent a HIDDEN LAYER
    # Softmax represent possiblity like 0.54
    # This is the output layer
    # We use the highest possibility answer in json file
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

        # Pass the data
        # This model wants to figure out the relate of usage of some words
    model.fit(train, output, n_epoch=epoch, batch_size=batch, show_metric=metric)
    return model




def chat_assistant(inp, model, words, labels, data):
    results = model.predict([convert2bag(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)