#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from colorama import Fore, Style
import tflearn
import tensorflow as tf
import random
import json
import pickle


def tokenize(sentence):
    words = sentence.split(" ")
    zamir_malekiat = ["ت", "م", "ش", "تان", "شان", "مان"]
    neshane_jam = ["ان", "ات", "ها", "های"]
    pasvand = ["بی", "با", "ن", "ب"]
    pishvand = ["ی"]
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
            if fw.starswith(pish):
                fw = fw.replace(pish, "")

        result_words.append(fw)

    return result_words


def convert2bag(sentence, words):
    """Convert the sentence to a bag of words
    Containing 0 or 1 . Each 0 or 1 means
    If a word is in a sentence or not
    So model can predict by the training data

    Param:
        sentence (str): string that is wanted to be converted
        words (array): ALL words of ALL sentences"""
    bag = [0 for _ in words]

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
            words, training, output = pickle.load(file)
    except:  # No file saved as data.pickle
        return None

    return (words, training, output)


def save_proccessed(tup):
    """Save words, training, output set from currently proccessed data
    if the data gets too much we don't want to do it again
    Params:
        tup (tuple): a tuple containing words, training, output
    """
    with open("data.picke", "wb") as file:  # Write in binary
        pickle.dump(tup, file)


def get_axis(data, quite=False):
    """Create x and y axis based on data
    Data should be provided by main.py
    and based on subtitles translations
    data

    Params:
        data (array): containing sentences given from main.py after proccessing the subs
    Return:
        a tuple containing words, training, output
    """
    print(Fore.CYAN + 'Opening pickle file ...')
    opened_pickle = open_proccesed()
    if opened_pickle is not None:
        if not quite: print(Fore.GREEN + 'Pickle file loaded . ignoring proccessing')
        return opened_pickle
        
    if not quite: 
        print(Fore.RED + 'Pickle file doesn\'t exist')
        print(Style.RESET_ALL + 'Proccessing to create training set')

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

    if not quite: print(Fore.GREEN + str(len(words) + ' words loaded ...'))
    # Remove duplicates
    words = sorted(list(set(words)))

    train = []  # X
    output = []  # Y

    # Y preparations
    for y in pat_y:
        bag_y = convert2bag(y, words)
        output.append(bag_y)
    # X preparation
    for x in pat_x:
        bag_x = convert2bag(x, words)
        train.append(bag_x)
    if not quite: print(Fore.CYAN + str(len(train) + ' training set loaded'))

    # Numpy array is required for training
    # Also it's more efficent in Speed, Memory, Performance
    train = np.array(train)
    output = np.array(output)

    re = (words, train, output)

    if not quite: print(Fore.CYAN +'Saving to pickle file for second run'))
    save_proccessed(re)  # Save for the second run optimization
    return re


def train(re, HIDDEN_LAYERS=5, epoch=100, batch=10, metric=True):
    """Training based on train,output we had. It'll try to make connection between
    bag_x and bag_y .Remember that this model just understands what you said FOR NOW
    and returns a sentence from data and CAN NOT CREATE SENTENCES BY ITSELF
    Params:
        re (tuple): a tuple containing words, training, output
        HIDDEN_LAYERS (int): a number that specify hidden layers for our model
        epoch (int) : a number shows epoch for model
        batch (int) : a number shows batch size for model
        metric (boll) : shows log or not
    """

    # TODO CREATE SENTENCES BY ITSELF
    words, train, output = re
    # Reset all previous data
    tf.reset_default_graph()

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

    try:
        # If we have the model, Load it
        model.load("modedl.tflearn")
    except:
        # Pass the data
        # This model wants to figure out the relate of usage of some words
        # and intent tags
        model.fit(train, output, n_epoch=epoch, batch_size=batch, show_metric=metric)
        model.save("model.tflearn")


def chat():
    print("آماده ایم آماده")
    while True:
        inp = input("شما: ").lower()
        if inp.lower() == "برو بیرون":
            break

        # Model will return possibilty of intent tags
        re = model.predict([convert2bag(inp, words)])
        # Return the highest possibilty intent tag
        intent_tag = labels[np.argmax(re)]

        for i in data["intents"]:  # Loop through all of tags
            if i["tag"] == intent_tag:
                responses = i["responses"]
        print(random.choice(responses))  # Choose a random response from bot
