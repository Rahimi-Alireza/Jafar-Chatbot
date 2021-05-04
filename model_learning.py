#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

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


def initalize(data):
    """Create x and y axis based on data
    Data should be provided by main.py
    and based on subtitles translations
    data

    Params:
        data (array): containing sentences given from main.py after proccessing the subs
    """

    try:
        # We saved stemmed words and other stuff in a picke file
        # So we won't process it 2 times
        file = open("data.picke", "rb")  # read binary
        words, labels, training, output = pickle.load(file)
        file.close()
    except:
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

        # Remove duplicates
        words = sorted(list(set(words)))

        train = [] #X
        output = [] #Y
        
        #Y preparations
        for y in pat_y:
            bag_y = convert2bag(y, words)
            output.append(bag_y)
        #X preparation
        for x in pat_x:
            bag_x = convert2bag(x, words)
            train.append(bag_x)

        # Numpy array is required for training
        # Also it's more efficent in Speed, Memory, Performance
        train = np.array(train)
        output = np.array(output)

        file = open("data.picke", "wb")  # Write in binary
        # Save arrays to a pickle file
        pickle.dump((words, labels, training, output), file)
        file.close()

    # Reset all previous data
    tf.reset_default_graph()

    # Declare the input shape ,All train elements have same len
    net = tflearn.input_data(shape=[None, len(train[0])])
    # Each represent a HIDDEN LAYER
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
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
    model.fit(train, output, n_epoch=100, batch_size=10, show_metric=True)
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
