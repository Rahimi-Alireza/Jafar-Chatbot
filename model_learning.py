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


"""Create x and y axis based on data
Data should be provided by main.py
and based on subtitles translations"""
def initalize(data):
    try:
        # We saved stemmed words and other stuff in a picke file
        # So we won't process it 2 times
        file = open("data.picke", "rb")  # read binary
        words, labels, training, output = pickle.load(file)
        file.close()
    except:
        words = []
        labels = []
        pat = []
        pat_intent = []

        for i in data["intents"]:
            for pattern in i["pattern"]:
                # Split the root sentence to smaller words
                ws = tokenize(pattern)
                # Add all smaller ws to words list
                words.extend(ws)

                pat.append(ws)
                pat_intent.append(intent["tag"])

                if not i["tag"] in labels:
                    labels.append(i["tag"])

        #Remove duplicates
        words = sorted(list(set(words)))

        labels = sorted(labels)

        train = []
        output = []
        example_row = [0 for i in labels]

        for i in range(len(pat)):
            bag = []
            sentence = list(pat[i])

            for w in words:  # Loop through all of the words
                # Create a word detection system with 0 and 1
                if w in sentence:
                    bag.append(1)
                else:
                    bag.append(0)

            # Create a copy of example row
            output_row = list(example_row)
            # Change 0 to 1 where intent of the pattern is right
            output_row[labels.index(pat_intent[i])] = 1

            train.append(bag)
            output.append(output_row)

        # Numpy array is required for training
        # Also it's more efficent in speed, memory, performance
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


""" Convert the sentence to a bag of words
 Containing 0 or 1 . Each 0 or 1 means
 If a word is in a sentence or not
 So model can predict by the training data"""
def convert2bag(sentence, words):
    bag = [0 for _ in words]

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(_.lower()) for _ in sentence_words]

    for se in sentence_words:
        for i in range(len(words)):
            if words[i] == se:
                bag[i] = 1

    return np.array(bag)


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
