#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:04:14 2019

@author: ishanshrivastava
"""

from __future__ import division

import en_core_web_sm
from os.path import expanduser
from nltk.tag import StanfordNERTagger
import json
import ast
from joblib import dump
import sys
import argparse

def spacy_tag(sent):
    """
    :param sent: list of tokens
    :return: list of labels tagged by spacy
    """
    labels = []
    for word in sent:
        if nlp(word).ents:
            labels.append(nlp(word).ents[0].label_)
        else:
            labels.append("UNK")

    return labels
  
def get_ners1(pairs,st):
    """
    Comment_Ishan:
        :param sent: pairs is a list of tuples. Each tuple has 3 elementes. 1st
        element(0th) and 2nd element of each tuple is a list of words forming a
        sentence. 3rd element is the label.
        :return: list of NER tags by spacy and stanford parser for left and right sentences
    """
    pairs_unzipped = list(zip(*pairs))
    tagsLeft_spacy = [spacy_tag(sent) for sent in pairs_unzipped[0]]
    tagsRight_spacy = [spacy_tag(sent) for sent in pairs_unzipped[1]]
    tagsLeft_stanford = [[item[1] for item in sent] for sent in st.tag_sents(pairs_unzipped[0])]
    tagsRight_stanford = [[item[1] for item in sent] for sent in st.tag_sents(pairs_unzipped[1])]
    return tagsLeft_spacy,tagsRight_spacy,tagsLeft_stanford,tagsRight_stanford

def get_unaryWordFeatures1(pairs,st):
    """
    Comment_Ishan:
        :param sent: pairs is a list of tuples. Each tuple has 3 elementes. 1st
        element(0th) and 2nd element of each tuple is a list of words forming a
        sentence. 3rd element is the label.
        :return: list of tuples. Each tuple contains 2 elements. 1st element[0th]
        contains the left sentence's unary word features and 2nd element contains
        the right sentence's unary word features.
    """
    tagsLeft_spacy,tagsRight_spacy,tagsLeft_stanford,tagsRight_stanford = get_ners1(pairs,st)
    
    print(len(tagsLeft_spacy))
    print(len(tagsRight_spacy))
    print(len(tagsLeft_stanford))
    print(len(tagsRight_stanford))
    # Categories that we care about:
    names = ["PERSON", "ORG", "GPE", "PRODUCT", "ORGANIZATION", "LOCATION"]
    date_time = ["DATE", "TIME"]
    number = ["ORDINAL", "CARDINAL", "QUANTITY", "MONEY", "PERCENT"]
    
    unaryWordFeatures_Left = []
    unaryWordFeatures_Right = []
    count=0
    for spacy_Left, stanford_Left, spacy_Right, stanford_Right in zip(tagsLeft_spacy, tagsLeft_stanford,tagsRight_spacy, tagsRight_stanford):
        final = []
        for i in range(len(spacy_Left)):
            if spacy_Left[i] == 'UNK' and stanford_Left[i] == 'O':
                final.append([0, 0, 0, 1])
            elif spacy_Left[i] != 'UNK':
                if spacy_Left[i] in names:
                    final.append([1, 0, 0, 0])
                elif spacy_Left[i] in date_time:
                    final.append([0, 1, 0, 0])
                elif spacy_Left[i] in number:
                    final.append([0, 0, 1, 0])
            elif stanford_Left[i] != 'O':
                if stanford_Left[i] in names:
                    final.append([1, 0, 0, 0])
                elif stanford_Left[i] in date_time:
                    final.append([0, 1, 0, 0])
                elif stanford_Left[i] in number:
                    final.append([0, 0, 1, 0])

        unaryWordFeatures_Left.append(final)
        final = []
        for i in range(len(spacy_Right)):
            # Appending the values as list for each tag itself as you are running the loop.
            # Else will need to run it again for feature computation
            if spacy_Right[i] == 'UNK' and stanford_Right[i] == 'O':
                final.append([0, 0, 0, 1])
            elif spacy_Right[i] != 'UNK':
                if spacy_Right[i] in names:
                    final.append([1, 0, 0, 0])
                elif spacy_Right[i] in date_time:
                    final.append([0, 1, 0, 0])
                elif spacy_Right[i] in number:
                    final.append([0, 0, 1, 0])
            elif stanford_Right[i] != 'O':
                if stanford_Right[i] in names:
                    final.append([1, 0, 0, 0])
                elif stanford_Right[i] in date_time:
                    final.append([0, 1, 0, 0])
                elif stanford_Right[i] in number:
                    final.append([0, 0, 1, 0])

        unaryWordFeatures_Right.append(final)
        count+=1
    return list(zip(unaryWordFeatures_Left,unaryWordFeatures_Right))
  
def readJsonAndMakeTestPairs(filename):
    json_file = open(filename)
    json_str = json_file.readlines()
    pairs = list()
    for s in json_str:
        oneDict = ast.literal_eval(s)
        if oneDict['gold_label'] == '-':
            continue
        pairs.append(tuple((oneDict['sentence1'].split(' '),oneDict['sentence2'].split(' '),oneDict['gold_label'])))
    return pairs

def createNewListOfDict(pairs,pairs_UF):
    for i in range(len(pairs)):
        pairs[i]["premiseUF"] = pairs_UF[i][0]
        pairs[i]["hypothesisUF"] = pairs_UF[i][1]  
    return pairs

def writeNewTxt(filename,pairs):
    with open(filename, 'w') as f:
        for p in pairs:
            f.writelines(json.dumps(p)+"\n")
            
def readJsonl(filename):
    json_file = open(filename)
    json_str = json_file.readlines()
    pairs = list()
    for s in json_str:
        oneDict = ast.literal_eval(s)
        if oneDict['gold_label'] == '-':
            continue
        pairs.append(oneDict)
    return pairs

def getAndSave(filePathAndName, SAVE_DIR,st ):
    fileName = filePathAndName.split('/')[-1]
    print(f"Reading file : {fileName}")
    tupleL = readJsonAndMakeTestPairs(filePathAndName)
    jsonL = readJsonl(filePathAndName)
    unaryWordFeatures = get_unaryWordFeatures1(tupleL,st)
    print("Processed and saving unary word features for {0}".format(fileName))
    sys.stdout.flush()
    dump(unaryWordFeatures,SAVE_DIR+fileName.split('.')[0]+'_uwf.joblib')
    print("Saved unary word features for {0}".format(fileName))
    writeNewTxt(SAVE_DIR+fileName.split('.')[0]+'_wNERFeat.jsonl',createNewListOfDict(jsonL,unaryWordFeatures))
    print("Processed and saved NER Features for {0}".format(fileName))
 

SAVE_DIR = "/Users/ishanshrivastava/Documents/Masters At ASU/Thesis/cse_576_nli_project-master/src/data/"  
STANFORDN_NER_DIR = "/Users/ishanshrivastava/Documents/Masters At ASU/Thesis/cse_576_nli_project-master/src/external/DecAtt/"

nlp = en_core_web_sm.load()
st = StanfordNERTagger(expanduser(f"{STANFORDN_NER_DIR}stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz"),
                       expanduser(f"{STANFORDN_NER_DIR}stanford-ner-2018-10-16/stanford-ner-3.9.2.jar"),
                       encoding="utf-8", java_options='-mx4G')



parser = argparse.ArgumentParser()

parser.add_argument("--file_path_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The input folder name.")

args = parser.parse_args()
if args.file_path_name:
    getAndSave(args.file_path_name, SAVE_DIR,st )
