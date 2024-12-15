# -*- coding: utf-8 -*-

# ## this file is to generate artificial error for hindi text
# 
# ### artificial error generation for hindi text contains word level deletion, insertion, substitution and character level deletion, insertion, substitution


import random
import argparse
from tqdm import tqdm
import numpy as np
import aspell


random.seed(84)
np.random.seed(84)


def load_hindi_word_vocabulary(vocabulary_file):
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        return list(set(list(map(lambda x: x.strip(), f.readlines()))))


def insert_hindi_word_and_matra(word):
    matras = ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', 'ँ']
    word = list(word)
    hindi_alphabets = [
    "क", "ख", "ग", "घ", "च", "छ",
    "ज", "झ", "ट", "ठ", "ड",
    "ढ", "ण", "त", "थ", "द",
    "ध", "न", "प", "फ", "ब",
    "भ", "म", "य", "र", "ल", 
    "व", "श", "ष", "स", "ह", 
    "क्ष", "त्र", "ज्ञ"
    ]
    new_word = []
    for i in range(len(word)):
        # deletion of a word
        # with 0.01 probability, skip the word
        if random.random() < 0.01:
            continue

        # insertion of errors
        threshold_prob = 0.06

        if word[i] not in matras:
            # with 0.05 random probability, swap the words
            if random.random() < threshold_prob and (i+1) < len(word)-1:
                word[i], word[i+1] = word[i+1], word[i]
                new_word.append(word[i])
                continue

            # with 0.05 probability, add a random word
            if random.random() < threshold_prob:
                new_word.append(random.choice(hindi_alphabets))
            
        else:
            # with 0.05 probability, add a random matra
            if random.random() < threshold_prob:
                new_word.append(random.choice(matras))
            else:
                new_word.append(word[i])
        
    return ''.join(new_word)




def generate_error_for_word(index, words, error_type, word_vocabulary, aspell_speller):
    word = words[index]
    new_token = ""
    if error_type == 'replace':
        # replace the word with a random word from the vocabulary
        proposals = aspell_speller.suggest(word)
        if len(proposals) > 0:
            new_token = np.random.choice(proposals) 
    elif error_type == 'insert':
        # insert a random word from the vocabulary
        new_token = word + ' ' + random.choice(word_vocabulary)
    elif error_type == 'delete':
        # delete the word
        new_token = ''
    elif error_type == 'swap':
        # swap the word with a random word from the vocabulary
        if index+1 < len(words):
            words[index], words[index+1] = words[index+1], words[index]
            new_token = words[index]
    elif error_type == 'matras':
        new_token = insert_hindi_word_and_matra(words[index])
    else:
        raise ValueError("Invalid error type")
    
    return new_token


def generate_word_level_error(sentence, replace_prob, insert_prob, delete_prob, swap_prob, matras_prob,
                                             word_vocabulary, aspell_speller):
    words = sentence.split()
    mean = 0.2
    std = 0.1
    num_errors = np.random.normal(mean, std)
    # number of errors should be between 0 and 0.15
    lower_limit = mean - std
    upper_limit = mean + std
    while num_errors < lower_limit or num_errors > upper_limit:
        num_errors = np.random.normal(mean, std)
    
    num_errors = int(num_errors * len(words))
    
    
    if num_errors == 0:
        num_errors = 1

    error_indices = random.sample(range(len(words)), num_errors)
    
    new_words = []
    
    for i in range(len(words)):
        if i not in error_indices:
            new_words.append(words[i])
            continue
        # select a random error type
        error_type = np.random.choice(['replace', 'insert', 'delete', 'swap', "matras"],p = [replace_prob, insert_prob, delete_prob, swap_prob, matras_prob] )
        error_word = generate_error_for_word(i, words, error_type, word_vocabulary, aspell_speller)
        new_words.append(error_word)
        
    return ' '.join(new_words)
    


def generate_artificial_errors_from_text_file(src_file , tgt_file ,
                                              replace_prob, 
                                              insert_prob, 
                                              delete_prob, 
                                              swap_prob, 
                                              matras_prob, 
                                              errors_count = 1):
    
    print("Loading word vocabulary")
    word_vocabulary = load_hindi_word_vocabulary("hindi_dictionary/hi_IN.dic")
    # loading aspell speller
    
    # for this to word you should have 
        # apt-get install aspell aspell-doc libaspell-dev aspell-hi
        # pip3 install aspell-python-py3
    # installed in the system
    aspell_speller = aspell.Speller('lang', 'hi')
    with open(src_file, 'r' , encoding='utf-8') as src, \
    open(tgt_file + "_src", 'w' , encoding='utf-8') as tgt_src,\
    open(tgt_file + "_tgt", 'w' , encoding='utf-8') as tgt_tgt:
        lines = list(map(lambda x: x.strip(), src.readlines()))
        for line in tqdm(lines , desc = "Generating artificial errors"):
            line = line.strip().replace(' .' , "।").replace('( ', '(').replace(' )', ')').replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' :', ':').replace(' ;', ';')
            error_line = line
            for count in range(errors_count):
                error_line = generate_word_level_error(error_line, replace_prob,
                                                       insert_prob, 
                                                       delete_prob, 
                                                       swap_prob, 
                                                       matras_prob,
                                                    word_vocabulary, 
                                                    aspell_speller)
    
            tgt_src.write(error_line + '\n')
            tgt_tgt.write(line + "\n")




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--src_file', '-s', type=str, required=True, help='source file path')

    argparser.add_argument('--tgt_file', '-t', type=str, required=True, help='target file path')

    argparser.add_argument('--errors_count', '-ec', type=int, default=1, help='number of errors to be generated in a sentence')
    
    argparser.add_argument('--replace_prob', '-rp', type=float, default=0.3, help='probability of replacing a word')
    
    argparser.add_argument('--insert_prob', '-ip', type=float, default=0.15, help='probability of inserting a word')
    
    argparser.add_argument('--delete_prob', '-dp', type=float, default=0.15, help='probability of deleting a word')
    
    argparser.add_argument('--swap_prob', '-sp', type=float, default=0.1, help='probability of swapping a word')
    
    argparser.add_argument('--matras_prob', '-mp', type=float, default=0.3, help='probability of adding matras to a word')
    

    args = argparser.parse_args()

    random.seed(42)

    generate_artificial_errors_from_text_file(args.src_file, args.tgt_file,
                                              args.replace_prob,
                                                args.insert_prob,
                                                args.delete_prob,
                                                args.swap_prob,
                                                args.matras_prob,
                                              args.errors_count)

