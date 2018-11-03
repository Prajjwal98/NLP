# -*- coding: utf-8 -*-

from nltk.corpus import wordnet

syno = []  # Synonyms list
anto = []  # Antonyms list

for syn in wordnet.synsets("good"):
    for s in syn.lemmas():
        syno.append(s.name())
    for a in s.antonyms():
        anto.append(a.name())
        
print(set(syno))        
print(set(anto))

