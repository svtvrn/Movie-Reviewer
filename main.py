import numpy as np
import matplotlib as plt
from pathlib import Path

#vocabFile = Path("aclImdb/imdb.vocab").absolute()

#The first "vocabStart" words in the vocabulary will be skipped
vocabStart = 1000
#Every word after "vocabEnd" won't be checked.
vocabEnd = 20000

testDataVocab = open("aclImdb/test/labeledBow.feat","r")
reviewTokens = testDataVocab.readlines();
print(reviewTokens[0])