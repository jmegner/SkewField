#! /usr/bin/python
"""
initial author:   Kitty Yang
initial date:     2011-07-15

note: This is the first numbered version of KittySupplements.py and addresses
the issues raised in monster code review

note: has a reduced function for each class except SkewFieldLetter that takes
a relations array as a parameter. This may not be the most efficient way to
structure things, but it will compile. reduceLetter happens at the word level,
which seems kind of odd.
"""


FileVersion = "0.01"


import sys
import getopt
import re
import collections

from SkewField import *

def firstOfAlpha(self, alpha):
    for letter in sorted(self.letterCtr.keys()):
        if letter.alpha == alpha:
            return (letter.sub, self.letterCtr[letter])
    return None

def lastOfAlpha(self, alpha):
    for letter in reversed(sorted(self.letterCtr.keys())):
        if letter.alpha == alpha:
            return (letter.sub, self.letterCtr[letter])
    return None

def firstAbnormalLetter(self, relation):
    for letter in sorted(self.letterCtr.keys()):
        alpha = letter.alpha
        power = self.letterCtr[letter]
        subscript = letter.sub
        (minSubscript, minPower) = relation[alpha].firstOfAlpha(alpha)
        (maxSubscript, maxPower) = relation[alpha].lastOfAlpha(alpha)

        if not(
            (subscript < minSubscript and p in range (abs(minPower))) or
            (subscript > maxSubscript and p in range (abs(maxPower))) or
            (subscript in range (minSubscript, maxSubscript))):

            return letter

        return None

def reducedAtLetter(self, letter, relation):
    alpha = letter.alpha
    power = self.letterCtr[letter]
    subscript = letter.sub
    (minSubscript, minPower) = relation[alpha].firstOfAlpha(alpha)
    (maxSubscript, maxPower) = relation[alpha].lastOfAlpha(alpha)

    if(subscript < minSubscript):
        increment = subscript - minSubscript
        if minPower < 0:
            exponent = 1
        else:
            exponent = 0
        exponent += power // minPower
        newRelation = relation[alpha].increasedSubs(increment)
        newRelation = newRelation.raisedTo(-exponent)
        return self.times(newRelation)

    if(subscript >= maxSubscript):
        increment = subscript - maxSubscript
        if maxPower < 0:
            exponent = 1
        else:
            exponent = 0
        exponent += power // maxPower
        newRelation = relation[alpha].increasedSubs(increment)
        newRelation = newRelation.raisedTo(-exponent)
        return self.times(newRelation)

def reduced(self, relation):
    abnormalLetter = self.firstAbnormalLetter(relation)
    if abnormalLetter is None:
        return self
    else:
        return self.reducedAtLetter(abnormalLetter, relation).reduced(relation)

def raisedTo(self, power):
    newWord = SkewFieldWord()

    for letter, letterPower in self.letterCtr.items():
        newWord.letterCtr[letter.deepcopy()] = letterPower*power

    return newWord

def reducedSentence(self, relations):
    newWords = []
    for word in self.wordCtr.keys():
        newWords.append(word.reduced(relations))
    return SkewFieldSentence(newWords)
      
def reducedMonomial(self, relations):
    return SkewFieldMonomial(self.numer.reduced(relations),
                             self.denom.reduced(relations), self.tpower)

def reducedPolynomial(self, relations):
    newMonos = []
    for mono in self.monoDict.values():
        newMonos.append(mono.reduced(relations))
    return SkewFieldPolynomial(newMonos)



################################################################################
# MAIN
################################################################################

def main(argv=None):


    print("###################################################################")
    print("# FileVersion = " + FileVersion)
    print("###################################################################")
    print("")
    print("")

    
    SkewFieldWord.firstOfAlpha = firstOfAlpha
    SkewFieldWord.lastOfAlpha = lastOfAlpha    
    SkewFieldWord.firstAbnormalLetter = firstAbnormalLetter
    SkewFieldWord.reducedAtLetter = reducedAtLetter
    SkewFieldWord.reduced = reduced
    SkewFieldWord.raisedTo = raisedTo
    SkewFieldSentence.reduced = reducedSentence
    SkewFieldMonomial.reduced = reducedMonomial
    SkewFieldPolynomial.reduced = reducedPolynomial

    relations = []
    file = open("relation3_1.txt")

    for line in file:
        line = line[0:-1]
        relations.append(SkewFieldWord(line))

    print(relations)

    ltrs1 = [
        SkewFieldLetter("a", 0),
        SkewFieldLetter("b", 1),
        SkewFieldLetter("b", 1),
    ]

    wrd1 = SkewFieldWord(ltrs1)

    print(wrd1)
    
    print(wrd1.reduced(relations))

    wrd2Str = "a_3^1 * a_-1^2 * b_3^2"
    wrd2 = SkewFieldWord(wrd2Str)

    snt1Str = "1 * a_0^1 + 1 * b_0^1"
    snt1 = SkewFieldWord("a_0^1").plus(SkewFieldWord("b_0^1"))
    print("wrdA + wrdB = snt1 = " + str(snt1))

    print(snt1.reduced(relations))

    snt3Str = "1 * a_0^1 + 1 * b_0^1"
    snt3 = SkewFieldSentence(snt3Str)


    mono1Str = "(" + str(snt1) + ") / (" + str(snt3) + ") * T^-2"
    mono1 = SkewFieldMonomial(snt1, snt3, -2)

    print(mono1)

    mono2 = mono1.times(mono1.mInv())

    poly1 = SkewFieldPolynomial([mono1, mono2, mono1])
    print("poly1 = " + str(poly1))

    poly2 = mono1.plus(mono2)
    print(poly2)

    print(poly2.reduced(relations))

    mono3str = "(1 * a_0^1 + 2 * b_3^2) / (2 * b_0^-2*b_1^1) * T^1"
    mono4str = "(1 * a_0^1 * b_0^1) / (1) * T^0"

    mono3 = SkewFieldMonomial(mono3str)
    mono4 = SkewFieldMonomial(mono4str)

    poly3 = SkewFieldPolynomial([mono3, mono4])
    poly4 = SkewFieldPolynomial([mono3, mono4, mono3, mono4])

    print(mono3.times(mono4))    

    
        
if __name__ == "__main__":
    sys.exit(main())

