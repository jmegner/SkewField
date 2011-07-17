#! /usr/bin/python
"""
initial author:   Kitty Yang
initial date:     2011-07-15

note: grand polynomial division works!!!

"""


FileVersion = "0.02"


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

def quotient(self, denominator):
    numerator = self.deepcopy()
    result = []

    while(numerator.degree() >= denominator.degree()
          and not numerator.isZero()):
        leadDenominator = denominator.monoDict.get(
            denominator.degree(),
            SkewFieldMonomial.zero())
        leadNumerator = numerator.monoDict.get(
            numerator.degree(),
            SkewFieldMonomial.zero())
        mono = leadDenominator.mInv().times(leadNumerator)
        result.append(mono)
        product = denominator.times(mono.asPoly())
        numerator = numerator.plus(product.aInv())
        print(numerator)
        print(denominator)

    return SkewFieldPolynomial(result)




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
    SkewFieldPolynomial.quo = quotient

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

    #test reduce words
    print(wrd1)
    print(wrd1.reduced(relations))

    wrd2Str = "a_3^1 * a_-1^2 * b_3^2"
    wrd2 = SkewFieldWord(wrd2Str)

    snt1Str = "1 * a_0^1 + 1 * b_0^1"
    snt1 = SkewFieldWord("a_0^1").plus(SkewFieldWord("b_0^1"))
    print("wrdA + wrdB = snt1 = " + str(snt1))

    #test reduce sentences
    snt2 = snt1.reduced(relations)
    print(snt2)
    print(snt2.increasedSubs(5))

    snt3Str = "1 * a_0^1 + 1 * b_1^1"
    snt3 = SkewFieldSentence(snt3Str)

    #tested bug with plusSentenceParts
    lead1str = "(1 * a_0^1 + 1 * b_0^1) / (1 * a_0^1 + 1 * b_1^1) * T^2"
    lead2str = "(2 * a_0^1 + 2 * b_0^1) / (1 * a_0^1 + 1 * b_1^1) * T^2"

    lead1 = SkewFieldMonomial(lead1str)
    lead2 = SkewFieldMonomial(lead2str)

    quot = lead1.mInv().times(lead2)
    quotAInv = quot.aInv()

    trouble1 = lead1.times(quot)
    trouble1AInv = trouble1.aInv()
    trouble2 = lead2.plus(trouble1.aInv())


    print("lead1 = " + str(lead1))
    print("lead2 = " + str(lead2))
    print("quot = " + str(quot))
    print("trouble1 = " + str(trouble1))
    print("trouble2 = " + str(trouble2))

    cross1 = trouble1AInv.numer.times(lead2.denom)
    cross2 = trouble1AInv.denom.times(lead2.numer)

    print(cross1.plus(cross2))

    #tests polynomial long division
    print(lead2.asPoly().quo(lead1.asPoly()))
    
        
if __name__ == "__main__":
    sys.exit(main())

