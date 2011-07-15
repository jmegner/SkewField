#! /usr/bin/python
#
# initial author:   Kitty Yang
# initial date:     2011-07-15
#


import sys
import getopt
import re
import collections

# need this to access the SkewField stuff
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


# you can define new member functions
def simplifiedLetterStr(self):
    if self.sub == 0:
        return str(self.alpha)
    return str(self)

def main(argv=None):
    # here is how you add a member function to a class from SkewField.py;
    # note that the name can be different;
    SkewFieldLetter.simplifiedStr = simplifiedLetterStr
    SkewFieldWord.firstOfAlpha = firstOfAlpha
    SkewFieldWord.lastOfAlpha = lastOfAlpha    
    SkewFieldWord.firstAbnormalLetter = firstAbnormalLetter
    SkewFieldWord.reducedAtLetter = reducedAtLetter
    SkewFieldWord.reduced = reduced
    SkewFieldWord.raisedTo = raisedTo

    relation = []
    file = open("relation3_1.txt")

    for line in file:
        line = line[0:-1]
        relation.append(SkewFieldWord(line))

    print(relation)

    wrd1Str = "a_0^1 * b_1^2"

    ltrs1 = [
        SkewFieldLetter("a", 0),
        SkewFieldLetter("b", 1),
        SkewFieldLetter("b", 1),
    ]

    wrd1 = SkewFieldWord(ltrs1)

    print(wrd1)
    
    print(wrd1.reduced(relation))
        
if __name__ == "__main__":
    sys.exit(main())

