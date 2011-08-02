#! /usr/bin/env python
"""
initial author:   Jacob Egner
initial date:     2011-07-08

notable additions by Kitty Yang;

note: even though the standard python class Counter would be wonderful for
keeping counts of letters' powers and words' coefficients, consumers of this
code might have an old version of python that does not have the Counter class

note: I have adopted a pro-immutable philosophy when it comes to these
classes; member functions do not modify self or any other argument (except
when constructing self); each math operation results in a newly created
object; that is why there is a "plus" operation and not an "add" operation;

note:
below is a semi-accurate list of functions in each SkewField-ish class;
a class does not necessarily have every function in the list;

    __init__        constructor
    __str__         string representation; for pretty printing
    __repr__        more official string representation
    __hash__        so class can be used as a dict key
    __cmp__         defines an ordering on the class
    deepcopy        so we don't share references
    canonize        puts object into canonical form; simplifies some
    zero            the zero element of that class type
    one             the one element of that class type
    isZero          for identifying if object is a zero
    isOne           for identifying if object is a one
    isScalar        for identifying if object is a multiple of one
    asOneAbove      letter => word => sentence => monomial => polynomial
    asPoly          object copy promoted to a polynomial
    increasedSubs   copy with all component letter subscripts increased
    plus            result of addition operation
    minus           result of subtraction operation
    times           result of multiplication operation
    dividedBy       result of division operation
    aInv            additive inverse
    mInv            mulitiplicative inverse
    normalized      normalized form according to given relations

"""


global SFFileVersion
SFFileVersion = "0.21"


import sys
import getopt
import re
import collections
import math
import time


#global Fiddle
#Fiddle = False
#
#global WordHashCount
#WordHashCount = 0
#
#global WordEqCount
#WordEqCount = 0
#
#global WorstInputLetterCount
#WorstInputLetterCount = 0
#
#global WorstOutputLetterCount
#WorstOutputLetterCount = 0


# to help overcome our heartfelt loss of the treasured Counter class...
# one advantage is some auto-canonization from deleting zero-valued items
def updateCounts(counter, otherGuy):
    if isinstance(otherGuy, dict):
        for key, value in otherGuy.iteritems():
            if value == 0:
                continue

            newValue = counter.get(key, 0) + value

            if newValue == 0:
                counter.pop(key)
            else:
                counter[key] = newValue

    # else assume list/tuple/set
    else:
        for key in otherGuy:
            newValue = counter.get(key, 0) + 1

            if newValue == 0:
                counter.pop(key)
            else:
                counter[key] = newValue


################################################################################
# SkewFieldLetter is an alphabetical identifier and a subscript
# this representation can neither be a zero nor a one
#
# example: b_1
#   'b' is the alpha
#   '1' is the sub
#
class SkewFieldLetter():

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            #match = re.match("^([a-z]+)_?(-?\d*)$", args[0].replace(" ", ""))
            components = args[0].split("_")

            if len(components) != 2:
                raise ValueError("bad SkewFieldLetter() args " + str(args))

            self.alpha = SkewFieldLetter.alphaAsInt(components[0])
            self.sub = int(components[1])
        elif len(args) == 2:
            self.alpha = SkewFieldLetter.alphaAsInt(args[0])
            self.sub = int(args[1])
        else:
            raise ValueError("bad SkewFieldLetter() args " + str(args))


    def __str__(self):
        return SkewFieldLetter.alphaAsStr(self.alpha) + "_" + str(self.sub)


    def __repr__(self):
        return "SkewFieldLetter(\"" + str(self) + "\")"


    def __hash__(self):
        # imagine alpha as a row index and sub as a col index
        # and we're using the bijection of an infinite 2d array into an
        # infinite 1d array by traversing the 2d array diagonally;
        # this way our hash function is good at generating unique 1-tuples for
        # any combination of the alpha-sub 2-tuples

        #r = self.alpha
        #c = self.sub
        #return c * (c + 1) // 2 + r * c + r * (r + 1) // 2 + r

        # or we could go for this more efficient guy for sake of performance
        hashVal = self.sub * 128 + self.alpha + 1

        if hashVal == -1:
            return -2

        return hashVal


    def __eq__(self, other):
        if self.alpha != other.alpha:
            return False
        return self.sub == other.sub


    def __cmp__(self, other):
        if self.alpha < other.alpha:
            return -1
        if self.alpha > other.alpha:
            return 1
        if self.sub < other.sub:
            return -1
        if self.sub > other.sub:
            return 1
        return 0


    def deepcopy(self):
        return SkewFieldLetter(self.alpha, self.sub)


    def isZero(self):
        return False


    def isOne(self):
        return False


    def isScalar(self):
        return False


    def asWord(self):
        return SkewFieldWord([self])


    def asPoly(self):
        return self.asWord().asPoly()


    def increasedSubs(self, increment):
        return SkewFieldLetter(self.alpha, self.sub + increment)


    @staticmethod
    def alphaAsInt(alpha):
        if isinstance(alpha, int):
            return alpha
        elif isinstance(alpha, str):

            if re.match("^[a-z]+$", alpha) is None:
                raise ValueError("can not use arg " + str(alpha))

            intRep = -1;
            for char in alpha:
                intRep = (intRep + 1) * 26 + (ord(char) - ord("a"))

            return intRep

        else:
            raise ValueError("can not use arg " + str(alpha))


    @staticmethod
    def alphaAsStr(alpha):
        if isinstance(alpha, int):
            strRep = ""

            while True:
                remainder = alpha % 26
                strRep = chr(remainder + ord("a")) + strRep

                alpha = alpha // 26 - 1

                if alpha == -1:
                    break

            return strRep

        elif isinstance(alpha, str):
            return str(alpha)
        else:
            raise ValueError("can not use arg " + str(alpha))


################################################################################
# SkewFieldWord is the product of SkewFieldLetters;
# a SkewFieldWord can be 1 (empty letterCtr), but it can not be zero;
# for a zero, you need to go to the level of SkewFieldSentence
#
# example: b_1^2 * c_2;
#   'b_1' and 'c_2' are the letters (stored as letterCtr keys)
#   2 and 1 are the powers (storted as letterCtr values)
#
class SkewFieldWord():

    # letters argument can be str, tuple, list, set, or dict
    def __init__(self, letters = []):
        self.letterCtr = dict() # key is SkewFieldLetter, value is power

        if isinstance(letters, str):
            letters = letters.replace(" ", "")

            if letters == "" or letters == "1":
                self = SkewFieldWord.one()
            else:
                for letterWithPower in letters.split("*"):
                    (letterStr, power) = letterWithPower.split("^")

                    # if no exponentation symbol, letter implicitly raised to 0
                    if power is None:
                        power = 0

                    updateCounts(
                        self.letterCtr,
                        { SkewFieldLetter(letterStr) : int(power) }
                    )
        elif isinstance(letters, dict):
            self.letterCtr = dict(letters)
        else:
            updateCounts(self.letterCtr, letters)


    def __str__(self):
        if self.isOne():
            return "1"

        letterStrs = list()
        for letter in sorted(self.letterCtr.iterkeys()):
            letterStrs.append(str(letter) + "^" + str(self.letterCtr[letter]))

        return " * ".join(letterStrs)


    def __repr__(self):
        return "SkewFieldWord(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(tuple(sorted(self.letterCtr.items())))

        hashVal = 12345 * len(self.letterCtr)

        for letter, power in self.letterCtr.iteritems():
            #hashVal += hash(letter) * power
            #hashVal += (letter.sub * 128 + letter.alpha + 1) * power
            r = letter.sub * 128 + letter.alpha + 1
            c = power
            hashVal += c * (c + 1) // 2 + r * c + r * (r + 1) // 2 + r

        if hashVal == -1:
            return -2

        return hashVal




    # hopefully used by dictionaries in preference over __cmp__
    def __eq__(self, other):
        selfLen = len(self.letterCtr)
        otherLen = len(other.letterCtr)

        if selfLen != otherLen:
            return False

        for selfLetter, selfPower in self.letterCtr.iteritems():
            otherPower = other.letterCtr.get(selfLetter, 0)

            if selfPower != otherPower:
                return False

        return True


    def __cmp__(self, other):
        # go through letters in both words in order;
        # compare the letter, then compare the power

        selfSortedLetters = sorted(self.letterCtr.iterkeys())
        otherSortedLetters = sorted(other.letterCtr.iterkeys())

        # zip takes care of differently sized lists by stopping at end of
        # shorter list
        for selfLetter, otherLetter in zip(selfSortedLetters, otherSortedLetters):
            if selfLetter < otherLetter:
                return -1
            if selfLetter > otherLetter:
                return 1

            selfLetterPower = self.letterCtr[selfLetter]
            otherLetterPower = other.letterCtr[otherLetter]

            if selfLetterPower < otherLetterPower:
                return -1
            if selfLetterPower > otherLetterPower:
                return 1

        selfLen = len(self.letterCtr)
        otherLen = len(other.letterCtr)

        if selfLen < otherLen:
            return -1
        if selfLen > otherLen:
            return 1

        # words are same
        return 0


    def deepcopy(self):
        return SkewFieldWord(self.letterCtr)


    # delete all words with coefficient 0
    def canonize(self):
        for letter in self.letterCtr.keys():
            if self.letterCtr[letter] == 0:
                self.letterCtr.pop(letter)


    @staticmethod
    def one():
        return SkewFieldWord()


    def isZero(self):
        return False


    def isOne(self):
        return len(self.letterCtr) == 0


    def isScalar(self):
        return self.isOne()


    def asSentence(self):
        return SkewFieldSentence([self])


    def asPoly(self):
        return self.asSentence().asPoly()


    def increasedSubs(self, increment):
        newLetterCtr = dict()
        for letter, power in self.letterCtr.iteritems():
            newLetter = SkewFieldLetter(letter.alpha, letter.sub + increment)
            newLetterCtr[newLetter] = power
        return SkewFieldWord(newLetterCtr)


    # warning: SkewFieldSentence produced
    def plus(self, other):
        return SkewFieldSentence([self, other])


    # warning: SkewFieldSentence produced
    def minus(self, other):
        return self.asSentence().minus(other.asSentence())


    def times(self, other):
        productLetterCtr = dict(self.letterCtr)
        updateCounts(productLetterCtr, other.letterCtr)
        return SkewFieldWord(productLetterCtr)


    def dividedBy(self, other):
        return self.times(other.mInv())


    def mInv(self):
        mInvLetterCtr = dict()
        for letter, power in self.letterCtr.iteritems():
            mInvLetterCtr[letter] = -power
        return SkewFieldWord(mInvLetterCtr)


    def extremesOfAlpha(self, alpha):
        minSub = sys.maxint
        powerForMinSub = 0

        maxSub = -sys.maxint
        powerForMaxSub = 0

        for letter, power in self.letterCtr.iteritems():
            if letter.alpha == alpha and letter.sub < minSub:
                minSub = letter.sub
                powerForMinSub = power

            if letter.alpha == alpha and letter.sub > maxSub:
                maxSub = letter.sub
                powerForMaxSub = power

        if minSub == sys.maxint:
            return None

        return (minSub, powerForMinSub, maxSub, powerForMaxSub)


    def someAbnormalLetter(self, relations):
        for letter, power in self.letterCtr.iteritems():
            alpha = letter.alpha
            sub = letter.sub
            (minSub, minPower, maxSub, maxPower) \
                = relations[alpha].extremesOfAlpha(alpha)

            # letter is abnormal if it is outside of the left-inclusive
            # minSub..maxSub range and it's power is outside the
            # 0..abs(relPower) range

            if sub < minSub and not (0 <= power < abs(minPower)):
                return (letter, minSub, minPower)

            if sub >= maxSub and not (0 <= power < abs(maxPower)):
                return (letter, maxSub, maxPower)

        return (None, None, None)


    def normalizedAtLetter(self, letter, relSub, relPower, relations):
        alpha = letter.alpha
        sub = letter.sub
        power = self.letterCtr[letter]

        exponent = -(power // relPower)

        if power % relPower != 0 and relPower < 0:
            exponent -= 1

        increment = sub - relSub

        newRelation = relations[alpha].increasedSubs(increment)
        newRelation = newRelation.raisedTo(exponent)

        return self.times(newRelation)


    def normalized(self, relations):
        result = self

        while True:
            (abnormalLetter, relSub, relPower) \
                = result.someAbnormalLetter(relations)

            if abnormalLetter is None:
                break

            result = result.normalizedAtLetter(abnormalLetter,
                relSub, relPower, relations)

        return result


    def raisedTo(self, power):
        resultLetterCtr = dict()

        if power != 0:
            for letter, letterPower in self.letterCtr.iteritems():
                if letterPower != 0:
                    resultLetterCtr[letter] = letterPower * power

        return SkewFieldWord(resultLetterCtr)


    def letterSet(self):
        return set(self.letterCtr.iterkeys())


    def numLetters(self):
        return len(self.letterCtr)


################################################################################
# SkewFieldSentence is the sum of SkewFieldWords;
# can also be thought of as a polynomial of SkewFieldLetters
#
# SkewFieldSentence can be a one (identity word with coefficient = 1)
# SkewFieldSentence can be a zero (empty wordCtr)
#
# example: 3 * a_0^1 + 2 * b_1^2 * c_2^1 ;
#   'a_0^1' and 'b_1^2 * c_2^1' are the words (stored as wordCtr keys)
#   3 and 2 are the coefficients (storted as wordCtr values)
#
class SkewFieldSentence():


    # words argument can be str, tuple, list, set, or dict
    def __init__(self, words = []):
        self.wordCtr = dict() # key is SkewFieldWord, value is coef

        if isinstance(words, str):
            for coefWithWord in words.split("+"):
                splitResults = coefWithWord.split("*", 1)

                coef = splitResults[0]
                if len(splitResults) > 1:
                    word = SkewFieldWord(splitResults[1])
                else:
                    word = SkewFieldWord.one()

                coefInt = int(coef)

                if(coefInt != 0):
                    updateCounts(self.wordCtr, { word : coefInt })
        elif isinstance(words, dict):
            self.wordCtr = dict(words)
        else:
            updateCounts(self.wordCtr, words)

        #self.canonize()


    def __str__(self):
        if self.isZero():
            return "0"
        if self.isScalar():
            return str(self.wordCtr.values()[0])

        wordStrs = list()
        for word in sorted(self.wordCtr.iterkeys()):
            coef = self.wordCtr[word]
            if word.isOne():
                wordStrs.append(str(coef))
            else:
                wordStrs.append(str(coef) + " * " + str(word))
        return " + ".join(wordStrs)


    def __repr__(self):
        return "SkewFieldSentence(\"" + str(self) + "\")"


    def __hash__(self):
        hashVal = 0

        for word, coef in self.wordCtr.iteritems():
            hashVal += hash(word) * coef

        if hashVal == -1:
            return -2

        return hashVal


    def __eq__(self, other):
        selfLen = len(self.wordCtr)
        otherLen = len(other.wordCtr)

        if selfLen != otherLen:
            return False

        for selfWord, selfCoef in self.wordCtr.iteritems():
            otherCoef = other.wordCtr.get(selfWord, 0)

            if selfCoef != otherCoef:
                return False

        return True

    def __cmp__(self, other):
        selfSortedWords = sorted(self.wordCtr.iterkeys())
        otherSortedWords = sorted(other.wordCtr.iterkeys())

        # zip takes care of differently sized lists by stopping at end of
        # shorter list
        for selfWord, otherWord in zip(selfSortedWords, otherSortedWords):
            if selfWord < otherWord:
                return -1
            if selfWord > otherWord:
                return 1

            selfWordCoef = self.wordCtr[selfWord]
            otherWordCoef = other.wordCtr[otherWord]

            if selfWordCoef < otherWordCoef:
                return -1
            if selfWordCoef > otherWordCoef:
                return 1

        if len(selfSortedWords) < len(otherSortedWords):
            return -1
        if len(selfSortedWords) > len(otherSortedWords):
            return 1

        return 0


    def deepcopy(self):
        return SkewFieldSentence(self.wordCtr)


    # delete all words with coeff 0
    def canonize(self):
        for word in self.wordCtr.keys():
            if self.wordCtr[word] == 0:
                self.wordCtr.pop(word)


    @staticmethod
    def zero():
        return SkewFieldSentence()


    @staticmethod
    def one():
        return SkewFieldSentence([SkewFieldWord.one()])


    def isZero(self):
        return len(self.wordCtr) == 0


    def isOne(self):
        return self == SkewFieldSentence.one()


    def isScalar(self):
        return (len(self.wordCtr) == 0) \
            or (len(self.wordCtr) == 1 and self.wordCtr.keys()[0].isOne())


    def asMono(self):
        return SkewFieldMonomial(self, SkewFieldSentence.one(), 0)


    def asPoly(self):
        return self.asMono().asPoly()


    def increasedSubs(self, increment):
        resultWordCtr = dict()
        for word, coef in self.wordCtr.iteritems():
            resultWordCtr[word.increasedSubs(increment)] = coef
        return SkewFieldSentence(resultWordCtr)


    def plus(self, other):
        resultWordCtr = dict(self.wordCtr)
        updateCounts(resultWordCtr, other.wordCtr)
        return SkewFieldSentence(resultWordCtr)


    def minus(self, other):
        return self.plus(other.aInv())


    def times(self, other):
        prodWordCtr = dict()

        for selfWord, selfWordCoef in self.wordCtr.iteritems():
            for otherWord, otherWordCoef in other.wordCtr.iteritems():
                #updateCounts(prodWordCtr,
                #    { selfWord.times(otherWord) : selfWordCoef * otherWordCoef})

                termWord = selfWord.times(otherWord)
                termCoef = selfWordCoef * otherWordCoef

                newCoef = prodWordCtr.get(termWord, 0) + termCoef

                if newCoef == 0:
                    prodWordCtr.pop(termWord)
                else:
                    prodWordCtr[termWord] = newCoef

        return SkewFieldSentence(prodWordCtr)


    def aInv(self):
        invWordCtr = dict()

        for word, coef in self.wordCtr.items():
            invWordCtr[word] = -coef

        return SkewFieldSentence(invWordCtr)


    def normalized(self, relations):
        normalizedWordCtr = dict()

        for word, coef in self.wordCtr.iteritems():
            updateCounts(normalizedWordCtr,
                { word.normalized(relations) : coef })

        return SkewFieldSentence(normalizedWordCtr)


    def numLetters(self):
        letterCount = 0

        for word in self.wordCtr.iterkeys():
            letterCount += word.numLetters()

        return letterCount


################################################################################
# SkewFieldMonomial is a sentence-fraction times T to some power
#
# convention is for T to be on right-hand side and the SkewFieldSentence
# numerator and denominator to be on the left-hand side
#
# example: (2 * a_0^2) / (3 * b_1^3) * T^3
#
class SkewFieldMonomial():

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            # looks scary, but just a check for (sentence)/(sentence)*T^power
            match = re.match(
                "^ \( (.*) \)  /  \( (.*) \) \* T \^ (-? \d+) $",
                args[0].replace(" ", ""),
                re.VERBOSE,
            )

            # if not in formal mono-string format, try to parse as a sentence
            # with implicit denom=one and tpower=0
            if match is None:
                self.numer = SkewFieldSentence(args[0])
                self.denom = SkewFieldSentence.one()
                self.tpower = 0
            else:
                (numerStr, denomStr, tpowerStr) = match.groups(0)

                self.numer = SkewFieldSentence(numerStr)
                self.denom = SkewFieldSentence(denomStr)
                self.tpower = int(tpowerStr)

        elif len(args) == 3:
            self.numer = args[0]  # type SkewFieldSentence
            self.denom = args[1]  # type SkewFieldSentence
            self.tpower = args[2] # type integer

        else:
            raise ValueError("bad SkewFieldMonomial() args " + str(args))

        self.canonize()


    def __str__(self):
        return "(" + str(self.numer) + ") / (" + str(self.denom) \
            + ") * T^" + str(self.tpower)


    def __repr__(self):
        return "SkewFieldMonomial(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(str(self))


    def __cmp__(self, other):
        # note: due to not simplifying monos, this comparison only really works
        # well as an equality tester; less-than and greater-than are not
        # particularly useful between similarly-tpowered monos

        if self.tpower < other.tpower:
            return -1
        if self.tpower > other.tpower:
            return 1

        # since we don't simplify monos, we need to do a special check
        # to see if similarly-tpowered monos are equal but in different forms

        sumMono = self.plusSentencePart(other.aInv())

        if sumMono.numer.isZero():
            return 0

        # now back to straightforward comparison

        if self.numer < other.numer:
            return -1
        if self.numer > other.numer:
            return 1

        if self.denom < other.denom:
            return -1
        if self.denom > other.denom:
            return 1

        return 0


    def deepcopy(self):
        return SkewFieldMonomial(
            self.numer.deepcopy(),
            self.denom.deepcopy(),
            self.tpower,
        )


    def canonize(self):
        if self.numer == SkewFieldSentence.zero():
            self.denom = SkewFieldSentence.one()
            self.tpower = 0
        if self.numer == self.denom:
            self.numer = SkewFieldSentence.one()
            self.denom = SkewFieldSentence.one()


    @staticmethod
    def zero():
        return SkewFieldMonomial(
            SkewFieldSentence.zero(),
            SkewFieldSentence.one(),
            0,
        )


    @staticmethod
    def one():
        return SkewFieldMonomial(
            SkewFieldSentence.one(),
            SkewFieldSentence.one(),
            0,
        )


    def isZero(self):
        return self.numer.isZero()


    def isOne(self):
        return self.numer.isOne() and self.denom.isOne() and self.tpower == 0


    def isScalar(self):
        return (self.numer.isScalar() and self.denom.isScalar()
            and self.tpower == 0)


    def asPoly(self):
        return SkewFieldPolynomial([self])


    def increasedSubs(self, increment):
        return SkewFieldMonomial(
            self.numer.increasedSubs(increment),
            self.denom.increasedSubs(increment),
            self.tpower,
        )


    # warning: returns SkewFieldPolynomial
    def plus(self, other):
        return SkewFieldPolynomial([self, other])


    # warning: returns SkewFieldPolynomial
    def minus(self, other):
        return self.plus(other.aInv())


    def times(self, other):
        # product's tpower is easy
        prodTPower = self.tpower + other.tpower

        # new numer and denom is trickier; must increase subscripts in other's
        # letters by the amount of self.tpower (because of commutation),
        # then we can multiply

        adjustedOtherNumer = other.numer.increasedSubs(self.tpower)
        adjustedOtherDenom = other.denom.increasedSubs(self.tpower)

        if self.numer == adjustedOtherDenom:
            prodNumer = adjustedOtherNumer
            prodDenom = self.denom
        elif self.denom == adjustedOtherNumer:
            prodNumer = self.numer
            prodDenom = adjustedOtherDenom
        else:
            prodNumer = self.numer.times(adjustedOtherNumer)
            prodDenom = self.denom.times(adjustedOtherDenom)

        return SkewFieldMonomial(prodNumer, prodDenom, prodTPower)


    def dividedBy(self, other):
        return self.times(other.mInv())


    def aInv(self):
        return SkewFieldMonomial(
            self.numer.aInv(),
            self.denom,
            self.tpower,
        )


    def mInv(self):
        return SkewFieldMonomial(
            self.denom.increasedSubs(-self.tpower),
            self.numer.increasedSubs(-self.tpower),
            -self.tpower,
        )


    # for easy addition of self's numer/denom to other's numer/denom
    # if common denoms, use equation n1/d + n2/d = (n1 + n2) / d
    # else use equation n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1 * d2)
    def plusSentencePart(self, other):
        if self.denom == other.denom:
            return SkewFieldMonomial(
                self.numer.plus(other.numer),
                self.denom,
                self.tpower)
        return SkewFieldMonomial(
            self.numer.times(other.denom).plus(other.numer.times(self.denom)),
            self.denom.times(other.denom),
            self.tpower)


    def normalized(self, relations):
        return SkewFieldMonomial(
            self.numer.normalized(relations),
            self.denom.normalized(relations),
            self.tpower)


    def numLetters(self):
        return self.numer.numLetters() + self.denom.numLetters()


################################################################################
# SkewFieldPolynomial is basically a sum of differently tpowered monomials
# self.monoDict is a dict with tpowers as keys and monomials as values
# so, yes, there is some redundancy with the tpower also in the monomial
#
class SkewFieldPolynomial():


    # monos argument can be str or list (not dict)
    def __init__(self, monos = []):
        self.monoDict = dict() # key is tpower, value is monomial

        monoList = [] # to hold list of SkewFieldMonomials

        if isinstance(monos, str):
            for monoStr in monos.replace(" ", "").split("++"):
                monoList.append(SkewFieldMonomial(monoStr))
        elif isinstance(monos, dict):
            monoList = list(monos.values())
        else:
            monoList = monos

        # iterate to add the monos into our polynomial
        for otherMono in monoList:

            # if already have mono of that tpower, must add them together
            if otherMono.tpower in self.monoDict:
                myMono = self.monoDict.pop(otherMono.tpower)
                self.monoDict[otherMono.tpower] \
                    = myMono.plusSentencePart(otherMono)

            # else don't have mono of that tpower; simple insert
            else:
                # decided not to deepcopy otherMono for performance
                self.monoDict[otherMono.tpower] = otherMono

        self.canonize()


    def __str__(self):
        # temporarily using "++" as super-plus operator to make it more apparent
        # that we are adding polys
        if self.isZero():
            return "0"
        return " ++ ".join(map(str, self.asMonoList()))


    def __repr__(self):
        return "SkewFieldPolynomial(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(str(self))


    def __cmp__(self, other):
        if self.degree() < other.degree():
            return -1
        if self.degree() > other.degree():
            return 1

        selfMonoList = self.asMonoList()
        otherMonoList = other.asMonoList()

        for selfMono, otherMono in zip(selfMonoList, otherMonoList):
            if selfMono < otherMono:
                return -1
            if selfMono > otherMono:
                return 1

        if len(selfMonoList) < len(otherMonoList):
            return -1
        if len(selfMonoList) > len(otherMonoList):
            return 1

        return 0


    def deepcopy(self):
        return SkewFieldPolynomial(self.monoDict.values())


    def canonize(self):
        # even though polys usually forbid negative powers, we will sometimes
        # be dealing with negative powers, but the code is left in as a comment
        # for if/when we change our minds about the restriction

        #for tpower in self.monoDict.keys():
        #    if tpower < 0:
        #        raise ValueError("polys can not have negative powers: "
        #            + str(tpower))

        # remove monomials that are zero
        for tpower, mono in self.monoDict.items():
            if mono.isZero():
                self.monoDict.pop(tpower)


    @staticmethod
    def zero():
        return SkewFieldPolynomial()


    @staticmethod
    def one():
        return SkewFieldPolynomial([SkewFieldMonomial.one()])


    def isZero(self):
        return len(self.monoDict) == 0


    def isOne(self):
        return self.isZero() \
           or (len(self.monoDict) == 1 and self.monoDict.values()[0].isOne())


    def isScalar(self):
        return self.isZero() \
           or (len(self.monoDict) == 1 and self.monoDict.values()[0].isScalar())


    def asPoly(self):
        return self.deepcopy()


    # highest tpower first
    def asMonoList(self):
        return list(reversed(sorted(self.monoDict.values())))


    def asPowerList(self):
        return list(reversed(sorted(self.monoDict.keys())))


    def increasedSubs(self, increment):
        monoList = []
        for mono in self.monoDict.itervalues():
            monoList.append(mono.increasedSubs(increment))
        return SkewFieldPolynomial(monoList)


    def plus(self, other):
        return SkewFieldPolynomial(
            self.monoDict.values() + other.monoDict.values())


    def minus(self, other):
        return self.plus(other.aInv())


    def times(self, other):
        product = []
        for selfMono in self.monoDict.values():
            for otherMono in other.monoDict.values():
                product.append(selfMono.times(otherMono))

        return SkewFieldPolynomial(product)


    def leftQuotient(self, denominator):
        return self.leftQuotientAndRemainder(denominator)[0]


    def leftRemainder(self, denominator):
        return self.leftQuotientAndRemainder(denominator)[1]


    # quotient * denominator + remainder = numerator
    def leftQuotientAndRemainder(self, denominator):
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
            mono = leadNumerator.times(leadDenominator.mInv())
            result.append(mono)
            product = mono.asPoly().times(denominator)
            numerator = numerator.plus(product.aInv())

        return (SkewFieldPolynomial(result), numerator)


    def rightQuotient(self, denominator):
        return self.rightQuotientAndRemainder(denominator)[0]


    def rightRemainder(self, denominator):
        return self.rightQuotientAndRemainder(denominator)[1]


    # denominator * quotient + remainder = numerator
    def rightQuotientAndRemainder(self, denominator):
        numerator = self
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

        return (SkewFieldPolynomial(result), numerator)


    def aInv(self):
        result = []
        for mono in self.monoDict.values():
            result.append(mono.aInv())

        return SkewFieldPolynomial(result)


    def degree(self):
        if self.isZero():
            # zero-polys do not have a degree; we could return "None",
            # but a negative number is more convenient
            return -1
        else:
            return self.asPowerList()[0]


    def powerDiff(self):
        if self.isZero():
            # zero-polys do not have a degree; we could return "None",
            # but -1 is more convenient
            return 0
        else:
            return self.degree() - self.lowestPower()


    def highestMono(self):
        return self.asMonoList()[0]


    def lowestMono(self):
        return self.asMonoList()[-1]


    # no need for highestPower since that is taken care of by degree
    def lowestPower(self):
        return self.asPowerList()[-1]


    def numerDenomList(self):
        monoList = self.asMonoList()
        return [mono.numer for mono in monoList] \
            + [mono.denom for mono in monoList]


    def normalized(self, relations):
        newMonos = []
        for mono in self.monoDict.itervalues():
            newMonos.append(mono.normalized(relations))
        return SkewFieldPolynomial(newMonos)


    def numLetters(self):
        letterCount = 0

        for mono in self.monoDict.itervalues():
            letterCount += mono.numLetters()

        return letterCount


################################################################################
# MAIN
################################################################################

def noPrint(arg):
    pass

def yesPrint(arg):
    print(arg)


def SkewFieldMain(argv=None):

    cprint = noPrint

    if not(argv is None) and len(argv) > 1:
        for arg in argv[1:]:
            if arg == "-v" or arg == "--verbose":
                cprint = yesPrint

    cprint("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    cprint("@ SkewField.py FileVersion = " + SFFileVersion)
    cprint("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    cprint("")
    cprint("LETTER ###########################################################")

    global ltrA0Str
    ltrA0Str = "a_0"

    global ltrA0
    ltrA0 = SkewFieldLetter("a", 0)
    cprint("ltrA0 = " + str(ltrA0))

    # basic construction-representation test
    assert(str(ltrA0) == ltrA0Str)

    global ltrA0Again
    ltrA0Again = SkewFieldLetter(ltrA0Str)
    global ltrA1
    ltrA1 = SkewFieldLetter("a_1")
    global ltrB0
    ltrB0 = SkewFieldLetter("b_0")

    # test comparisons
    assert(ltrA0 == ltrA0Again)
    assert(ltrA0 == ltrA0)
    assert(ltrA0 < ltrA1)
    assert(ltrA1 > ltrA0)
    assert(ltrA1 < ltrB0)
    assert(ltrB0 > ltrA1)

    # test basic operations

    assert(ltrA0.increasedSubs(1) == ltrA1)

    assert(SkewFieldLetter.alphaAsStr("ab") == str("ab"))
    assert(SkewFieldLetter.alphaAsStr(27) == str("ab"))
    assert(SkewFieldLetter.alphaAsInt("ab") == 27)
    assert(SkewFieldLetter.alphaAsInt(27) == 27)

    # test operations did not violate immutability
    assert(ltrA0Again == ltrA0)
    assert(str(ltrA0Again) == str(ltrA0))

    # do just-run-it test with all of them

    global ltrList
    ltrList = [ ltrA0, ltrA0Again, ltrA1, ltrB0, ]

    for ltr in ltrList:
        str(ltr)
        ltr == ltr
        ltr.deepcopy()
        ltr.isZero()
        ltr.isOne()
        ltr.isScalar()
        ltr.asWord()
        ltr.asPoly()
        ltr.increasedSubs(1)

    cprint("")
    cprint("WORD #############################################################")

    # basic construction-representation test

    global wrd1Str
    wrd1Str = "a_0^1 * b_1^2"

    global ltrs1
    ltrs1 = [
        SkewFieldLetter("a", 0),
        SkewFieldLetter("b", 1),
        SkewFieldLetter("b", 1),
    ]

    global wrd1
    wrd1 = SkewFieldWord(ltrs1)
    cprint("wrd1 = " + str(wrd1))

    assert(str(wrd1) == wrd1Str)
    assert(wrd1 == wrd1)

    # test identity comparison

    global wrd1Again
    wrd1Again = SkewFieldWord(wrd1Str)
    cprint("wrd1Again = " + str(wrd1Again))
    assert(wrd1 == wrd1Again)

    # test increasedSubs
    global wrd2Str
    wrd2Str = "a_2^1 * b_3^2"
    global wrd2
    wrd2 = wrd1.increasedSubs(2)
    cprint("wrd2 = " + str(wrd2))
    assert(str(wrd2) == wrd2Str)

    # test alternate constructor and canonization

    global wrd3Str
    wrd3Str = "b_1^3 * c_2^1"

    global ltrs3Dict
    ltrs3Dict = {
        SkewFieldLetter("b", 1) : 3,
        SkewFieldLetter("c", 2) : 1,
    }

    global wrd3
    wrd3 = SkewFieldWord(ltrs3Dict)
    cprint("wrd3 = " + str(wrd3))
    assert(str(wrd3) == wrd3Str)

    # test cmp a bit

    assert(wrd1 < wrd3)
    assert(wrd3 > wrd1)

    # test mInv and times a bit

    global wrd1MInvStr
    wrd1MInvStr = "a_0^-1 * b_1^-2"
    global wrd1MInv
    wrd1MInv = wrd1.mInv()
    cprint("wrd1MInv = " + str(wrd1MInv))
    assert(str(wrd1MInv) == wrd1MInvStr)
    assert(wrd1 == wrd1.mInv().mInv())
    cprint("wrd1 * wrd1MInv = " + str(wrd1.times(wrd1MInv)))
    assert(SkewFieldWord.one() == wrd1.times(wrd1MInv))

    # test times and dividedBy

    global wrd4Str
    wrd4Str = "a_0^1 * b_1^5 * c_2^1"

    global wrd4
    wrd4 = wrd1.times(wrd3)
    cprint("wrd1 * wrd3 = wrd4 = " + str(wrd4))
    assert(str(wrd4) == wrd4Str)

    global wrd5Str
    wrd5Str = "a_0^1 * b_1^5 * c_2^1"

    global wrd5
    wrd5 = wrd3.times(wrd1)
    cprint("wrd3 * wrd1 = wrd5 = " + str(wrd5))
    assert(str(wrd5) == wrd5Str)

    assert(wrd5.dividedBy(wrd3) == wrd1)
    assert(wrd5.dividedBy(wrd1) == wrd3)

    # do just-run-it tests

    global wrdList
    wrdList = [ wrd1, wrd2, wrd3, wrd4, wrd5, ]

    for wrd in wrdList:
        str(wrd)
        wrd == wrd
        wrd.deepcopy()
        wrd.isZero()
        wrd.isOne()
        wrd.isScalar()
        wrd.asSentence()
        wrd.asPoly()
        wrd.increasedSubs(1)
        wrd.plus(wrd)
        wrd.minus(wrd)
        wrd.times(wrd)
        wrd.dividedBy(wrd)
        wrd.mInv()
        wrd.extremesOfAlpha(0)

    cprint("")
    cprint("SENTENCE #########################################################")

    # test wrd+wrd => sentence

    global snt1Str
    snt1Str = "1 * a_0^1 + 1 * b_0^1"
    global snt1
    snt1 = SkewFieldWord("a_0^1").plus(SkewFieldWord("b_0^1"))
    cprint("wrdA + wrdB = snt1 = " + str(snt1))
    assert(str(snt1) == snt1Str)

    global snt1Again
    snt1Again = SkewFieldSentence(snt1Str)
    assert(snt1 == snt1Again)
    assert(str(snt1Again) == snt1Str)

    # test some scalar sentences

    global snt2Str
    snt2Str = "2"
    global snt2
    snt2 = SkewFieldSentence(snt2Str)
    cprint("snt2 = " + str(snt2))
    assert(str(snt2) == snt2Str)

    global snt3Str
    snt3Str = "3 + 4 * a_0^5 + 6 * a_0^7 * b_1^8"
    global snt3
    snt3 = SkewFieldSentence(snt3Str)
    cprint("snt3 = " + str(snt3))
    assert(str(snt3) == snt3Str)

    assert(not snt1.isScalar())
    assert(snt2.isScalar())
    assert(not snt3.isScalar())

    # test increased subs

    global snt99Str
    snt99Str = "3 + 4 * a_-2^5 + 6 * a_-2^7 * b_-1^8"
    global snt99
    snt99 = snt3.increasedSubs(-2)
    cprint("snt99 = " + str(snt99))
    assert(str(snt99) == snt99Str)

    # test aInv and plus a bit

    global snt3AInvStr
    snt3AInvStr = "-3 + -4 * a_0^5 + -6 * a_0^7 * b_1^8"

    global snt3AInv
    snt3AInv = snt3.aInv()
    cprint("snt3AInv = " + str(snt3AInv))
    assert(str(snt3AInv) == snt3AInvStr)

    global snt4Str
    snt4Str = "0"

    global snt4
    snt4 = snt3.plus(snt3AInv)
    cprint("snt3 + snt3AInv = snt4 = " + str(snt4))
    assert(str(snt4) == snt4Str)

    # test plus a bit more

    global snt5Str
    snt5Str = "2 + 3 * b_0^1 + 4 * c_0^1"
    global snt5
    snt5 = SkewFieldSentence(snt5Str)
    cprint("snt5 = " + str(snt5))
    assert(str(snt5) == snt5Str)

    global snt6Str
    snt6Str = "2 + 1 * a_0^1 + 4 * b_0^1 + 4 * c_0^1"
    global snt6
    snt6 = snt1.plus(snt5)
    cprint("snt6 = " + str(snt6))
    assert(str(snt6) == snt6Str)

    # test commutivity of plus
    assert(snt1.plus(snt5) == snt5.plus(snt1))

    # test times - associativity and commutivity

    global snt7Str
    snt7Str = "3 + 5 * a_0^7"
    global snt7
    snt7 = SkewFieldSentence(snt7Str)
    cprint("snt7 = " + str(snt7))
    assert(str(snt7) == snt7Str)

    global snt8Str
    snt8Str = "6 * a_0^1 + 10 * a_0^7 * b_0^1 + 10 * a_0^8 + 6 * b_0^1"
    #snt8Str = "6 * a_0^1 + 10 * a_0^8 + 6 * b_0^1 + 10 * a_0^7 * b_0^1"
    global snt8
    snt8 = snt2.times(snt1).times(snt7)
    cprint("(snt2 * snt1) * snt7 = snt8 = " + str(snt8))
    assert(str(snt8) == snt8Str)

    global snt9
    snt9 = snt2.times(snt1.times(snt7))
    cprint("snt2 * (snt1 * snt7) = snt9 = " + str(snt9))
    assert(snt9 == snt8)

    global snt10
    snt10 = snt7.times(snt2.times(snt1))
    cprint("snt7 * snt2 * snt1 = snt10 = " + str(snt10))
    assert(snt10 == snt8)

    # do just-run-it tests

    SkewFieldSentence.zero()
    SkewFieldSentence.one()

    global sntList
    sntList = [
        snt1, snt2, snt3, snt3AInv, snt4, snt5,
        snt6, snt7, snt8, snt9, snt10, snt99,
    ]

    for snt in sntList:
        str(snt)
        snt == snt
        snt.deepcopy()
        snt.isZero()
        snt.isOne()
        snt.isScalar()
        snt.asMono()
        snt.asPoly()
        snt.increasedSubs(1)
        snt.plus(snt)
        snt.minus(snt)
        snt.times(snt)
        snt.aInv()

    cprint("")
    cprint("MONOMIAL #########################################################")

    # basic construction-representation test

    # (1 * a_0^1 + 1 * b_0^1) / (3 + 5 * a_0^7) * T^-2
    global mono1Str
    mono1Str = "(" + str(snt1) + ") / (" + str(snt7) + ") * T^-2"
    global mono1
    mono1 = SkewFieldMonomial(snt1, snt7, -2)
    cprint("mono1 = " + str(mono1))
    assert(str(mono1) == mono1Str)

    global mono1Again
    mono1Again = SkewFieldMonomial(mono1Str)
    assert(str(mono1Again) == mono1Str)

    # test mInv and test times a bit

    global mono1MInvStr
    mono1MInvStr = "(3 + 5 * a_2^7) / (1 * a_2^1 + 1 * b_2^1) * T^2"
    global mono1MInv
    mono1MInv = mono1.mInv()
    cprint("mono1.mInv() = mono1Inv = " + str(mono1MInv))
    assert(str(mono1MInv) == mono1MInvStr)

    global monoOneStr
    monoOneStr = str(SkewFieldMonomial.one())

    global mono2
    mono2 = mono1.times(mono1MInv)
    cprint("mono1 * mono1MInv = mono2 = " + str(mono2))
    assert(str(mono2) == monoOneStr)

    global mono3
    mono3 = mono1MInv.times(mono1)
    cprint("mono1MInv * mono1 = mono3 = " + str(mono3))
    assert(str(mono3) == monoOneStr)

    cprint("mono1 * (mono1^-1) = " + str(mono1.times(mono1.mInv())))
    assert(SkewFieldMonomial.one() == mono1.times(mono1.mInv()))

    # test times some more

    global mono4Str
    mono4Str = "(" + str(snt1) + ") / (1) * T^3"
    global mono4
    mono4 = SkewFieldMonomial(snt1, SkewFieldSentence.one(), 3)
    cprint("mono4 = " + str(mono4))
    assert(str(mono4) == mono4Str)

    global mono5Str
    mono5Str = "(" + str(snt1.times(snt1)) + ") / (" + str(snt7) + ") * T^1"
    global mono5
    mono5 = mono1.times(mono4.increasedSubs(2))
    cprint("mono5Str = " + str(mono5Str))
    cprint("mono1 * mono4.incSubs(2) = mono5 = " + str(mono5))
    assert(str(mono5) == mono5Str)

    # test times, particularly with one

    global mono6
    mono6 = mono4.times(SkewFieldMonomial.one())
    cprint("mono4 * 1 = mono6 = " + str(mono6))
    assert(mono6 == mono4)

    global mono7
    mono7 = SkewFieldMonomial.one().times(mono4)
    cprint("1 * mono4 = mono7 = " + str(mono7))
    assert(mono7 == mono4)

    # test that b - a * (a^-1 * b) == 0

    global monoAStr
    monoAStr = "(1 * a_0^1) / (1) * T^0"
    global monoBStr
    monoBStr = "(1 * b_0^1) / (1) * T^0"

    global monoA
    monoA = SkewFieldMonomial(monoAStr)
    global monoB
    monoB = SkewFieldMonomial(monoBStr)

    global monoAr
    monoAr = monoA.mInv()
    global monoArtB
    monoArtB = monoAr.times(monoB)
    global monoAtArtB
    monoAtArtB = monoA.times(monoArtB)
    global monoBmAtArtB
    monoBmAtArtB = monoB.minus(monoAtArtB)
    cprint("monoA = " + str(monoA))
    cprint("monoB = " + str(monoB))
    cprint("monoAr = " + str(monoAr))
    cprint("monoArtB = " + str(monoArtB))
    cprint("monoAtArtB = " + str(monoAtArtB))
    cprint("monoBmAtArtB = " + str(monoBmAtArtB))

    assert(monoBmAtArtB.isZero())

    # just-run-it tests

    SkewFieldMonomial.zero()
    SkewFieldMonomial.one()

    global monoList
    monoList = [
        mono1, mono1MInv, mono2, mono3, mono4, mono5, mono6, mono7,
        monoA, monoB, monoAr, monoArtB, monoAtArtB,
    ]

    for mono in monoList:
        str(mono)
        mono == mono
        mono.deepcopy()
        mono.isZero()
        mono.isOne()
        mono.isScalar()
        if mono.tpower >= 0:
            mono.asPoly()
        mono.increasedSubs(1)
        if mono.tpower >= 0:
            mono.plus(mono)
        mono.minus(mono)
        mono.times(mono)
        mono.dividedBy(mono)
        mono.aInv()
        mono.mInv()
        mono.plusSentencePart(mono)

    cprint("")
    cprint("POLYNOMIAL #######################################################")

    # basic construction and representation test

    global poly1Str
    poly1Str = "(2 * a_0^1 + 2 * b_0^1) / (1) * T^3 ++ (1 * a_0^1) / (1) * T^0"

    global poly1
    poly1 = SkewFieldPolynomial([mono4, monoA, mono4])
    cprint("poly1 = " + str(poly1))
    assert(str(poly1) == poly1Str)

    global poly2Str
    poly2Str = "(1 * b_0^1) / (1) * T^3"

    global poly2
    poly2 = SkewFieldPolynomial(poly2Str)
    cprint("poly2 = " + str(poly2))
    assert(str(poly2) == poly2Str)

    global poly3Str
    poly3Str = "(1 * a_0^1) / (1 * b_0^1) * T^3"

    global poly3
    poly3 = SkewFieldPolynomial(poly3Str)
    cprint("poly3 = " + str(poly3))
    assert(str(poly3) == poly3Str)

    global poly4Str
    poly4Str = "1"

    global poly4
    poly4 = SkewFieldPolynomial(poly4Str)
    cprint("poly4 = " + str(poly4))

    # test cmp a bit
    assert(poly1 == poly1)
    assert(poly1 < poly2)
    assert(poly1 > poly3)

    # test the special value functions

    assert(SkewFieldPolynomial.zero().isZero())
    assert(SkewFieldPolynomial.zero().isScalar())
    assert(SkewFieldPolynomial.one().isOne())
    assert(SkewFieldPolynomial.one().isScalar())
    assert(not SkewFieldPolynomial.one().isZero())
    assert(not poly1.isScalar())

    # test rightQuotientAndRemainder

    poly5 = SkewFieldPolynomial([mono1.mInv(), mono4, mono4])
    cprint("poly5 = " + str(poly5))
    poly6 = SkewFieldPolynomial([mono1.mInv(), mono3, mono4, mono4])
    cprint("poly6 = " + str(poly6))

    (poly65QuotR, poly65RemainR) = poly6.rightQuotientAndRemainder(poly5)
    cprint("poly6.rightQuotient(poly5) = poly65QuotR = " + str(poly65QuotR))
    cprint("poly6.rightRemainder(poly5) = poly65RemainR = " + str(poly65RemainR))
    assert(poly65QuotR == poly6.rightQuotient(poly5))
    assert(poly65RemainR == poly6.rightRemainder(poly5))
    assert(poly6 == poly5.times(poly65QuotR).plus(poly65RemainR))

    # poly long division with answers that actually vary between left-division
    # and right-division

    polyDividendStr = (
        "(1 * c_0^1) / (1) * T^2 ++ " +
        "(1 * b_0^1) / (1) * T^1 ++ " +
        "(1 * a_0^1) / (1) * T^0")
    polyDivisorStr = (
        "(1 * e_0^1) / (1) * T^1 ++ " +
        "(1 * d_0^1) / (1) * T^0")

    polyDividend = SkewFieldPolynomial(polyDividendStr)
    polyDivisor = SkewFieldPolynomial(polyDivisorStr)

    cprint("polyDividend = " + str(polyDividend))
    cprint("polyDivisor = " + str(polyDivisor))

    (polyLeftQuotient, polyLeftRemainder) \
        = polyDividend.leftQuotientAndRemainder(polyDivisor)

    cprint("polyLeftQuotient = " + str(polyLeftQuotient))
    cprint("polyLeftRemainder = " + str(polyLeftRemainder))
    assert(polyLeftQuotient == polyDividend.leftQuotient(polyDivisor))
    assert(polyLeftRemainder == polyDividend.leftRemainder(polyDivisor))

    assert(polyLeftQuotient.times(polyDivisor).plus(polyLeftRemainder)
        == polyDividend)

    (polyRightQuotient, polyRightRemainder) \
        = polyDividend.rightQuotientAndRemainder(polyDivisor)

    cprint("polyRightQuotient = " + str(polyRightQuotient))
    cprint("polyRightRemainder = " + str(polyRightRemainder))
    assert(polyRightQuotient == polyDividend.rightQuotient(polyDivisor))
    assert(polyRightRemainder == polyDividend.rightRemainder(polyDivisor))

    assert(polyDivisor.times(polyRightQuotient).plus(polyRightRemainder)
        == polyDividend)

    # test powerDiff
    # should be equal if all powers non-negative and has constant term
    assert(poly1.degree() == poly1.powerDiff())
    assert(poly6.degree() == poly6.powerDiff())

    # test lowestPower
    cprint("poly1 = " + str(poly1))
    poly1LowPower = poly1.lowestPower()
    cprint("poly1.lowestPower() = " + str(poly1LowPower))
    poly1Deg = poly1.degree()
    cprint("poly1.degree() = " + str((poly1Deg)))
    assert(poly1Deg - poly1LowPower == poly1.powerDiff())

    # RESUME: asMonoList
    # TODO: much more poly testing

    # just-run-it tests

    SkewFieldPolynomial.zero()
    SkewFieldPolynomial.one()

    global polyList
    polyList = [
        poly1,
    ]

    for poly in polyList:
        str(poly)
        poly == poly
        poly.deepcopy()
        poly.isZero()
        poly.isOne()
        poly.isScalar()
        poly.asPoly()
        poly.asMonoList()
        poly.increasedSubs(1)
        poly.plus(poly)
        poly.minus(poly)
        poly.times(poly)
        poly.rightQuotient(poly)
        poly.aInv()
        poly.degree()
        poly.highestMono()


    cprint("")
    cprint("RELATIONS ########################################################")

    # test that relation a_0^1 should be able to get rid of any
    # letter with alpha=a

    relsA0_1 = [ SkewFieldWord("a_0^1"), ]

    cprint("relsA0_1 = " + str(relsA0_1))

    # quickly test that 1 normalized is 1
    assert(SkewFieldWord.one().normalized(relsA0_1).isOne())

    wrd50Str = "a_3^3"
    wrd50 = SkewFieldWord(wrd50Str)
    cprint("wrd50 = " + str(wrd50))
    wrd50Norm = wrd50.normalized(relsA0_1)
    cprint("wrd50Norm = " + str(wrd50Norm))
    assert(wrd50Norm.isOne())

    wrd51Str = "a_-4^-4"
    wrd51 = SkewFieldWord(wrd51Str)
    cprint("wrd51 = " + str(wrd51))
    wrd51Norm = wrd51.normalized(relsA0_1)
    cprint("wrd51Norm = " + str(wrd51Norm))
    assert(wrd51Norm.isOne())

    wrd52Str = "a_0^1"
    wrd52 = SkewFieldWord(wrd52Str)
    cprint("wrd52 = " + str(wrd52))
    wrd52Norm = wrd52.normalized(relsA0_1)
    cprint("wrd52Norm = " + str(wrd52Norm))
    assert(wrd52Norm.isOne())

    # test that we raise the relation-word to the right power in the
    # combination of cases of negative powers vs positive powers and zero
    # remainder versus nonzero remainder

    relsP = [
        SkewFieldWord("a_0^3"),
    ]

    relsN = [
        SkewFieldWord("a_0^-3"),
    ]

    wrdP6Str = "a_0^6"
    wrdN6Str = "a_0^-6"
    wrdP7Str = "a_0^7"
    wrdN7Str = "a_0^-7"

    wrdP6 = SkewFieldWord(wrdP6Str)
    wrdN6 = SkewFieldWord(wrdN6Str)
    wrdP7 = SkewFieldWord(wrdP7Str)
    wrdN7 = SkewFieldWord(wrdN7Str)

    wrdP6P = wrdP6.normalized(relsP)
    wrdP6N = wrdP6.normalized(relsN)
    wrdN6P = wrdN6.normalized(relsP)
    wrdN6N = wrdN6.normalized(relsN)

    assert(wrdP6P.isOne())
    assert(wrdP6N.isOne())
    assert(wrdN6P.isOne())
    assert(wrdN6N.isOne())

    wrdA0_1 = SkewFieldWord("a_0^1")
    wrdA0_2 = SkewFieldWord("a_0^2")

    wrdP7P = wrdP7.normalized(relsP)
    wrdP7N = wrdP7.normalized(relsN)
    wrdN7P = wrdN7.normalized(relsP)
    wrdN7N = wrdN7.normalized(relsN)

    assert(wrdP7P == wrdA0_1)
    assert(wrdP7N == wrdA0_1)
    assert(wrdN7P == wrdA0_2)
    assert(wrdN7N == wrdA0_2)

    cprint("negative powers vs positive powers and zero vs nonzero remainders checked")

    # general word, sentence, mono, poly testing

    global relsBasic
    relsBasic = [
        SkewFieldWord("a_0^1 * b_0^-1 * b_1^1"),
        SkewFieldWord("b_0^1 * b_1^-1 * b_2^1"),
    ]

    cprint("relsBasic = [ " + ",  ".join(map(str,relsBasic)) + " ]")

    # test word reduction

    global wrdsBeforeNormalization
    wrdsBeforeNormalization = [
        wrd1,
        wrd2,
    ]

    global wrdsAfterNormalization
    wrdsAfterNormalization = []

    global wrdStrsAfterNormalization 
    wrdStrsAfterNormalization = [
        "b_0^1 * b_1^1",
        "b_0^-2 * b_1^1",
    ]

    for wrd, wrdNormalizedStr in zip(wrdsBeforeNormalization, wrdStrsAfterNormalization):
        wrdNormalized = wrd.normalized(relsBasic)
        wrdsAfterNormalization.append(wrdNormalized)

        cprint("wrd = " + str(wrd))
        cprint("    normalized = " + str(wrdNormalized))
        cprint("    answer  = " + wrdNormalizedStr)
        assert(str(wrdNormalized) == wrdNormalizedStr)

    # test sentence reduction

    global sntsBeforeNormalization
    sntsBeforeNormalization = [
        snt1,
        snt2,
    ]

    global sntsAfterNormalization
    sntsAfterNormalization = []

    global sntStrsAfterNormalization 
    sntStrsAfterNormalization = [
        "1 * b_0^1 + 1 * b_0^1 * b_1^-1",
        "2",
    ]

    for snt, sntNormalizedStr in zip(sntsBeforeNormalization, sntStrsAfterNormalization):
        sntNormalized = snt.normalized(relsBasic)
        sntsAfterNormalization.append(sntNormalized)

        cprint("snt = " + str(snt))
        cprint("    normalized = " + str(sntNormalized))
        cprint("    answer  = " + sntNormalizedStr)

    # test mono reduction

    global monosBeforeNormalization
    monosBeforeNormalization = [
        mono1,
        mono4,
    ]

    global monosAfterNormalization
    monosAfterNormalization = []

    global monoStrsAfterNormalization 
    monoStrsAfterNormalization = [
        "(1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1 + 1 * b_0^7 * b_1^-7) * T^-2",
        "(1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1) * T^3",
    ]

    for mono, monoNormalizedStr in zip(monosBeforeNormalization, monoStrsAfterNormalization):
        monoNormalized = mono.normalized(relsBasic)
        monosAfterNormalization.append(monoNormalized)

        cprint("mono = " + str(mono))
        cprint("    normalized = " + str(monoNormalized))
        cprint("    answer  = " + monoNormalizedStr)

    # test poly normalization

    global polysBeforeNormalization
    polysBeforeNormalization = [
        poly1,
    ]

    global polysAfterNormalization
    polysAfterNormalization = []

    global polyStrsAfterNormalization 
    polyStrsAfterNormalization = [
        "(1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1) * T^3 ++ (1 * b_0^1 * b_1^-1) / (1) * T^0",
    ]

    for poly, polyNormalizedStr in zip(polysBeforeNormalization, polyStrsAfterNormalization):
        polyNormalized = poly.normalized(relsBasic)
        polysAfterNormalization.append(polyNormalized)

        cprint("poly = " + str(poly))
        cprint("    normalized = " + str(polyNormalized))
        cprint("    answer  = " + polyNormalizedStr)

    cprint("")
    cprint("MISC #############################################################")

    cprint("")
    cprint("END OF SKEWFIELD.PY TEST BATTERY #################################")

    dividend = SkewFieldPolynomial(
        "(1 * c_0^1) / (1) * T^2 ++ " +
        "(1 * b_0^1) / (1) * T^1 ++ " +
        "(1 * a_0^1) / (1) * T^0")

    divisor = SkewFieldPolynomial(
        "(1 * e_0^1) / (1) * T^1 ++ " +
        "(1 * d_0^1) / (1) * T^0")

    cprint("dividend = " + str(dividend))
    cprint("divisor = " + str(divisor))

    leftQ = dividend.leftQuotient(divisor)
    leftProd = leftQ.times(divisor)

    cprint("leftQ = " + str(leftQ))
    cprint("leftProd = " + str(leftProd))

    rightQ = dividend.rightQuotient(divisor)
    rightProd = divisor.times(rightQ)

    cprint("rightQ = " + str(rightQ))
    cprint("rightProd = " + str(rightProd))


    return 0


if __name__ == "__main__":
    sys.exit(SkewFieldMain(sys.argv))


