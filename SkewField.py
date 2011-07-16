#! /usr/bin/python
"""
initial author:   Jacob Egner
initial date:     2011-07-08

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
    times           result of multiplication operation
    aInv            additive inverse
    mInv            mulitiplicative inverse

"""


FileVersion = "0.10"


import sys
import getopt
import re
import collections
import math


# to help overcome our heartfelt loss of the treasured Counter class...
# one advantage is some auto-canonization from deleting zero-valued items
def updateCounts(counter, otherGuy):
    # step1: merge

    if isinstance(otherGuy, dict):
        for key, value in otherGuy.items():
            key = key.deepcopy()
            counter[key] = counter.get(key, 0) + value;
    # else assume list/tuple/set
    else:
        for key in otherGuy:
            key = key.deepcopy()
            counter[key] = counter.get(key, 0) + 1;

    # step2: simplify by removing zero-valued items
    for key, value in counter.items():
        if counter[key] == 0:
            counter.pop(key)


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
        return hash(str(self))


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


    def alphaAsInt(alpha):
        if isinstance(alpha, int):
            return alpha
        elif isinstance(alpha, str):

            if re.match("^[a-z]+$", alpha) == None:
                raise ValueError("can not use arg " + str(alpha))

            intRep = -1;
            for char in alpha:
                intRep = (intRep + 1) * 26 + (ord(char) - ord("a"))

            return intRep

        else:
            raise ValueError("can not use arg " + str(alpha))
    alphaAsInt = staticmethod(alphaAsInt)


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
    alphaAsStr = staticmethod(alphaAsStr)


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
                    if power == None:
                        power = 0

                    updateCounts(
                        self.letterCtr,
                        { SkewFieldLetter(letterStr) : int(power) }
                    )
        else:
            updateCounts(self.letterCtr, letters)

        self.canonize()


    def __str__(self):
        if self == SkewFieldWord.one():
            return "1"

        letterStrs = list()
        for letter in sorted(self.letterCtr.keys()):
            letterStrs.append(str(letter) + "^" + str(self.letterCtr[letter]))

        return " * ".join(letterStrs)


    def __repr__(self):
        return "SkewFieldWord(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(str(self))


    def __cmp__(self, other):
        # go through letters in each word in order;
        # compare the letter, then compare the power

        selfSortedLetters = sorted(self.letterCtr.keys())
        otherSortedLetters = sorted(other.letterCtr.keys())

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

        if len(selfSortedLetters) < len(otherSortedLetters):
            return -1
        if len(selfSortedLetters) > len(otherSortedLetters):
            return 1

        # words are same
        return 0


    def deepcopy(self):
        cpy = SkewFieldWord()
        for letter, power in self.letterCtr.items():
            cpy.letterCtr[letter.deepcopy()] = int(power)
        return cpy


    # delete all words with coefficient 0
    def canonize(self):
        for letter in self.letterCtr.keys():
            if self.letterCtr[letter] == 0:
                self.letterCtr.pop(letter)


    def one():
        return SkewFieldWord()
    one = staticmethod(one)


    def isZero(self):
        return False


    def isOne(self):
        return self == SkewFieldWord.one()


    def isScalar(self):
        return self.isOne()


    def asSentence(self):
        return SkewFieldSentence([self])


    def asPoly(self):
        return self.asSentence().asPoly()


    def increasedSubs(self, increment):
        result = SkewFieldWord()
        for letter, power in self.letterCtr.items():
            newLetter = letter.deepcopy()
            newLetter.sub += increment
            result.letterCtr[newLetter] = int(power)
        return result


    # warning: SkewFieldSentence produced
    def plus(self, other):
        return SkewFieldSentence([self, other])


    def times(self, other):
        product = SkewFieldWord(self.letterCtr)
        updateCounts(product.letterCtr, other.letterCtr)
        product.canonize() # probably not needed
        return product

    def mInv(self):
        inverse = SkewFieldWord()
        for letter, power in self.letterCtr.items():
            inverse.letterCtr[letter] = -power
        return inverse


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

                updateCounts(self.wordCtr, { word : int(coef) })
        else:
            updateCounts(self.wordCtr, words)

        self.canonize()


    def __str__(self):
        if self.isZero():
            return "0"
        if self.isScalar():
            return str(self.wordCtr.values()[0])

        wordStrs = list()
        for word in sorted(self.wordCtr.keys()):
            coef = self.wordCtr[word]
            if word.isOne():
                wordStrs.append(str(coef))
            else:
                wordStrs.append(str(coef) + " * " + str(word))
        return " + ".join(wordStrs)


    def __repr__(self):
        return "SkewFieldSentence(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(str(self))


    def __cmp__(self, other):
        selfSortedWords = sorted(self.wordCtr.keys())
        otherSortedWords = sorted(other.wordCtr.keys())

        # zip takes care of differently sized lists by stopping at end of
        # shorter list
        for selfWord, otherWord in zip(selfSortedWords, otherSortedWords):
            if selfWord < otherWord:
                return -1
            if selfWord > otherWord:
                return 1

            selfWordPower = self.wordCtr[selfWord]
            otherWordPower = other.wordCtr[otherWord]

            if selfWordPower < otherWordPower:
                return -1
            if selfWordPower > otherWordPower:
                return 1

        if len(selfSortedWords) < len(otherSortedWords):
            return -1
        if len(selfSortedWords) > len(otherSortedWords):
            return 1

        return 0


    def deepcopy(self):
        cpy = SkewFieldSentence()
        for word, coef in self.wordCtr.items():
            cpy.wordCtr[word.deepcopy()] = int(coef)
        return cpy


    # delete all words with coeff 0
    def canonize(self):
        for word in self.wordCtr.keys():
            if self.wordCtr[word] == 0:
                self.wordCtr.pop(word)


    def zero():
        return SkewFieldSentence()
    zero = staticmethod(zero)


    def one():
        return SkewFieldSentence([SkewFieldWord.one()])
    one = staticmethod(one)


    def isZero(self):
        return self == SkewFieldSentence.zero()


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
        result = SkewFieldSentence()
        for word, coef in self.wordCtr.items():
            result.wordCtr[word.increasedSubs(increment)] = coef
        return result


    def plus(self, other):
        result = SkewFieldSentence(self.wordCtr)
        updateCounts(result.wordCtr, other.wordCtr)
        result.canonize() # probably not needed
        return result


    def times(self, other):
        product = SkewFieldSentence() # empty sentence
        for selfWord, selfWordCoeff in self.wordCtr.items():
            for otherWord, otherWordCoeff in other.wordCtr.items():
                updateCounts(
                    product.wordCtr,
                    { selfWord.times(otherWord) : selfWordCoeff * otherWordCoeff})
        product.canonize();
        return product


    def aInv(self):
        inverse = SkewFieldSentence()
        for word, coef in self.wordCtr.items():
            inverse.wordCtr[word] = -coef
        return inverse


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

            if match == None:
                raise ValueError("bad SkewFieldMonomial() args " + str(args))
            else:
                (numerStr, denomStr, tpowerStr) = match.groups(0)

                self.numer = SkewFieldSentence(numerStr)
                self.denom = SkewFieldSentence(denomStr)
                self.tpower = int(tpowerStr)

        elif len(args) == 3:
            self.numer = args[0].deepcopy() # type SkewFieldSentence
            self.denom = args[1].deepcopy() # type SkewFieldSentence
            self.tpower = int(args[2])      # type integer

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
        if self.tpower < other.tpower:
            return -1
        if self.tpower > other.tpower:
            return 1


        if self.numer < other.numer:
            return -1
        if self.numer > other.numer:
            return 1

        if self.denom < other.denom:
            return -1
        if self.denom > other.denom:
            return 1

        return 0


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

        sumMono = self.plusSentencePart(other)

        if sumMono.numer == SkewFieldSentence.zero():
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


    def zero():
        return SkewFieldMonomial(
            SkewFieldSentence.zero(),
            SkewFieldSentence.one(),
            0,
        )
    zero = staticmethod(zero)


    def one():
        return SkewFieldMonomial(
            SkewFieldSentence.one(),
            SkewFieldSentence.one(),
            0,
        )
    one = staticmethod(one)


    def isZero(self):
        return self == SkewFieldMonomial.zero()


    def isOne(self):
        return self == SkewFieldMonomial.one()


    def isScalar(self):
        return (self.numer == SkewFieldSentence.isScalar()
            and self.denom == SkewFieldSentence.isScalar()
            and self.tpower == 0)


    def asOneAbove(self):
        return SkewFieldPolynomial([self])


    def asPoly(self):
        return self.asOneAbove()


    def increasedSubs(self, increment):
        return SkewFieldMonomial(
            self.numer.increasedSubs(increment),
            self.denom.increasedSubs(increment),
            self.tpower,
        )


    # warning: returns SkewFieldPolynomial
    def plus(self, other):
        return SkewFieldPolynomial([self, other])


    def times(self, other):
        product = SkewFieldMonomial.zero()

        # product's tpower is easy
        product.tpower = self.tpower + other.tpower

        # new numer and denom is trickier; must increase subscripts in other's
        # letters by the amount of self.tpower (because of commutation),
        # then we can multiply

        product.numer = self.numer.times(
            other.numer.increasedSubs(self.tpower))
        product.denom = self.denom.times(
            other.denom.increasedSubs(self.tpower))

        product.canonize()
        return product


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
            self.numer.times(other.denom).plus(other.denom.times(self.numer)),
            self.denom.times(other.denom),
            self.tpower)


################################################################################
# SkewFieldPolynomial is basically a sum of differently tpowered monomials
# self.monoDict is a dict with tpowers as keys and monomials as values
# so, yes, there is some redundancy with the tpower also in the monomial
#
class SkewFieldPolynomial():


    # monos argument can be str, tuple, list, set, or dict
    def __init__(self, monos = []):
        self.monoDict = dict() # key is tpower, value is monomial

        if isinstance(monos, str):
            raise ValueError("poly str constructor not supported yet")

        # else assume monos is a collection of monomials
        else:
            # iterate to add the monos into our polynomial
            for otherMono in monos:

                # if already have mono of that tpower, must add them together
                if otherMono.tpower in self.monoDict:
                    myMono = self.monoDict.pop(otherMono.tpower)
                    self.monoDict[otherMono.tpower] \
                        = myMono.plusSentencePart(otherMono)

                # else don't have mono of that tpower; simple insert
                else:
                    self.monoDict[otherMono.tpower] = otherMono.deepcopy();

        self.canonize()


    def __str__(self):
        # temporarily using "++" as super-plus operator to make it more apparent
        # that we are adding polys
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
        # remove monomials that are zero
        for tpower, mono in self.monoDict.items():
            if mono.isZero():
                self.monoDict.pop(tpower)


    def zero():
        return SkewFieldPolynomial()
    zero = staticmethod(zero)


    def one():
        return SkewFieldPolynomial([SkewFieldMonomial.one()])
    one = staticmethod(one)


    def isZero(self):
        return self == SkewFieldPolynomial.zero()


    def isOne(self):
        return self == SkewFieldPolynomial.one()


    def isScalar(self):
        return self.isZero() \
           or (len(self.monoDict) == 1 and self.monoDict.values()[0].isScalar())


    def asPoly(self):
        return self.deepcopy()


    def asMonoList(self):
        return list(reversed(sorted(self.monoDict.values())))


    def increasedSubs(self, increment):
        result = SkewFieldPolynomial()
        for power, mono in self.monoDict.items():
            result.monoDict[power] = mono.increasedSubs(increment)
        return result


    def plus(self, other):
        return SkewFieldPolynomial(
            self.monoDict.values() + other.monoDict.values())


    def times(self, other):
        product = []
        for selfMono in self.monoDict.values():
            for otherMono in other.monoDict.values():
                product.append(selfMono.times(otherMono))

        return SkewFieldPolynomial(product)

    def aInv(self):
        result = []
        for mono in self.monoDict.values():
            result.append(mono.aInv())

        return SkewFieldPolynomial(result)


    def degree(self):
        if len(self.monoDict.keys()) == 0:
            return 0
        else:
            return sorted(self.monoDict.keys())[-1]


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
            numerator = numerator.plus(denominator.times(mono.asPoly()).aInv())
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

    print("LETTER ############################################################")

    ltrA0Str = "a_0"

    ltrA0 = SkewFieldLetter("a", 0)
    print("ltrA0 = " + str(ltrA0))

    # basic construction-representation test
    assert(str(ltrA0) == ltrA0Str)

    ltrA0Again = SkewFieldLetter(ltrA0Str)
    ltrA1 = SkewFieldLetter("a_1")
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

    # test operations did not violate immutability
    assert(ltrA0Again == ltrA0)
    assert(str(ltrA0Again) == str(ltrA0))

    print("WORD ##############################################################")

    # basic construction-representation test

    wrd1Str = "a_0^1 * b_1^2"

    ltrs1 = [
        SkewFieldLetter("a", 0),
        SkewFieldLetter("b", 1),
        SkewFieldLetter("b", 1),
    ]

    wrd1 = SkewFieldWord(ltrs1)
    print("wrd1 = " + str(wrd1))

    assert(str(wrd1) == wrd1Str)
    assert(wrd1 == wrd1)

    # test identity comparison
    wrd1Again = SkewFieldWord(wrd1Str)
    print("wrd1Again = " + str(wrd1Again))
    assert(wrd1 == wrd1Again)

    # test increasedSubs
    wrd2Str = "a_2^1 * b_3^2"
    wrd2 = wrd1.increasedSubs(2)
    print("wrd2 = " + str(wrd2))
    assert(str(wrd2) == wrd2Str)

    # test alternate constructor and canonization

    wrd3Str = "b_1^3 * c_2^1"

    ltrs3Dict = {
        SkewFieldLetter("b", 1) : 3,
        SkewFieldLetter("c", 2) : 1,
        SkewFieldLetter("d", 3) : 0,
    }

    wrd3 = SkewFieldWord(ltrs3Dict)
    print("wrd3 = " + str(wrd3))
    assert(str(wrd3) == wrd3Str)

    # test mInv and times a bit

    wrd1MInvStr = "a_0^-1 * b_1^-2"
    wrd1MInv = wrd1.mInv()
    print("wrd1MInv = " + str(wrd1MInv))
    assert(str(wrd1MInv) == wrd1MInvStr)
    assert(wrd1 == wrd1.mInv().mInv())
    print("wrd1 * wrd1MInv = " + str(wrd1.times(wrd1MInv)))
    assert(SkewFieldWord.one() == wrd1.times(wrd1MInv))

    # test times a bit more

    wrd4Str = "a_0^1 * b_1^5 * c_2^1"

    wrd4 = wrd1.times(wrd3)
    print("wrd1 * wrd3 = wrd4 = " + str(wrd4))
    assert(str(wrd4) == wrd4Str)

    wrd5Str = "a_0^1 * b_1^5 * c_2^1"

    wrd5 = wrd3.times(wrd1)
    print("wrd3 * wrd1 = wrd5 = " + str(wrd5))
    assert(str(wrd5) == wrd5Str)

    print("SENTENCE ##########################################################")

    # test wrd+wrd => sentence

    snt1Str = "1 * a_0^1 + 1 * b_0^1"
    snt1 = SkewFieldWord("a_0^1").plus(SkewFieldWord("b_0^1"))
    print("wrdA + wrdB = snt1 = " + str(snt1))
    assert(str(snt1) == snt1Str)

    snt1Again = SkewFieldSentence(snt1Str)
    assert(snt1 == snt1Again)
    assert(str(snt1Again) == snt1Str)

    # test some scalar sentences

    snt2Str = "2"
    snt2 = SkewFieldSentence(snt2Str)
    print("snt2 = " + str(snt2))
    assert(str(snt2) == snt2Str)

    snt3Str = "3 + 4 * a_0^5 + 6 * a_0^7 * b_1^8"
    snt3 = SkewFieldSentence(snt3Str)
    print("snt3 = " + str(snt3))
    assert(str(snt3) == snt3Str)

    assert(not snt1.isScalar())
    assert(snt2.isScalar())
    assert(not snt3.isScalar())

    # test increased subs

    snt99Str = "3 + 4 * a_-2^5 + 6 * a_-2^7 * b_-1^8"
    snt99 = snt3.increasedSubs(-2)
    print("snt99 = " + str(snt99))
    assert(str(snt99) == snt99Str)

    # test aInv and plus a bit

    snt3AInvStr = "-3 + -4 * a_0^5 + -6 * a_0^7 * b_1^8"

    snt3AInv = snt3.aInv()
    print("snt3AInv = " + str(snt3AInv))
    assert(str(snt3AInv) == snt3AInvStr)

    snt4Str = "0"

    snt4 = snt3.plus(snt3AInv)
    print("snt3 + snt3AInv = snt4 = " + str(snt4))
    assert(str(snt4) == snt4Str)

    # test plus a bit more

    snt5Str = "2 + 3 * b_0^1 + 4 * c_0^1"
    snt5 = SkewFieldSentence(snt5Str)
    print("snt5 = " + str(snt5))
    assert(str(snt5) == snt5Str)

    snt6Str = "2 + 1 * a_0^1 + 4 * b_0^1 + 4 * c_0^1"
    snt6 = snt1.plus(snt5)
    print("snt6 = " + str(snt6))
    assert(str(snt6) == snt6Str)

    # test commutivity of plus
    assert(snt1.plus(snt5) == snt5.plus(snt1))

    # test times - associativity and commutivity

    snt7Str = "3 + 5 * a_0^7"
    snt7 = SkewFieldSentence(snt7Str)
    print("snt7 = " + str(snt7))
    assert(str(snt7) == snt7Str)

    snt8Str = "6 * a_0^1 + 10 * a_0^7 * b_0^1 + 10 * a_0^8 + 6 * b_0^1"
    snt8 = snt2.times(snt1).times(snt7)
    print("(snt2 * snt1) * snt7 = snt8 = " + str(snt8))
    assert(str(snt8) == snt8Str)

    snt9 = snt2.times(snt1.times(snt7))
    print("snt2 * (snt1 * snt7) = snt9 = " + str(snt9))
    assert(snt9 == snt8)

    snt10 = snt7.times(snt2.times(snt1))
    print("snt7 * snt2 * snt1 = snt10 = " + str(snt10))
    assert(snt10 == snt8)

    print("MONOMIAL ##########################################################")

    # basic construction-representation test

    # (1 * a_0^1 + 1 * b_0^1) / (3 + 5 * a_0^7) * T^-2
    mono1Str = "(" + str(snt1) + ") / (" + str(snt7) + ") * T^-2"
    mono1 = SkewFieldMonomial(snt1, snt7, -2)
    print("mono1 = " + str(mono1))
    assert(str(mono1) == mono1Str)

    mono1Again = SkewFieldMonomial(mono1Str)
    assert(str(mono1Again) == mono1Str)

    # test mInv and test times a bit

    mono1MInvStr = "(3 + 5 * a_2^7) / (1 * a_2^1 + 1 * b_2^1) * T^2"
    mono1MInv = mono1.mInv()
    print("mono1.mInv() = mono1Inv = " + str(mono1MInv))
    assert(str(mono1MInv) == mono1MInvStr)

    monoOneStr = str(SkewFieldMonomial.one())

    mono2 = mono1.times(mono1MInv)
    print("mono1 * mono1MInv = mono2 = " + str(mono2))
    assert(str(mono2) == monoOneStr)

    mono3 = mono1MInv.times(mono1)
    print("mono1MInv * mono1 = mono3 = " + str(mono3))
    assert(str(mono3) == monoOneStr)

    # test times some more

    mono4Str = "(" + str(snt1) + ") / (1) * T^3"
    mono4 = SkewFieldMonomial(snt1, SkewFieldSentence.one(), 3)
    print("mono4 = " + str(mono4))
    assert(str(mono4) == mono4Str)

    mono5Str = "(" + str(snt1.times(snt1)) + ") / (" + str(snt7) + ") * T^1"
    mono5 = mono1.times(mono4.increasedSubs(2))
    print("mono5Str = " + str(mono5Str))
    print("mono1 * mono4.incSubs(2) = mono5 = " + str(mono5))
    assert(str(mono5) == mono5Str)

    # TODO: more monomial testing

    print("POLYNOMIAL ########################################################")

    poly1 = SkewFieldPolynomial([mono1, mono2, mono1])
    print("poly1 = " + str(poly1))

    print(SkewFieldPolynomial.zero().isZero())

    return 0


if __name__ == "__main__":
    sys.exit(main())


