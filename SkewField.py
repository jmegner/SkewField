#! /usr/bin/python
#
# initial author:   Jacob Egner
# initial date:     2011-07-08
#
# note: even though the standard python class Counter would be wonderful for
# keeping counts of letters' powers and words' coefficients, consumers of this
# code might have an old version of python that does not have the Counter class
#
# note: here a semi-accurate list of functions in each SkewField-ish class:
#   __init__
#   __str__
#   __repr__
#   __hash__
#   __cmp__
#   deepcopy
#   canonize
#   zero
#   one
#   isZero
#   isOne
#   isScalar
#   asOneAbove
#   asPoly
#   increasedSubs
#   plus
#   times
#   addInverse
#   multInverse
#


FileVersion = "0.02"


import sys
import getopt
import re
import collections
#from collections import Counter


# to help overcome our heartfelt loss of the treasured Counter class...
# one advantage is some auto-canonization from deleting zero-valued items
def updateCounts(counter, otherGuy):
    # step1: merge

    if isinstance(otherGuy, dict):
        for key, value in otherGuy.items():
            counter[key] = counter.get(key, 0) + value;
    # else assume list/tuple/set
    else:
        for key in otherGuy:
            counter[key] = counter.get(key, 0) + 1;

    # step2: simplify by removing zero-valued items
    for key, value in counter.items():
        if counter[key] == 0:
            counter.pop(key)


class SkewFieldLetter():
    # SkewFieldLetter is an alphabetical identifier and a subscript
    # this representation can neither be a zero nor a one
    #
    # example: b_1
    #   'b' is the alpha
    #   '1' is the sub

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            m = re.match("([a-z]+)_?(-?\d*)", args[0].strip())
            if m == None:
                raise ValueError("bad SkewFieldLetter() args " + str(args))
            else:
                (self.alpha, subAsString) = m.groups(0)
                self.sub = int(subAsString)
        elif len(args) == 2:
            self.alpha = str(args[0])    # string of alphabetic chars
            self.sub = int(args[1])      # integer subscript
        else:
            raise ValueError("bad SkewFieldLetter() args " + str(args))


    # so class can be prettily printed
    def __str__(self):
        return str(self.alpha) + "_" + str(self.sub)


    def __repr__(self):
        return "SkewFieldLetter(\"" + str(self) + "\")"


    # so class can be a key in a dict or Counter object
    def __hash__(self):
        return hash(str(self))


    # so class is sortable by the 'sorted' function
    # simple comparison based alpha length, then alpha value, then sub
    def __cmp__(self, other):
        if len(self.alpha) < len(other.alpha):
            return -1
        if len(self.alpha) > len(other.alpha):
            return 1
        if self.alpha < other.alpha:
            return -1
        if self.alpha > other.alpha:
            return 1
        if self.sub < other.sub:
            return -1
        if self.sub > other.sub:
            return 1
        return 0


    def isZero(self):
        return False


    def isOne(self):
        return False


    def isScalar(self):
        return False


    def deepcopy(self):
        return SkewFieldLetter(self.alpha, self.sub)


    def asWord(self):
        return SkewFieldWord([self])


    def asPoly(self):
        return self.asWord().asPoly()


class SkewFieldWord():
    # SkewFieldWord is the product of SkewFieldLetters;
    # a SkewFieldWord can be 1 (empty letterCtr), but it can not be zero;
    # for a zero, you need to go to the level of SkewFieldSentence
    #
    # example: b_1^2 * c_2;
    #   'b_1' and 'c_2' are the letters (stored as letterCtr keys)
    #   2 and 1 are the powers (storted as letterCtr values)

    # letters argument can be str, tuple, list, set, or dict
    def __init__(self, letters = []):
        self.letterCtr = dict() # key is SkewFieldLetter, value is power

        if isinstance(letters, str):
            letters = letters.strip()

            if letters == "1":
                self = SkewFieldWord.one()
            else:
                for letterWithPower in letters.split(" * "):
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

        # zip takes care of differently sized lists by stopping at end of shorter list
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
        for letter, power in self.items():
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

    def mInverse(self):
        inverse = SkewFieldWord()
        for letter, power in self.letterCtr.items():
            inverse.letterCtr[letter] = -power
        return inverse


class SkewFieldSentence():
    # SkewFieldSentence is the sum of SkewFieldWords;
    # can also be thought of as a polynomial of SkewFieldLetters
    #
    # SkewFieldSentence can be a one (identity word with coefficient = 1)
    # SkewFieldSentence can be a zero (empty wordCtr)
    #
    # example: 3 * a_0^1 + 2 * b_1^2 * c_2^1 ;
    #   'a_0^1' and 'b_1^2 * c_2^1' are the words (stored as wordCtr keys)
    #   3 and 2 are the coefficients (storted as wordCtr values)


    # words argument can be str, tuple, list, set, or dict
    def __init__(self, words = []):
        self.wordCtr = dict() # key is SkewFieldWord, value is coef

        if isinstance(words, str):
            words = words.strip()
            if words == "0":
                self = SkewFieldSentence.zero()
            elif words == "1":
                self = SkewFieldSentence.one()
            else:
                for coefWithWord in words.split(" + "):
                    (coef, wordStr) = coefWithWord.split(" * ", 1)
                    updateCounts(
                        self.wordCtr,
                        { SkewFieldWord(wordStr) : int(coef) }
                    )
        else:
            updateCounts(self.wordCtr, words)

        self.canonize()


    def __str__(self):
        if self.isScalar():
            return str(self.wordCtr.values()[0])

        wordStrs = list()
        for word in sorted(self.wordCtr.keys()):
            wordStrs.append(str(self.wordCtr[word]) + " * " + str(word))
        return " + ".join(wordStrs)


    def __repr__(self):
        return "SkewFieldSentence(\"" + str(self) + "\")"


    def __hash__(self):
        return hash(str(self))


    def __cmp__(self, other):
        selfSortedWords = sorted(self.wordCtr.keys())
        otherSortedWords = sorted(other.wordCtr.keys())

        # zip takes care of differently sized lists by stopping at end of shorter list
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
        for word, coef in self.items():
            cpy.wordCtr[word.deepcopy()] = coef
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
        print("    sentence.times: " + str(product))
        return product


    def aInverse(self):
        inverse = SkewFieldSentence()
        for word, coef in self.wordCtr.items():
            inverse.wordCtr[word] = -coef
        return inverse


class SkewFieldMonomial():
    # convention is for T to be on right-hand side and the SkewFieldSentence
    # numerator and denominator to be on the left-hand side

    def __init__(
            self,
            numer = SkewFieldSentence.zero(),
            denom = SkewFieldSentence.one(),
            tpower = 0,
    ):
        self.numer = numer      # type SkewFieldSentence
        self.denom = denom      # type SkewFieldSentence
        self.tpower = tpower    # type integer
        self.canonize()


    def __str__(self):
        return str(self.numer) + " / " + str(self.denom) \
            + " * T^" + str(self.tpower)


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
            self.tpower = 0


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

        print("    mono.times:")
        print("        beforeCanon: " + str(product))
        product.canonize()
        print("        afterCanon:  " + str(product))
        return product


    def aInverse(self):
        return SkewFieldMonomial(
            self.numer.aInverse(),
            self.denom,
            self.tpower,
        )


    def mInverse(self):
        return SkewFieldMonomial(
            self.denom.increasedSubs(-self.tpower),
            self.numer.increasedSubs(-self.tpower),
            -self.tpower,
        )


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

    assert(str(ltrA0) == ltrA0Str)

    ltrA0Again = SkewFieldLetter(ltrA0Str)
    ltrA1 = SkewFieldLetter("a_1")
    ltrB0 = SkewFieldLetter("b_0")

    blah = ltrA0.deepcopy()

    # test comparisons
    assert(ltrA0 == ltrA0Again)
    assert(ltrA0 == ltrA0)
    assert(ltrA0 < ltrA1)
    assert(ltrA1 > ltrA0)
    assert(ltrA1 < ltrB0)
    assert(ltrB0 > ltrA1)

    print("WORD ##############################################################")

    wrd1Str = "a_0^1 * b_1^2"

    ltrs1 = [
        SkewFieldLetter("a", 0),
        SkewFieldLetter("b", 1),
        SkewFieldLetter("b", 1),
    ]

    # make word with list of letterCtr
    wrd1 = SkewFieldWord(ltrs1)
    print("wrd1 = " + str(wrd1))

    assert(str(wrd1) == wrd1Str)
    assert(wrd1 == wrd1)

    wrd1Again = SkewFieldWord(wrd1Str)
    print("wrd1Again = " + str(wrd1Again))
    assert(wrd1 == wrd1Again)

    wrd2Str = "a_2^1 * b_3^2"
    wrd2 = wrd1.increasedSubs(2)
    print("wrd2 = " + str(wrd2))
    assert(str(wrd2) == wrd2Str)

    wrd3Str = "b_1^3 * c_2^1"

    ltrs3Dict = {
        SkewFieldLetter("b", 1) : 3,
        SkewFieldLetter("c", 2) : 1,
        SkewFieldLetter("d", 3) : 0,
    }

    wrd3 = SkewFieldWord(ltrs3Dict)
    print("wrd3 = " + str(wrd3))
    assert(str(wrd3) == wrd3Str)

    wrd1MInvStr = "a_0^-1 * b_1^-2"
    wrd1MInv = wrd1.mInverse()
    print("wrd1MInv = " + str(wrd1MInv))
    assert(str(wrd1MInv) == wrd1MInvStr)
    assert(wrd1 == wrd1.mInverse().mInverse())
    print("wrd1 * wrd1MInv = " + str(wrd1.times(wrd1MInv)))
    assert(SkewFieldWord.one() == wrd1.times(wrd1MInv))

    wrd4Str = "a_0^1 * b_1^5 * c_2^1"

    wrd4 = wrd1.times(wrd3)
    print("wrd1 * wrd3 = wrd4 = " + str(wrd4))
    assert(str(wrd4) == wrd4Str)

    print("SENTENCE ##########################################################")

    snt1Str = "1 * a_0^1 + 1 * b_0^1"
    snt1 = SkewFieldWord("a_0^1").plus(SkewFieldWord("b_0^1"))
    print("wrdA + wrdB = snt1 = " + str(snt1))
    assert(str(snt1) == snt1Str)

    return 1
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    sentence2 = sentence1.times(sentence1)
    print("sentence1 * sentence1 = sentence2 = " + str(sentence2))

    mono1 = SkewFieldMonomial(sentence2, sentence1, 2)
    print("mono1 = " + str(mono1))

    # TODO: uncomment when have poly class
    #print("mono1 + -mono1 = " + str(mono1.plus(mono1.aInverse())))

    mono2 = SkewFieldMonomial(sentence1, SkewFieldSentence.one(), 3)
    print("mono2  = " + str(mono2))

    mono2MultInv = mono2.mInverse()
    print("mono2MultInv  = " + str(mono2MultInv))

    print("mono2 * mono2.mInv() = " + str(mono2.times(mono2.mInverse())))
    print("mono2 * mono2MultInv = " + str(mono2.times(mono2MultInv)))

    return 0


if __name__ == "__main__":
    sys.exit(main())


