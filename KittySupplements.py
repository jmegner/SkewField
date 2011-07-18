#! /usr/bin/python
"""
initial author:   Kitty Yang
initial date:     2011-07-15

note: grand polynomial division still works!!!

"""


FileVersion = "0.04"


import sys
import getopt
import re
import collections

import SkewField
from SkewField import *


# I thought about doing testing on firstAbnormalLetter, but it's difficult to
# explicitly check the function for accuracy
# However, calling all the reduced functions that depend on this one seems to
# implicity verify. I reduce already reduced words, and check to see if they
# are equal. This feels sufficient.
def firstAbnormalLetter(self, relations):
    for letter in sorted(self.letterCtr.keys()):
        alpha = letter.alpha
        power = self.letterCtr[letter]
        subscript = letter.sub
        (minSubscript, minPower) = relations[alpha].firstOfAlpha(alpha)
        (maxSubscript, maxPower) = relations[alpha].lastOfAlpha(alpha)

        if not(
            (subscript < minSubscript and power in range (abs(minPower))) or
            (subscript > maxSubscript and power in range (abs(maxPower))) or
            (subscript in range (minSubscript, maxSubscript))):

            return letter

        return None
# redid sentence's reduce function because I was not fully aware of how the
# coefficients in sentence works. Hooray for testing.

# SENTENCE #####################################################################
def reduced(self, relations):
    nSentence = SkewFieldSentence()
    for word in self.wordCtr.keys():
        updateCounts(nSentence.wordCtr,
                     { word.reduced(relations) : self.wordCtr[word]})
    nSentence.canonize();
    return nSentence
 
# POLYNOMIAL ###################################################################

# Will the degree function of poly change?
# Applies polynomial long division algorithm to output quotient and remainder
def quotientRemainder(self, denominator):
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

    return (SkewFieldPolynomial(result), numerator)

# For one reason or another, I can't name this method quotient. It will remain
# almost t-less.
def quotien(self, denominator):
    return self.quotientRemainder(denominator)[0]


################################################################################
# MAIN
################################################################################

def main(argv=None):

    SkewFieldMain()

    print("")
    print("")
    print("###################################################################")
    print("# KittySupplements test battery")
    print("###################################################################")
    print("")
    print("")

    SkewFieldWord.firstAbnormalLetter = firstAbnormalLetter
    SkewFieldSentence.newreduced = reduced
    SkewFieldPolynomial.quotientRemainder = quotientRemainder
    SkewFieldPolynomial.quot = quotien

    j = SkewField


    #test reduce words
    print("wrd1 = " + str(j.wrd1))
    wrd1reduced = j.wrd1.reduced(j.relations1)
    print("wrd1 reduced = " + str(wrd1reduced))

    wrd1reducedstr = "b_0^1 * b_1^1"
    assert(wrd1reduced == wrd1reduced.reduced(j.relations1))
    assert(str(wrd1reduced) == wrd1reducedstr)


    print("wrd2 = " + str(j.wrd2))
    wrd2reduced = j.wrd2.reduced(j.relations1)
    print("wrd2 reduced = " + str(wrd2reduced))

    wrd2reducedstr = "b_0^-1 * b_1^1 * b_3^1"
    assert(wrd2reduced == wrd2reduced.reduced(j.relations1))
    assert(str(wrd2reduced) == wrd2reducedstr)


    #test reduce sentences
    print("snt1 = " + str(j.snt1))
    snt1reduced = j.snt1.newreduced(j.relations1)
    print("snt1 reduced = " + str(snt1reduced))

    snt1reducedstr = "1 * b_0^1 + 1 * b_0^1 * b_1^-1"
    assert(snt1reduced == snt1reduced.newreduced(j.relations1))
    assert(str(snt1reduced) == snt1reducedstr)


    print("snt2 = " + str(j.snt2))
    snt2reduced = j.snt2.newreduced(j.relations1)
    print("snt2 reduced = " + str(snt2reduced))

    snt2reducedstr = "2"
    assert(snt2reduced == snt2reduced.newreduced(j.relations1))
    assert(str(snt2reduced) == snt2reducedstr)

    #test reduce monomials
    print("mono1 = " + str(j.mono1))
    mono1reduced = j.mono1.reduced(j.relations1)
    print("mono1 reduced = " + str(mono1reduced))

    #this line goes past 80 characters, but I don't know how to wrap it
    mono1reducedstr = "(1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1 + 1 * b_0^7 * b_1^-7) * T^-2"
    assert(mono1reduced == mono1reduced.reduced(j.relations1))
    assert(str(mono1reduced) == mono1reducedstr)

    print("mono4 = " + str(j.mono4))
    mono4reduced = j.mono4.reduced(j.relations1)
    print("mono4 reduced = " + str(mono4reduced))

    mono4reducedstr = "(1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1) * T^3"
    assert(mono4reduced == mono4reduced.reduced(j.relations1))
    assert(str(mono4reduced) == mono4reducedstr)

    #test reduce poly
    print("poly1 = " + str(j.poly1))
    poly1reduced = j.poly1.reduced(j.relations1)
    print("poly1 reduced = " + str(poly1reduced))

    poly1reducedstr = "(1) / (1) * T^0 ++ (1 * b_0^1 + 1 * b_0^1 * b_1^-1) / (1 + 1 * b_0^7 * b_1^-7) * T^-2"
    assert(poly1reduced == poly1reduced.reduced(j.relations1))
    assert(str(poly1reduced) == poly1reducedstr)


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

    assert(trouble2 == SkewFieldPolynomial.zero())


    cross1 = trouble1AInv.numer.times(lead2.denom)
    cross2 = trouble1AInv.denom.times(lead2.numer)

    assert(cross1 != cross2)

    #tests polynomial long division

    poly2 = SkewFieldPolynomial([mono1.mInv(), mono4, mono4])
    print("poly2 = " + str(poly2))
    poly3 = SkewFieldPolynomial([mono1.mInv(), mono3, mono4, mono4])
    print("poly3 = " + str(poly3))

    quotient = poly3.quot(poly2)
    (quo, remainder) = poly3.quotientRemainder(poly2)
    print("quotient of poly3 and poly2 = " + str(quotient))
    print(remainder)
    assert(quotient == quo)
    assert(poly3 == poly2.times(quotient).plus(remainder))


if __name__ == "__main__":
    sys.exit(main())

