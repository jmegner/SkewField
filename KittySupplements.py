#! /usr/bin/python
"""
initial author:   Kitty Yang
initial date:     2011-07-15

note: grand polynomial division works!!!

"""


FileVersion = "0.03"


import sys
import getopt
import re
import collections

import SkewField
from SkewField import *


# JME_REMARK: this was merged after I changed 'p' to 'power' in the if-statement
# the use of 'p' makes me suspicious how well this function was tested;
# I have left the (corrected) function here so you will notice this remark
# and you might have to change it further after more testing
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


# JME_REMARK: in reducedAtLetter and the multiple reduced functions, I changed
# the "relation" argument to "relations" since the variable does hold multiple
# relations

# JME_REMARK: in raisedTo, you had "letterPower*power"; you need a space on
# each side of binary operators






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


    # JME_REMARK: merged but still left in your file
    SkewFieldWord.firstAbnormalLetter = firstAbnormalLetter

    # JME_REMARK: no need to do input from file;
    #
    # besides, we want our test data to be here until it grows so big
    # it is a huge hassle to keep it here;
    #
    # also, I changed "relation" to "relations1"

    # so we can refer to SkewField's global variables without typing
    # "SkewField." every time
    j = SkewField


    # JME_REMARK: you need asserts; for instance:
    # assert(str(wrd1Reduced) == wrd1ReducedStr)
    #
    # after you have added asserts, I will merge in these tests
    # even if you don't want to manually check huuuge results like quot, at
    # least capture the string rep of quot as it now and assert quot with that
    # string rep; that way we will be alerted if quot ever changes; but be sure
    # to make a comment that the assert is not confident

    #test reduce words
    print(j.wrd1)
    print(j.wrd1.reduced(j.relations1))

    #test reduce sentences
    snt1Reduced = j.snt1.reduced(j.relations1)
    print(snt1Reduced)
    print(snt1Reduced.increasedSubs(5))

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
    print(lead2.asPoly().quotient(lead1.asPoly()))


if __name__ == "__main__":
    sys.exit(main())

