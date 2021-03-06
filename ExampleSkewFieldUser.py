#! /usr/bin/python
"""
initial author:   Jacob Egner
initial date:     2011-07-08

this is an example on how to use and add to the classes in SkewField.py
"""


import sys
import getopt
import re
import collections

# need this to access the SkewField stuff
from SkewField import *

# you can define new member functions
def simplifiedLetterStr(self):
    if self.sub == 0:
        return SkewFieldLetter.alphaAsStr(self.alpha)
    return str(self)


def main(argv=None):
    # here is how you add a member function to a class from SkewField.py; note
    # that the name can be different; note that you can also override existing
    # functions
    SkewFieldLetter.simplifiedStr = simplifiedLetterStr

    # you should call SkewField.py's test battery before your own
    # unless your functionality DELIBERATELY and CORRECTLY violates old
    # behavior
    SkewFieldMain()

    print("")
    print("")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@ ExampleSkewFieldUser.py")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("")
    print("")

    print("LETTER ############################################################")

    ltr = SkewFieldLetter("a_0")
    print("ltr in usual string form: " + str(ltr))
    print("ltr in simplified string form: " + ltr.simplifiedStr())

    assert(str(ltr) == "a_0")
    assert(ltr.simplifiedStr() == "a")

    return 0


if __name__ == "__main__":
    sys.exit(main())


