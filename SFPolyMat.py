#! /usr/bin/env python

FileVersion = "0.2"

import sys
import getopt
import re
import collections

import SkewField
from SkewField import *

#use for column operations
def rightQuot(self, denominator):
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

    return SkewFieldPolynomial(result)

#use for row operations
def leftQuot(self, denominator):
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

    return SkewFieldPolynomial(result)

#use for col operations
def rightQuotient(self, denominator):
    if(self.degree() >= denominator.degree()):
        leadDenominator = denominator.monoDict.get(
            denominator.degree(),
            SkewFieldMonomial.zero())
        leadNumerator = self.monoDict.get(
            self.degree(),
            SkewFieldMonomial.zero())
        mono = leadDenominator.mInv().times(leadNumerator)
        return mono.asPoly()
    else:
        return SkewFieldPolynomial.zero()

#use for row operations
def leftQuotient(self, denominator):
    if(self.degree() >= denominator.degree()):
        leadDenominator = denominator.monoDict.get(
            denominator.degree(),
            SkewFieldMonomial.zero())
        leadNumerator = self.monoDict.get(
            self.degree(),
            SkewFieldMonomial.zero())
        mono = leadNumerator.times(leadDenominator.mInv())
        return mono.asPoly()
    else:
        return SkewFieldPolynomial.zero()


class SFPolyMat():

#-------initializer-------#
    def __init__(self, mat, rels):
        self.mat = mat
        self.rels = rels

    def __str__(self):
        [[str(poly) for poly in rowArray] for rowArray in self.mat]

#-------methods to be used with skew field-------#

    def reduce(self):
        for row in range(self.nRows()):
            for col in range(self.nCols()):
                self.mat[row][col] = self.mat[row][col].reduced(self.rels)

    #swaps rows i and j in the matrix. Implimented over skew field
    def swapRows(self, i, j):
        temp = self.mat[i]
        self.mat[i] = self.mat[j]
        self.mat[j] = temp

    #swaps columns i and j in the matrix. Implimented over skew field
    def swapColumns(self,i, j):
        for k in range(len(self.mat)):
            #print "in swapColumns at " + str(k)
            temp = self.mat[k][i]
            self.mat[k][i] = self.mat[k][j]
            self.mat[k][j] = temp

    #scales row by multiplier. Implimented over skew field.
    def scaleRow(self, row, multiplier):
        for col in range(self.nCols()):
            poly = multiplier.times(self.mat[row][col])
            self.mat[row][col] = poly.reduced(self.rels)

    #scales col by multiplier. Implimented over skew field
    def scaleColumn(self, col, multiplier):
        for row in range(self.nRows()):
            nPoly = self.mat[row][col].times(multiplier)
            self.mat[row][col] = nPoly.reduced(self.rels)

    #adds multiplier*j to i. Should work over skew field
    def addMultOfRow(self, i, j, multiplier):
        #print "adding " + str(multiplier) +"*" +str(j) + " to row " +str(i)
        temp = []
        for col in range(self.nCols()):
            poly = multiplier.times(self.mat[j][col])
            temp.append(poly.reduced(self.rels))
        for col in range(self.nCols()):
            poly = self.mat[i][col].plus(temp[col])
            self.mat[i][col] = poly.reduced(self.rels)
        #print self.mat

    #adding mult*j to i. Should work over skew field
    def addMultOfColumn(self, i, j, multiplier):
        #print "adding " + str(multiplier) +"*" + str(j) + " to col " + str(i)
        temp = []
        for row in range(self.nRows()):
            poly = self.mat[row][j].times(multiplier)
            temp.append(poly.reduced(self.rels))
        for row in range(self.nCols()):
            poly = self.mat[row][i].plus(temp[row])
            self.mat[row][i] = poly.reduced(self.rels)
        #print self.mat

    #number of rows in the matrix. OK over skew field
    def nRows(self):
       return len(self.mat)

    #number of columns in the matrix. OK over skew field
    def nCols(self):
       return self.nRows()

    #will return the quotient to multiply row by. OK over skew field
#    def quotient(self,poly1,poly2):
#        q = poly1.quotient(poly2)
#        return q.reduced(self.rels)


#-------methods to diagonalize-------#

    def clearRow(self, row):
        minDegree = 0
        for col in range(self.nCols()):
            #print ("in clearRow " + str(row) + ", " + str(col) + " ok")
            if not self.mat[row][col].isZero():
                #print "in row " + str(row) + ", " + str(col) + " is not zero"
                if self.mat[row][col].lowestPower() < minDegree:
                    minDegree = self.mat[row][col].lowestPower()
        #print minDegree
        tPower = SkewFieldPolynomial([SkewFieldMonomial(
            SkewFieldSentence.one(),SkewFieldSentence.one(),-minDegree)])
        #print tPower
        self.scaleRow(row, tPower)

    def killNegatives(self):
        for row in range(self.nRows()):
            #print (str(row) + " ok")
            self.clearRow(row)
        print "done killing negatives"

   #finds position of minimum degree elt of matrix starting at (i,i)
    def minPosition(self, i):
        mindeg = -2
        minPosition = (i, i)
        for row in range(i,self.nRows()):
            for col in range(i, self.nCols()):
                #print self.degree(self.mat[row][col])
                if not self.mat[row][col].isZero():
                    if mindeg == -2 or self.mat[row][col].degree() < mindeg:
                        mindeg = self.mat[row][col].degree()
                        minPosition = (row, col)
                        #print minPosition
        return minPosition

    def minToTop(self, row):
        minRowCol = self.minPosition(row)
        #print str(min)
        self.swapRows(row, minRowCol[0])
        #print "swapped rows"
        self.swapColumns(row ,minRowCol[1])
        #print "swapped columns"

    def killColEntry(self, i, j):
        q = leftQuotient(self.mat[j][i], self.mat[i][i])
        q = q.reduced(self.rels)
        q = q.aInv()
        #print "result of dividing col entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfRow(j, i, q.reduced(self.rels))

    def killRowEntry(self, i, j):
        q = rightQuotient(self.mat[i][j], self.mat[i][i])
        q = q.reduced(self.rels)
        q = q.aInv()
        #print "result of dividing row entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfColumn(j, i, q.reduced(self.rels))

    def killRowCol(self,i):
        self.minToTop(i)
        #print self.mat
        for row in range(i + 1, self.nRows()):
            if not self.mat[row][i].isZero():
                self.killColEntry(i, row)
                self.killRowCol(i)
        for col in range(i + 1, self.nCols()):
            if not self.mat[i][col].isZero():
                self.killRowEntry(i, col)
                self.killRowCol(i)
        #print self.mat

    def diagonalize(self):
        self.reduce()
        self.killNegatives()
        for i in range(self.nRows()):
            #print "in at " + str(i)
            self.killRowCol(i)

    #Only applicable for diagonal matrix
    def delta1(self):
        det = 0
        for row in range(self.nCols()):
            det += self.mat[row][row].powerDiff()
        return det


################################################################################
# MAIN
################################################################################


SkewFieldPolynomial.leftQuotient = leftQuotient
SkewFieldPolynomial.rightQuotient = rightQuotient

def main(argv=None):

    relations3_1 = [SkewFieldWord("a_0^1 * b_0^-1 * b_1^1"),
                    SkewFieldWord("b_0^1 * b_1^-1 * b_2^1")]

    testmatrix3_1 = [[SkewFieldPolynomial("(1 + -1 * b_1^-1) / (1) * T^0"),
                      SkewFieldPolynomial("(-1 * a_0^-1 * b_1^-1) / (1) * T^0"),
                      SkewFieldPolynomial("0")],
                     [SkewFieldPolynomial("(1 + -1 * a_1^1) / (1) * T^0"),
                      SkewFieldPolynomial("(1) / (1) * T^1"),
                      SkewFieldPolynomial("(-1 * b_0^1 * b_1^-1 + 1 * b_1^-1) / (1 + -1 * b_1^-1) * T^1 ++ (-1 * b_0^-1 + 1 * b_0^-1 * b_2^-1) / (1 + -1 * b_1^-1) * T^0")],
                     [SkewFieldPolynomial("(1 + -1 * a_1^-1 * b_1^1) / (1) * T^0"),
                      SkewFieldPolynomial("(-1 * a_1^-1 * b_1^1) / (1) * T^1 ++ (1 * a_1^-1 * b_1^1) / (1) * T^0"),
                      SkewFieldPolynomial("(1 + -1 * b_1^-1 * b_2^1) / (1 + -1 * b_1^-1) * T^1 ++ (-1 * b_0^-1 + 1 * b_0^-1 * b_2^1) / (1 + -1 * b_1^-1) * T^0")]]

    #print("Swap rows 0 and 2 = " + str(mat1swap.mat))
    #mat1.swapRows(0, 2)
    #assert(mat1.mat == mat1swap.mat)

    #for row in range(len(testmatrix3_1)):
        #for col in range(len(testmatrix3_1)):
            #print(testmatrix3_1[row][col].reduced(relations3_1))

    matrix3_1 = SFPolyMat(testmatrix3_1, relations3_1)
    matrix3_1.diagonalize()
    print("matrix3_1.delta1() = " + str(matrix3_1.delta1()))

    testmat4_1 = [[SkewFieldPolynomial("(1 + -1 * a_1^1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_1^1 * b_0^-1) / (1) * T^0"),
                   SkewFieldPolynomial("0")],
                  [SkewFieldPolynomial("(1 + -1 * b_1^-1 * c_1^1) / (1) * T^0"),
                   SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(-1 * b_1^-1 * c_1^1) / (1) * T^1 ++ (1 * b_1^-1 * c_1^1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1")],
                  [SkewFieldPolynomial("(-1 * a_2^1 * b_1^1 * c_2^-1 + 1 * b_1^1) / (1) * T^1 ++ (1 + -1 * a_1^-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(1 * b_1^1) / (1) * T^2 ++ (-1 * a_1^-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^1"),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^2")],
                  [SkewFieldPolynomial("(1 + -1 * a_2^-1 * c_2^1) / (1) * T^1 ++ (1 + -1 * a_2^-1 * c_1^-1 * c_2^1) / (1) * T^0"),
                   SkewFieldPolynomial("(-1 * a_2^-1 * c_2^1) / (1) * T^2"),
                   SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * a_2^-1 * c_1^-1 * c_2^1) / (1) * T^1")]]


    #print(testmat4_1)

    rel4_1 = [SkewFieldWord("a_0^1 * b_-1^-1"), SkewFieldWord("b_0^1 * c_0^1 * c_1^-1"), SkewFieldWord("c_0^1 * c_1^-3 * c_2^1")]

    mat4_1 = SFPolyMat(testmat4_1, rel4_1)
    mat4_1.diagonalize()
    print("matrix4_1.delta1() = " + str(matrix4_1.delta1()))

    return 1

    relations6_2 = [SkewFieldWord("a_0^1 * b_-1^-1 * e_-1^1 * e_0^-1"),
                 SkewFieldWord("b_0^1 * c_0^-1 * e_1^1"),
                 SkewFieldWord("c_0^1 * e_-1^1 * e_0^-2 * e_1^1 * e_2^-1"),
                 SkewFieldWord("d_0^1 * e_-1^-1 * e_1^-1"),
                 SkewFieldWord("e_0^1 * e_1^-3 * e_2^3 * e_3^-3 * e_4^1")]

    testmatrix6_2 = [[SkewFieldPolynomial("(1 + -1 * b_2^-1 * d_3^-1) / (1) * T^1 ++ (1 + -1 * b_2^-1* d_2^1 * d_3^-1) / (1) * T^0"),
                      SkewFieldPolynomial("(-1 * b_2^-1 * d_3^-1) / (1) * T^2"),
                      SkewFieldPolynomial.zero(),
                      SkewFieldPolynomial.zero(),
                      SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * b_2^-1 * d_2^1 * d_3^-1) / (1) * T^1"),
                      SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 + -1 * b_1^1 * e_1^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0"),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(-1 * b_1^1 * e_1^-1) / (1) * T^1 ++ (1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^1) / (1) * T^0"),
                      SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_3^1 * c_1^1) / (1) * T^0"),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 * a_4^-1 + -1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^1 ++ (1 + -1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^0"),
                   SkewFieldPolynomial("(1 * a_4^-1) / (1) * T^2 ++ (-1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^1"),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^2"),
                   SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 + -1 * a_4^1 * d_2^-1) / (1) * T^0"),
                    SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1) / (1) * T^1 ++ (1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^-1 * e_1^1) / (1) * T^0"),
                    SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(-1 * c_1^-1 * e_1^1) / (1) * T^1 ++ (1 * c_1^-1 * e_1^1) / (1) * T^0"),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(1) / (1) * T^1")]]

    #print("relations6_2 = " + str(relations6_2))
    #print("testmatrix6_2 = " +str(testmatrix6_2))

    mat6_2 = SFPolyMat(testmatrix6_2, relations6_2)
    mat6_2.diagonalize()
    print(mat6_2.delta1())

    assert(mat6_2.mat == testmatrix6_2)




if __name__ == "__main__":
    sys.exit(main())

