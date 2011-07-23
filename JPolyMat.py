#! /usr/bin/env python

FileVersion = "0.1"

import sys
import getopt
import re
import collections

from SkewField import *

# note: rows use left-multiplications and left-quotients
# cols use right-multiplications and right-quotients


class JPolyMat():

    def __init__(self, mat, rels):
        self.mat = [[poly for poly in row] for row in mat]
        self.rels = list(rels)

        if len(self.mat) != len(self.mat[0]):
            raise ValueError("must be square matrix")


    def __str__(self):
        return str([[str(poly) for poly in rowArray] for rowArray in self.mat])


    def niceRep(self):
        rep = str([[str(elem) for elem in row]  for row in self.mat])
        rep = rep.replace("\"", "").replace("\'", "").replace("],", "],\n")
        return rep


    def copy(self):
        return JPolyMat(self.mat, self.rels)


    def reduce(self):
        self.mat = [[poly.reduced(self.rels) for poly in row] for row in self.mat]


    def getRow(self, rowIdx):
        return list(self.mat[rowIdx])


    def setRow(self, rowIdx, otherRow):
        self.mat[rowIdx] = list(otherRow)


    def getCol(self, colIdx):
        return [row[colIdx] for row in self.mat]


    def setCol(self, colIdx, otherCol):
        for rowIdx in self.rowRange():
            self.mat[rowIdx][colIdx] = otherCol[rowIdx]


    def swapRows(self, row1, row2):
        swapHolder = self.mat[row1]
        self.mat[row1] = self.mat[row2]
        self.mat[row2] = swapHolder


    def swapCols(self, col1, col2):
        for row in self.rowRange():
            (self.mat[row][col1], self.mat[row][col2]) \
                = (self.mat[row][col2], self.mat[row][col1])


    # note: left multiplies; return reduced(scaler * row)
    def scaledRow(self, rowIdx, scaler):
        return [scaler.times(poly).reduced(self.rels) for poly
            in self.getRow(rowIdx)]


    # note: right multiplies; returns reduced(col * scaler)
    def scaledCol(self, colIdx, scaler):
        return [poly.times(scaler).reduced(self.rels) for poly
            in self.getCol(colIdx)]


    def addToRow(self, rowIdx, otherRow):
        self.setRow(
            rowIdx,
            [poly1.plus(poly2).reduced(self.rels) for (poly1, poly2)
                in zip(self.getRow(rowIdx), otherRow)]
            )


    def addToCol(self, colIdx, otherCol):
        self.setCol(
            colIdx,
            [poly1.plus(poly2).reduced(self.rels) for (poly1, poly2)
                in zip(self.getCol(colIdx), otherCol)]
            )

    def addMultOfRow(self, destRowIdx, srcRowIdx, scaler):
        self.addToRow(destRowIdx, self.scaledRow(srcRowIdx, scaler))


    def addMultOfCol(self, destColIdx, srcColIdx, scaler):
        self.addToCol(destColIdx, self.scaledCol(srcColIdx, scaler))


################################################################################
## THE DIVIDE; Jacob approved stuff above
################################################################################

    # note: left multiplies; row = reduced(multiplier * row)
    def scaleRow(self, row, multiplier):
        self.mat[row] = [multiplier.times(poly).reduced(self.rels)
            for poly in self.mat[row]]


    # note: right multiplies; col = reduced(col * multiplier)
    def scaleCol(self, col, multiplier):
        for row in self.rowRange():
            poly = self.mat[row][col].times(multiplier).reduced(self.rels)
            self.mat[row][col] = poly


    #adds multiplier*j to i. Should work over skew field
    def addMultOfRow(self, destRow, srcRow, multiplier):
        temp = []
        for col in range(self.nCols()):
            poly = multiplier.times(self.mat[srcRow][col])
            poly = poly.reduced(self.rels)
            temp.append(poly)
        for col in range(self.nCols()):
            poly = self.mat[destRow][col].plus(temp[col])
            self.mat[destRow][col] = poly.reduced(self.rels)
        #print self.mat


    #adding mult*j to i. Should work over skew field
    def addMultOfCol(self, i, j, multiplier):
        #print "adding " + str(multiplier) +"*" + str(j) + " to col " + str(i)
        temp = []
        for row in range(self.nRows()):
            poly = self.mat[row][j].times(multiplier)
            temp.append(poly.reduced(self.rels))
        for row in range(self.nCols()):
            poly = self.mat[row][i].plus(temp[row])
            self.mat[row][i] = poly.reduced(self.rels)
        #print self.mat


    def nRows(self):
       return len(self.mat)


    def nCols(self):
       return self.nRows() # remember it is a square matrix


    def rowRange(self):
       return range(self.nRows())


    def colRange(self):
       return range(self.nCols())


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
        #print "done killing negatives"


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
        self.swapCols(row ,minRowCol[1])
        #print "swapped cols"


    def killColEntry(self, i, j):
        q = self.mat[j][i].leftQuotient(self.mat[i][i])
        q = q.reduced(self.rels)
        q = q.aInv()
        #print "result of dividing col entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfRow(j, i, q)


    def killRowEntry(self, i, j):
        q = self.mat[i][j].rightQuotient(self.mat[i][i])
        q = q.reduced(self.rels)
        q = q.aInv()
        #print "result of dividing row entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfCol(j, i, q)


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
        #print(self.mat)
        self.killNegatives()
        #print(self.mat)
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


def testBattery():
    polyA = SkewFieldLetter("a_0").asPoly()
    polyB = SkewFieldLetter("b_0").asPoly()
    polyC = SkewFieldLetter("c_0").asPoly()
    polyD = SkewFieldLetter("d_0").asPoly()
    polyT = SkewFieldPolynomial("(1) / (1) * T^1")

    pMat1 = [
        [ polyA, polyB, ],
        [ polyC, polyD, ],
    ]

    rels1 = [
        SkewFieldWord("a_-999^1 * a_999^1"),
        SkewFieldWord("b_-999^1 * b_999^1"),
        SkewFieldWord("c_-999^1 * c_999^1"),
        SkewFieldWord("d_-999^1 * d_999^1"),
    ]

    jMat1Orig = JPolyMat(pMat1, rels1)

    jMat1 = jMat1Orig.copy()

    ############################################################################
    # test getRow, getCol, setRow, setCol

    row0 = jMat1.getRow(0)
    assert(row0 == [polyA, polyB])

    row1 = jMat1.getRow(1)
    assert(row1 == [polyC, polyD])

    jMat1.setRow(0, [polyB, polyA])
    assert(jMat1.mat ==
        [[polyB, polyA],
         [polyC, polyD]])

    # reset
    jMat1 = jMat1Orig.copy()

    jMat1.setCol(0, [polyC, polyA])
    assert(jMat1.mat ==
        [[polyC, polyB],
         [polyA, polyD]])

    ############################################################################
    # test swapRows, swapCols

    # reset
    jMat1 = jMat1Orig.copy()

    # do-nothing swap
    jMat1.swapRows(0, 0)
    assert(jMat1.mat == pMat1)

    # do-something swap
    jMat1.swapRows(0, 1)
    assert(jMat1.mat ==
        [[polyC, polyD],
         [polyA, polyB]])

    # and swap them back
    jMat1.swapRows(0, 1)
    assert(jMat1.mat == pMat1)

    # do-nothing swap
    jMat1.swapCols(0, 0)
    assert(jMat1.mat == pMat1)

    # do-something swap
    jMat1.swapCols(0, 1)
    assert(jMat1.mat ==
        [[polyB, polyA],
         [polyD, polyC]])

    # and swap them back
    jMat1.swapCols(0, 1)
    assert(jMat1.mat == pMat1)

    ############################################################################
    # test scaledRow, scaledCol

    # polyT on left for rows
    polyTA = polyT.times(polyA)
    polyTB = polyT.times(polyB)

    # polyT on right for cols
    polyAT = polyA.times(polyT)
    polyCT = polyC.times(polyT)

    row0ScaledByT = jMat1.scaledRow(0, polyT)
    assert(row0ScaledByT == [polyTA, polyTB])

    col0ScaledByT = jMat1.scaledCol(0, polyT)
    assert(col0ScaledByT == [polyAT, polyCT])

    assert(jMat1.mat == pMat1)

    ############################################################################
    # test addToRow, addToCol

    polyAplusC = polyA.plus(polyC)
    polyBplusD = polyB.plus(polyD)

    polyAplusB = polyA.plus(polyB)
    polyCplusD = polyC.plus(polyD)

    jMat1.addToRow(0, jMat1.getRow(1))
    assert(jMat1.mat ==
        [[polyAplusC, polyBplusD],
         [polyC,      polyD]])

    # reset
    jMat1 = jMat1Orig.copy()

    jMat1.addToCol(0, jMat1.getCol(1))
    assert(jMat1.mat ==
        [[polyAplusB, polyB],
         [polyCplusD, polyD]])

    ############################################################################
    # TODO test addMultOfRow, addMultOfCol



def main(argv=None):

    testBattery()

    tMat3_1 = [
        [
            SkewFieldPolynomial("(1 + -1 * b_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_0^-1 * b_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * b_0^1 * b_1^-1 + 1 * b_1^-1) / (1 + -1 * b_1^-1) * T^1 ++ (-1 * b_0^-1 + 1 * b_0^-1 * b_2^-1) / (1 + -1 * b_1^-1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^-1 * b_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_1^-1 * b_1^1) / (1) * T^1 ++ (1 * a_1^-1 * b_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(1 + -1 * b_1^-1 * b_2^1) / (1 + -1 * b_1^-1) * T^1 ++ (-1 * b_0^-1 + 1 * b_0^-1 * b_2^1) / (1 + -1 * b_1^-1) * T^0"),
        ],
    ]

    rels3_1 = [
        SkewFieldWord("a_0^1 * b_0^-1 * b_1^1"),
        SkewFieldWord("b_0^1 * b_1^-1 * b_2^1"),
    ]

    #print("Swap rows 0 and 2 = " + str(mat1swap.mat))
    #mat1.swapRows(0, 2)
    #assert(mat1.mat == mat1swap.mat)

    #for row in range(len(tMat3_1)):
        #for col in range(len(tMat3_1)):
            #print(tMat3_1[row][col].reduced(rels3_1))

    mat3_1 = JPolyMat(tMat3_1, rels3_1)
    mat3_1.diagonalize()
    print("mat3_1.delta1() = " + str(mat3_1.delta1()))

    assert(mat3_1.delta1() == 1)

    tMat4_1 = [
        [
            SkewFieldPolynomial("(1 + -1 * a_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_1^1 * b_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * b_1^-1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * b_1^-1 * c_1^1) / (1) * T^1 ++ (1 * b_1^-1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
        ], [
            SkewFieldPolynomial("(-1 * a_2^1 * b_1^1 * c_2^-1 + 1 * b_1^1) / (1) * T^1 ++ (1 + -1 * a_1^-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^0"),
            SkewFieldPolynomial("(1 * b_1^1) / (1) * T^2 ++ (-1 * a_1^-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^1"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_2^1 * b_1^1 * c_2^-1) / (1) * T^2"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_2^-1 * c_2^1) / (1) * T^1 ++ (1 + -1 * a_2^-1 * c_1^-1 * c_2^1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_2^-1 * c_2^1) / (1) * T^2"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * a_2^-1 * c_1^-1 * c_2^1) / (1) * T^1"),
        ],
    ]


    #print(tMat4_1)

    rels4_1 = [
        SkewFieldWord("a_0^1 * b_-1^-1"),
        SkewFieldWord("b_0^1 * c_0^1 * c_1^-1"),
        SkewFieldWord("c_0^1 * c_1^-3 * c_2^1"),
    ]

    mat4_1 = JPolyMat(tMat4_1, rels4_1)
    mat4_1.diagonalize()
    print("mat4_1.delta1() = " + str(mat4_1.delta1()))

    assert(mat4_1.delta1() == 1)

    tMat5_1 = [
        [
            SkewFieldPolynomial("(1 + -1 * c_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_0^-1 * c_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * c_1^-1) / (1) * T^1 ++ (1 * a_0^-1 * c_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^1 * d_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_1^1 * b_0^-1 * d_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * a_1^1 * d_1^-1) / (1) * T^1 ++ (1 * a_1^1 * b_0^-1 * d_1^-1) / (1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * b_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^-1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_1^-1 * c_1^1) / (1) * T^1 ++ (1 * a_1^-1 * c_1^1 * d_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_1^-1 * c_1^1 * d_0^-1) / (1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * b_1^-1 * d_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * b_1^-1 * d_1^1) / (1) * T^1 ++ (1 * b_1^-1 * d_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
        ],
    ]


    rels5_1 = [
        SkewFieldWord("a_0^1 * b_0^-1 * c_1^1 * d_1^-1"),
        SkewFieldWord("b_0^1 * c_0^-1 * d_1^1"),
        SkewFieldWord("c_0^1 * d_-2^-1 * d_0^-1"),
        SkewFieldWord("d_0^1 * d_1^-1 * d_2^1 * d_3^-1 * d_4^1"),
    ]

    mat5_1 = JPolyMat(tMat5_1, rels5_1)
    mat5_1.diagonalize()
    print("mat5_1.delta1() = " + str(mat5_1.delta1()))

    assert(mat5_1.delta1() == 3)

    return 1

    tMat5_2 = [
        [
            SkewFieldPolynomial("(1 + -1 * b_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_0^-1 * b_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * b_1^-1) / (1) * T^1 ++ (1 * a_0^-1 * b_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"), SkewFieldPolynomial("0")
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^1 * d_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_1^1 * b_0^-1 * d_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * a_1^1 * d_1^-1) / (1) * T^1 ++ (1 * a_1^1 * b_0^-1 * d_1^-1) / (1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * b_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_1^-1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * a_1^-1 * c_1^1) / (1) * T^1 ++ (1 * a_1^-1 * c_1^1 * d_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_1^-1 * c_1^1 * d_0^-1) / (1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * c_1^-1 * d_1^1) / (1) * T^0"),
            SkewFieldPolynomial("0"), SkewFieldPolynomial("0"),
            SkewFieldPolynomial("(-1 * c_1^-1 * d_1^1) / (1) * T^1 ++ (1 * c_1^-1 * d_1^1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
        ],
    ]

    rel5_2 = [
        SkewFieldWord("a_0^1 * b_-1^-1 * d_-1^1 * d_0^-1"),
        SkewFieldWord("b_0^1 * c_-1^-1"),
        SkewFieldWord("c_0^1 * d_1^-1 * d_2^2"),
        SkewFieldWord("d_0^2 * d_1^-3 * d_2^2"),
    ]

    mat5_2 = JPolyMat(tMat5_2, rel5_2)
    mat5_2.diagonalize()
    print("mat5_2.delta1() = " + str(mat5_2.delta1()))

    assert(mat5_2.delta1() == 1)

    return 1 # program does not work beyond this point

    #This is smaller matrix for 6_1 -- only 2x2, but entries are
    #more complicated. SkewFied does not like that
    tMatSmall6_1 = [
        [
            SkewFieldPolynomial("(1 * a_0^1 * a_1^-2 + -1 * a_0^1 * a_1^-2 * a_2^2) / (1) * T^1 ++ (1 * a_0^1 + -1 * a_0^1 * a_1^-5 * a_2^2) / (1) * T^0"),
            SkewFieldPolynomial("(1 * a_0^1 * a_1^-2 + 1 * a_0^1 * a_1^-2 * a_2^1) / (1) * T^2 ++ (-1 * a_0^1 * a_1^-5 * a_2^2 + -1 * a_0^1 * a_1^-4 * a_2^2 + -1 * a_0^1 * a_1^-3 * a_2^2 + -1 * a_0^1 * a_1^-2 + -1 * a_0^1 * a_1^-1) / (1) * T^1 ++ (1 + 1 * a_0^1 * a_1^-5 * a_2^2) / (1) * T^0"),
        ], [
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
        ],
    ]

    relsSmall6_1 = [
        SkewFieldWord("a_0^2 * a_1^-5 * a_2^2")
    ]

    matSmall6_1 = JPolyMat(tMatSmall6_1,relsSmall6_1)
    matSmall6_1.diagonalize()
    print("matSmall6_1.delta1 = " + str(matSmall6_1.delta1()))
    assert(matSmall6_1.delta1() == 1)


    tMat6_2 = [
        [
            SkewFieldPolynomial("(1 + -1 * b_2^-1 * d_3^-1) / (1) * T^1 ++ (1 + -1 * b_2^-1* d_2^1 * d_3^-1) / (1) * T^0"),
            SkewFieldPolynomial("(-1 * b_2^-1 * d_3^-1) / (1) * T^2"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * b_2^-1 * d_2^1 * d_3^-1) / (1) * T^1"),
            SkewFieldPolynomial.zero(),
        ], [
            SkewFieldPolynomial("(1 + -1 * b_1^1 * e_1^-1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(-1 * b_1^1 * e_1^-1) / (1) * T^1 ++ (1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0")
        ], [
            SkewFieldPolynomial("(1 + -1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_3^1 * c_1^1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
        ], [
            SkewFieldPolynomial("(1 * a_4^-1 + -1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^1 ++ (1 + -1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^0"),
            SkewFieldPolynomial("(1 * a_4^-1) / (1) * T^2 ++ (-1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^1"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^2"),
            SkewFieldPolynomial.zero(),
        ], [
            SkewFieldPolynomial("(1 + -1 * a_4^1 * d_2^-1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1) / (1) * T^1 ++ (1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0"),
            SkewFieldPolynomial("(1) / (1) * T^1"),
            SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0"),
        ], [
            SkewFieldPolynomial("(1 + -1 * c_1^-1 * e_1^1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(-1 * c_1^-1 * e_1^1) / (1) * T^1 ++ (1 * c_1^-1 * e_1^1) / (1) * T^0"),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial.zero(),
            SkewFieldPolynomial("(1) / (1) * T^1"),
        ],
    ]

    rels6_2 = [SkewFieldWord("a_0^1 * b_-1^-1 * e_-1^1 * e_0^-1"),
                 SkewFieldWord("b_0^1 * c_0^-1 * e_1^1"),
                 SkewFieldWord("c_0^1 * e_-1^1 * e_0^-2 * e_1^1 * e_2^-1"),
                 SkewFieldWord("d_0^1 * e_-1^-1 * e_1^-1"),
                 SkewFieldWord("e_0^1 * e_1^-3 * e_2^3 * e_3^-3 * e_4^1")]

    mat6_2 = JPolyMat(tMat6_2, rels6_2)
    mat6_2.diagonalize()
    print(mat6_2.delta1())

    assert(mat6_2.delta1() == 3)



if __name__ == "__main__":
    sys.exit(main())

