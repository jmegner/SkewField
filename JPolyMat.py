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


    def degRep(self):
        degMat = [[poly.degree() for poly in row] for row in self.mat]
        rep = str(degMat)
        rep = rep.replace("-1", "Z")
        rep = rep.replace("\"", "").replace("\'", "").replace("],", "],\n")
        return rep


    def numLetters(self):
        letterCount = 0

        for rowIdx in self.rowRange():
            for colIdx in self.colRange():
                letterCount += self.mat[rowIdx][colIdx].numLetters()

        return letterCount


    def normalize(self):
        self.mat = [[poly.normalized(self.rels) for poly in row] for row in self.mat]


    def nRows(self):
       return len(self.mat)


    def nCols(self):
       return self.nRows() # remember it is a square matrix


    def rowRange(self):
       return range(self.nRows())


    def colRange(self):
       return range(self.nCols())


    def getRow(self, rowIdx):
        return list(self.mat[rowIdx])


    def setRow(self, rowIdx, otherRow):
        self.mat[rowIdx] = list(otherRow)


    def getCol(self, colIdx):
        return [row[colIdx] for row in self.mat]


    def setCol(self, colIdx, otherCol):
        for rowIdx in self.rowRange():
            self.mat[rowIdx][colIdx] = otherCol[rowIdx]


    def numNonzeroPolys(self, polyList):
        nonzeroCount = 0

        for poly in polyList:
            if not poly.isZero():
                nonzeroCount += 1

        return nonzeroCount


    def swapRows(self, row1, row2):
        swapHolder = self.mat[row1]
        self.mat[row1] = self.mat[row2]
        self.mat[row2] = swapHolder


    def swapCols(self, col1, col2):
        for row in self.rowRange():
            (self.mat[row][col1], self.mat[row][col2]) \
                = (self.mat[row][col2], self.mat[row][col1])


    # note: left multiplies; return normalized(scaler * row)
    def scaledRow(self, rowIdx, scaler):
        return [scaler.times(poly).normalized(self.rels) for poly
            in self.getRow(rowIdx)]


    # note: right multiplies; returns normalized(col * scaler)
    def scaledCol(self, colIdx, scaler):
        return [poly.times(scaler).normalized(self.rels) for poly
            in self.getCol(colIdx)]


    def addToRow(self, rowIdx, otherRow):
        self.setRow(
            rowIdx,
            [poly1.plus(poly2).normalized(self.rels) for (poly1, poly2)
                in zip(self.getRow(rowIdx), otherRow)]
            )


    def addToCol(self, colIdx, otherCol):
        self.setCol(
            colIdx,
            [poly1.plus(poly2).normalized(self.rels) for (poly1, poly2)
                in zip(self.getCol(colIdx), otherCol)]
            )

    def addMultOfRow(self, destRowIdx, srcRowIdx, scaler):
        self.addToRow(destRowIdx, self.scaledRow(srcRowIdx, scaler))


    def addMultOfCol(self, destColIdx, srcColIdx, scaler):
        self.addToCol(destColIdx, self.scaledCol(srcColIdx, scaler))


    # note: left multiplies; row = normalized(multiplier * row)
    def scaleRow(self, row, multiplier):
        self.mat[row] = [multiplier.times(poly) #.normalized(self.rels)
            for poly in self.mat[row]]


    # note: right multiplies; col = normalized(col * multiplier)
    def scaleCol(self, col, multiplier):
        for row in self.rowRange():
            poly = self.mat[row][col].times(multiplier).normalized(self.rels)
            self.mat[row][col] = poly


    def centerTPowersOfRows(self):
        for rowIdx in self.rowRange():
            minTPower = sys.maxint

            for poly in self.getRow(rowIdx):
                if not poly.isZero():
                    minTPower = min(minTPower, poly.lowestPower())

            #if minTPower != sys.maxint and minTPower != 0:
            if minTPower < 0:
                # multiplying this will get row's minimum tpower to be exactly zero
                tpowerScaler = SkewFieldMonomial(
                    SkewFieldSentence.one(),
                    SkewFieldSentence.one(),
                    -minTPower
                    ).asPoly()

                self.scaleRow(rowIdx, tpowerScaler)


    def posOfMinInSubmat(self, startingRowAndCol):
        minDegree = sys.maxint
        posOfMin = (None, None)

        for rowIdx in range(startingRowAndCol, self.nRows()):
            for colIdx in range(startingRowAndCol, self.nCols()):
                poly = self.mat[rowIdx][colIdx]

                if not poly.isZero():
                    if poly.degree() < minDegree:
                        minDegree = poly.degree()
                        posOfMin = (rowIdx, colIdx)

        return posOfMin


    def minToPivotPosition(self, pivotRowAndCol):
        minRowCol = self.posOfMinInSubmat(pivotRowAndCol)
        if not(minRowCol[0] is None):
            self.swapRows(pivotRowAndCol, minRowCol[0])
            self.swapCols(pivotRowAndCol ,minRowCol[1])


    def downgradeRowEntry(self, pivotRowAndCol, targetColIdx):
        pivotPoly = self.mat[pivotRowAndCol][pivotRowAndCol]
        targetPoly = self.mat[pivotRowAndCol][targetColIdx]

        rightQuot = targetPoly.rightQuotient(pivotPoly)
        reducingScaler = rightQuot.normalized(self.rels).aInv()

        self.addMultOfCol(targetColIdx, pivotRowAndCol, reducingScaler)


    def downgradeColEntry(self, pivotRowAndCol, targetRowIdx):
        pivotPoly = self.mat[pivotRowAndCol][pivotRowAndCol]
        targetPoly = self.mat[targetRowIdx][pivotRowAndCol]

        leftQuot = targetPoly.leftQuotient(pivotPoly)
        reducingScaler = leftQuot.normalized(self.rels).aInv()

        self.addMultOfRow(targetRowIdx, pivotRowAndCol, reducingScaler)


    def killRowAndCol(self, pivotRowAndCol, doPrint = False):
        if doPrint:
            print("")
            print("killRowAndCol({})".format(pivotRowAndCol))
            print(self.degRep())

        # kill non-pivot elements in col
        while self.numNonzeroPolys(self.getCol(pivotRowAndCol)) > 1:
            self.minToPivotPosition(pivotRowAndCol)

            if doPrint:
                print("after minToPivotPosition({})".format(pivotRowAndCol))
                print(self.degRep())

            for targetRowIdx in range(pivotRowAndCol + 1, self.nRows()):
                if not self.mat[targetRowIdx][pivotRowAndCol].isZero():
                    self.downgradeColEntry(pivotRowAndCol, targetRowIdx)
                    print("numLetters = {}".format(self.numLetters()))

                    if doPrint:
                        print("after downgradeColEntry(pivotRC={}, targetR={})"
                            .format(pivotRowAndCol, targetRowIdx))
                        print(self.degRep())

                    break

        # kill non-pivot elements in row
        while self.numNonzeroPolys(self.getRow(pivotRowAndCol)) > 1:
            self.minToPivotPosition(pivotRowAndCol)
            if doPrint:
                print("after minToPivotPosition({})".format(pivotRowAndCol))
                print(self.degRep())

            for targetColIdx in range(pivotRowAndCol + 1, self.nCols()):
                if not self.mat[pivotRowAndCol][targetColIdx].isZero():
                    self.downgradeRowEntry(pivotRowAndCol, targetColIdx)
                    print("numLetters = {}".format(self.numLetters()))

                    if doPrint:
                        print("after downgradeRowEntry(pivotRC={}, targetC={})"
                            .format(pivotRowAndCol, targetColIdx))
                        print(self.degRep())

                    break


    #Only applicable for diagonal matrix
    def delta1(self):
        det = 0
        for rowAndCol in self.rowRange():
            det += self.mat[rowAndCol][rowAndCol].powerDiff()
        return det


    def diagonalize(self, doPrint = False):
        self.normalize()
        self.centerTPowersOfRows()

        print("numLetters = {}".format(self.numLetters()))

        for pivotRowAndCol in self.rowRange():
            if doPrint:
                print(self.niceRep())
                print("killRowAndCol({})...".format(pivotRowAndCol))
            self.killRowAndCol(pivotRowAndCol, doPrint)
            print("numLetters = {}".format(self.numLetters()))


################################################################################
# END OF CLASSES
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

    relsABCD = [
        SkewFieldWord("a_-999^1 * a_999^1"),
        SkewFieldWord("b_-999^1 * b_999^1"),
        SkewFieldWord("c_-999^1 * c_999^1"),
        SkewFieldWord("d_-999^1 * d_999^1"),
    ]

    jMat1Orig = JPolyMat(pMat1, relsABCD)

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
    # test addMultOfRow, addMultOfCol

    poly2 = SkewFieldSentence("2").asPoly()
    poly2A = polyA.times(poly2)
    poly2B = polyB.times(poly2)
    poly2C = polyC.times(poly2)
    poly2D = polyD.times(poly2)

    poly2AplusC = poly2A.plus(polyC)
    poly2BplusD = poly2B.plus(polyD)

    poly2AplusB = poly2A.plus(polyB)
    poly2CplusD = poly2C.plus(polyD)

    # reset
    jMat1 = jMat1Orig.copy()

    # add 2 * [A, B] to [C, D]
    jMat1.addMultOfRow(1, 0, poly2)
    assert(jMat1.mat ==
        [[polyA,       polyB],
         [poly2AplusC, poly2BplusD]])

    # reset
    jMat1 = jMat1Orig.copy()

    # add 2 * [A, C] to [B, D]
    jMat1.addMultOfCol(1, 0, poly2)
    assert(jMat1.mat ==
        [[polyA, poly2AplusB],
         [polyC, poly2CplusD]])

    ############################################################################
    # test centerTPowersOfRows

    # for before
    polyAT2 = polyAT.times(polyT)
    polyBT = polyB.times(polyT)
    # polyC is fine
    polyDTN3 = SkewFieldPolynomial("(1 * d_0^1) / (1) * T^-3")

    pMat2 = [
        [ polyAT2, polyBT, ],
        [ polyCT, polyDTN3, ],
    ]

    jMat2Orig = JPolyMat(pMat2, relsABCD)
    jMat2 = jMat2Orig.copy()

    # for after
    polyAT_SN1 = SkewFieldPolynomial("(1 * a_-1^1) / (1) * T^1")
    polyB_SN1 = SkewFieldPolynomial("(1 * b_-1^1) / (1) * T^0")
    polyCT4_S3 = SkewFieldPolynomial("(1 * c_3^1) / (1) * T^4")
    polyD_S3 = SkewFieldPolynomial("(1 * d_3^1) / (1) * T^0")

    jMat2.centerTPowersOfRows();

    ## assertion for if we allow tpowers to come down
    #assert(jMat2.mat ==
    #    [[polyAT_SN1, polyB_SN1],
    #     [polyCT4_S3, polyD_S3]])

    # assertion for if we only let tpowers go up
    assert(jMat2.mat ==
        [[polyAT2, polyBT],
         [polyCT4_S3, polyD_S3]])

    ############################################################################
    # test posOfMinInSubmat

    polyZe = SkewFieldPolynomial.zero()
    polyT0 = SkewFieldPolynomial("(1) / (1) * T^0")
    polyT1 = SkewFieldPolynomial("(1) / (1) * T^1")
    polyT2 = SkewFieldPolynomial("(1) / (1) * T^2")

    pMat3 = [
        [ polyT0, polyT0, polyT0, ],
        [ polyT0, polyT1, polyT0, ],
        [ polyT0, polyZe, polyT2, ],
    ]

    jMat3 = JPolyMat(pMat3, relsABCD)

    posOfMin3 = jMat3.posOfMinInSubmat(1)
    assert(posOfMin3 == (1, 2))

    pMat4 = [
        [ polyT0, polyT0, polyT0, ],
        [ polyT0, polyT1, polyZe, ],
        [ polyT0, polyT0, polyT2, ],
    ]

    jMat4 = JPolyMat(pMat4, relsABCD)

    posOfMin4 = jMat4.posOfMinInSubmat(1)
    assert(posOfMin4 == (2, 1))

    pMat5 = [
        [ polyT0, polyT0, polyT0, ],
        [ polyT0, polyT1, polyZe, ],
        [ polyT0, polyT2, polyT0, ],
    ]

    jMat5 = JPolyMat(pMat5, relsABCD)

    posOfMin5 = jMat5.posOfMinInSubmat(1)
    assert(posOfMin5 == (2, 2))

    pMat6 = [
        [ polyT0, polyT0, polyT0, ],
        [ polyT0, polyZe, polyZe, ],
        [ polyT0, polyZe, polyZe, ],
    ]

    jMat6 = JPolyMat(pMat6, relsABCD)

    posOfMin6 = jMat6.posOfMinInSubmat(1)
    assert(posOfMin6 == (None, None))

    ############################################################################
    # tests over
    print("test battery passed")


def getKnotFromFile(fileName):
    fileRead = open(fileName, "r")
    polyMatSize = 0
    polysGotten = 0

    pMat = []
    relations = []

    for line in fileRead:
        line = line.split("#")[0]
        line = line.strip()

        if len(line) == 0:
            continue

        # if haven't gotten mat size yet
        if polyMatSize == 0:
            polyMatSize = int(line)
            if polyMatSize == 0:
                raise ValueError("bad matrix size")
        else:
            if polysGotten < polyMatSize * polyMatSize:
                if polysGotten % polyMatSize == 0:
                    pMat.append([])

                pMat[-1].append(SkewFieldPolynomial(line))

                polysGotten += 1

            else:
                relations.append(SkewFieldWord(line))

    if polysGotten != polyMatSize * polyMatSize \
            or len(relations) != polyMatSize - 1:
        raise ValueError("file did not match stated size")

    return JPolyMat(pMat, relations)


def main(argv=None):

    testBattery()

    knot3_1 = getKnotFromFile("knot3_1.txt")
    knot4_1 = getKnotFromFile("knot4_1.txt")
    knot5_1 = getKnotFromFile("knot5_1.txt")
    knot5_2 = getKnotFromFile("knot5_2.txt")
    knotSmall6_1 = getKnotFromFile("knotSmall6_1.txt")
    knot6_2 = getKnotFromFile("knot6_2.txt")

    print("knot3_1...")
    knot3_1.diagonalize()
    print("knot3_1.delta1() = " + str(knot3_1.delta1()))
    assert(knot3_1.delta1() == 1)

    print("")
    print("knot4_1...")
    knot4_1.diagonalize()
    print("knot4_1.delta1() = " + str(knot4_1.delta1()))
    assert(knot4_1.delta1() == 1)

    print("")
    print("knot5_1...")
    knot5_1.diagonalize()
    print("knot5_1.delta1() = " + str(knot5_1.delta1()))
    assert(knot5_1.delta1() == 3)

    print("")
    print("knot5_2...")
    knot5_2.diagonalize()
    print("knot5_2.delta1() = " + str(knot5_2.delta1()))
    assert(knot5_2.delta1() == 1)

    #return 1

    print("")
    print("knotSmall6_1...")
    knotSmall6_1.diagonalize()
    print("knotSmall6_1.delta1() = " + str(knotSmall6_1.delta1()))
    assert(knotSmall6_1.delta1() == 1)

    print("")
    print("knot6_2...")
    knot6_2.diagonalize()
    print("knot6_2.delta1() = " + str(knot6_2.delta1()))
    assert(knot6_2.delta1() == 3)

    return 0


if __name__ == "__main__":
    sys.exit(main())

