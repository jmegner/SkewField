#! /usr/bin/env python

FileVersion = "0.1"

import sys
import getopt
import re
import collections

import SkewField
from SkewField import *

class SFPolyMat():

#-------initializer-------#    
    def __init__(self, mat, rels):
        self.mat = mat
        self.rels = rels

#-------methods to be used with skew field-------#
    
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
            nPoly = self.mat[row][col].times(multiplier)
            self.mat[row][col] = nPoly.reduced(self.rels)
    
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
            poly = self.mat[j][col].times(multiplier)
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
    def quotient(self,poly1,poly2):
        q = poly1.quotient(poly2)
        return q.reduced(self.rels)
        

#-------methods to diagonalize-------#

    def clearRow(self, row):
        minDegree = 0
        for col in range(self.nCols()):
            #print ("in clearRow " + str(row) + ", " + str(col) + " ok")
            if not self.mat[row][col].isZero():
                #print "in row " + str(row) + ", " + str(col) + " is not zero"
                if self.mat[row][col].lowestPower < minDegree:
                    minDegree = self.mat[row][col].lowestPower
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
        min = self.minPosition(row)
        #print str(min)
        self.swapRows(row, min[0])
        #print "swapped rows"
        self.swapColumns(row ,min[1])
        #print "swapped columns"
    
    def killColEntry(self, i, j):
        q = self.quotient(self.mat[j][i], self.mat[i][i])
        #print "result of dividing col entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfRow(j, i, q.aInv().reduced(self.rels))
    
    def killRowEntry(self, i, j):
        q = self.quotient(self.mat[i][j],self.mat[i][i])
        #print "result of dividing row entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.addMultOfColumn(j, i, q.aInv().reduced(self.rels))
        
    def killRowCol(self,i):
        self.minToTop(i)
        #print "minToTop successful"
        for row in range(i+1, self.nRows()):
            if not self.mat[row][i].isZero():
                self.killColEntry(i, row)
                self.killRowCol(i)
        for col in range(i+1, self.nCols()):
            if not self.mat[i][col].isZero():
                self.killRowEntry(i, col)
                self.killRowCol(i)
        #print self.mat
        
    def diagonalize(self):
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


def main(argv=None):


    relations6_2 = [SkewFieldWord("a_0^1 * b_-1^-1 * e_-1^1 * e_0^-1"),
                 SkewFieldWord("b_0^1 * c_0^-1 * e_1^1"),
                 SkewFieldWord("c_0^1 * e_-1^1 * e_0^-2 * e_1^1 * e_2^-1"),
                 SkewFieldWord("d_0^1 * e_-1^-1 * e_1^-1"),
                 SkewFieldWord("e_0^1 * e_1^-3 * e_2^3 * e_3^-3 * e_4^1")]

    testmatrix6_2 = [[SkewFieldPolynomial("(1 + -1 * b_2^-1 * d_3^-1) / (1) * T^1 ++ (1 + -1 * b_2^-1* d_2^1 * d_3^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(-1 * b_2^-1 * d_3^-1) / (1) * T^2"),
                   SkewFieldPolynomial.zero(), SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * b_2^-1 * d_2^1 * d_3^-1) / (1) * T^1"),
                   SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 + -1 * b_1^1 * e_1^-1) / (1) * T^0"), SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0"), SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial.zero(), SkewFieldPolynomial("(-1 * b_1^1 * e_1^-1) / (1) * T^1 ++ (1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^1) / (1) * T^0"), SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(1) / (1) * T^1"), SkewFieldPolynomial("(-1 * a_3^1 * c_1^1) / (1) * T^0"),
                   SkewFieldPolynomial.zero(), SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 * a_4^-1 + -1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^1 ++ (1 + -1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^0"),
                   SkewFieldPolynomial("(1 * a_4^-1) / (1) * T^2 ++ (-1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^1"),
                   SkewFieldPolynomial.zero(),SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^2"), SkewFieldPolynomial.zero()],
                  [SkewFieldPolynomial("(1 + -1 * a_4^1 * d_2^-1) / (1) * T^0"), SkewFieldPolynomial.zero(), SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1) / (1) * T^1 ++ (1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"), SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^-1 * e_1^1) / (1) * T^0"), SkewFieldPolynomial.zero(),
                   SkewFieldPolynomial("(-1 * c_1^-1 * e_1^1) / (1) * T^1 ++ (1 * c_1^-1 * e_1^1) / (1) * T^0"),
                   SkewFieldPolynomial.zero(), SkewFieldPolynomial.zero(), SkewFieldPolynomial("(1) / (1) * T^1")]]

    print("relations6_2 = " + str(relations6_2))
    print("testmatrix6_2 = " +str(testmatrix6_2))

    mat6_2 = SFPolyMat(testmatrix6_2, relations6_2)

    assert(mat6_2.mat == testmatrix6_2)



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

    rel1 = [SkewFieldWord("a_0^1 * a_5^-1 * c_1^1"),
            SkewFieldWord("b_0^1 * b_5^-1 * c_2^1"), 
            SkewFieldWord("c_0^2 * c_1^-1 * c_7^-1")]

    mat1 = [[SkewFieldPolynomial("(1 * a_0^1) / (1) * T^1"),
             SkewFieldPolynomial("(1 * b_0^1) / (1) * T^1"),
             SkewFieldPolynomial("(1 * c_0^1) / (1) * T^1")], 
            [SkewFieldPolynomial("(1 * a_0^2) / (1) * T^2"),
             SkewFieldPolynomial("(1 * b_0^2) / (1) * T^2"),
             SkewFieldPolynomial("(1 * c_0^2) / (1) * T^2")], 
            [SkewFieldPolynomial("(1 * a_0^3) / (1) * T^3"),
             SkewFieldPolynomial("(1 * b_0^3) / (1) * T^3"),
             SkewFieldPolynomial("(1 * c_0^3) / (1) * T^3")]]

    mat1 = SFPolyMat(mat1, rel1)

    #print("Swap rows 0 and 2 = " + str(mat1swap.mat))
    #mat1.swapRows(0, 2)
    #assert(mat1.mat == mat1swap.mat)

    #matrix3_1 = SFPolyMat(testmatrix3_1, relations3_1)
    #matrix3_1.diagonalize()
    #print(matrix3_1.delta1())


if __name__ == "__main__":
    sys.exit(main())

