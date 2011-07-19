#! /usr/bin/env python

FileVersion = "0.000"

import sys
import getopt
import re
import collections

import SkewField
from SkewField import *


class MatPoly():

#-------initializer-------#    
    def __init__(self,mat,rels):
        self.mat = mat
        self.rels = rels

#-------methods to be used with skew field-------#
    
    #finds the degree of a polynomial. Implimented over skew field
    def degree(self,elt):
        return elt.degree()
    
    #Finds lowest degree of power in Laurent polynomial. Implimented over skew field    
    def lowestpower(self,elt):
        return elt.lowestPower()
    
    #swaps rows i and j in the matrix. Implimented over skew field
    def swap_rows(self,i,j):
        temp = self.mat[i]
        self.mat[i] = self.mat[j]
        self.mat[j] = temp
    
    #swaps columns i and j in the matrix. Implimented over skew field
    def swap_columns(self,i, j):
        for k in range(0,len(self.mat)):
            #print "in swap_columns at " + str(k)
            temp = self.mat[k][i]
            self.mat[k][i] = self.mat[k][j]
            self.mat[k][j] = temp
    
    #scales row i by mult. Implimented over skew field.
    def scale_row(self,i,mult):
        for k in range(0,self.ncols()):
            self.mat[i][k] = self.mat[i][k].times(mult)
            self.mat[i][k] = self.mat[i][k].reduced(self.rels)
    
    #scales col i by mult. Implimented over skew field
    def scale_col(self, i, mult):
        for k in range(0,self.nrows()):
            self.mat[k][i] = self.mat[k][i].times(mult)
            self.mat[k][i] = self.mat[k][i].reduced(self.rels)
        
    #adding mult*j to i. Should work over skew field    
    def add_multiple_of_row(self,i,j,mult):
        print "adding " + str(mult) +"*" +str(j) + " to row " +str(i)
        temp = []
        for k in range(0,self.ncols()):
            a = self.mat[j][k].times(mult)
            temp.append(a.reduced(self.rels))
        for l in range(0,self.ncols()):
            self.mat[i][l] = self.mat[i][l].plus(temp[l])
            self.mat[i][l] = self.mat[i][l].reduced(self.rels)
        print self.mat
    
    #adding mult*j to i. Should work over skew field    
    def add_multiple_of_col(self,i,j,mult):
        print "adding " + str(mult) +"*" + str(j) + " to col " + str(i)
        temp = []
        for k in range(0,self.nrows()):
            a = self.mat[k][j].times(mult)
            temp.append(a.reduced(self.rels))
        for l in range(0,self.ncols()):
            self.mat[l][i] = self.mat[l][i].plus(temp[l])
            self.mat[l][i] = self.mat[l][i].reduced(self.rels)
        print self.mat          
                    
    #number of rows in the matrix. OK over skew field
    def nrows(self):
       return len(self.mat)
    
    #number of columns in the matrix. OK over skew field
    def ncols(self):
       return self.nrows()
    
    #will return the quotient to multiply row by. OK over skew field
    def div(self,poly1,poly2):
        q = poly1.quotient(poly2)
        return q.reduced(self.rels)
        
    def notzero(self,poly):
        return not poly.isZero()

#-------methods to diagonalize-------#

    def clear_row(self,i):
        mindeg = 0
        for j in range(0,self.ncols()):
            #print ("in clear_row " + str(i) + ", " + str(j) + " ok")
            if self.notzero(self.mat[i][j]):
                #print "in row " + str(i) + ", " + str(j) + " is not zero"
                if self.lowestpower(self.mat[i][j]) < mindeg:
                    mindeg = self.lowestpower(self.mat[i][j])
        #print mindeg
        pow = SkewFieldPolynomial([SkewFieldMonomial(SkewFieldSentence.one(),SkewFieldSentence.one(),-1*mindeg)])
        #print pow
        self.scale_row(i,pow)
    
    def kill_negatives(self):
        for i in range(0,self.nrows()):
            #print (str(i) + " ok")
            self.clear_row(i)
        print "done killing negatives"
            
   #finds position of minimum degree elt of matrix starting at (i,i)    
    def minpos(self,i):
        mindeg = -2
        minpos = (i, i)
        for j in range(i,self.nrows()):
            for k in range(i, self.ncols()):
                #print self.degree(self.mat[j][k])
                if self.degree(self.mat[j][k]) > -1:
                    if mindeg == -2 or self.degree(self.mat[j][k]) < mindeg:
                        mindeg = self.degree(self.mat[j][k])
                        minpos = (j, k)
                        #print minpos
        return minpos
        
    def mintotop(self,i):
        min = self.minpos(i)
        #print str(min)
        self.swap_rows(i,min[0])
        #print "swapped rows"
        self.swap_columns(i,min[1])
        #print "swapped columns"
    
    def killcolentry(self,i,j):
        q = self.div(self.mat[j][i],self.mat[i][i])
        print "result of dividing col entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.add_multiple_of_row(j,i,q.aInv().reduced(self.rels))
    
    def killrowentry(self,i,j):
        q = self.div(self.mat[i][j],self.mat[i][i])
        print "result of dividing row entry " + str(i) + " by " + str(j) + " is " + str(q)
        self.add_multiple_of_col(j,i,q.aInv().reduced(self.rels))
        
    def killrowcol(self,i):
        self.mintotop(i)
        print "mintotop successful"
        for j in range(i+1,self.nrows()):
            if self.notzero(self.mat[j][i]):
                self.killcolentry(i,j)
                self.killrowcol(i)
        for k in range(i+1,self.ncols()):
            if self.notzero(self.mat[i][k]):
                self.killrowentry(i,k)
                self.killrowcol(i)
        print self.mat
        
    def diagonalize(self):
        self.kill_negatives()
        for i in range(0,self.nrows()):
            print "in at " + str(i)
            self.killrowcol(i)
            
    #Only applicable for diagonal matrix
    def delta1(self):
        det = 0
        for row in range(self.ncol()):
            det += matrix[row][row].tpowerDiff()
        return det


################################################################################
# MAIN
################################################################################

def main(argv=None):


    relations = [SkewFieldWord("a_0^1 * b_-1^-1 * e_-1^1 * e_0^-1"),
                 SkewFieldWord("b_0^1 * c_0^-1 * e_1^1"),
                 SkewFieldWord("c_0^1 * e_-1^1 * e_0^-2 * e_1^1 * e_2^-1"),
                 SkewFieldWord("d_0^1 * e_-1^-1 * e_1^-1"),
                 SkewFieldWord("e_0^1 * e_1^-3 * e_2^3 * e_3^-3 * e_4^1")]

    testmatrix = [[SkewFieldPolynomial("(1 + -1 * b_2^-1 * d_3^-1) / (1) * T^1 ++ (1 + -1 * b_2^-1* d_2^1 * d_3^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(-1 * b_2^-1 * d_3^-1) / (1) * T^2"),
                   SkewFieldPolynomial("0"), SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(1) / (1) * T^2 ++ (-1 * b_2^-1 * d_2^1 * d_3^-1) / (1) * T^1"),
                   SkewFieldPolynomial("0")],
                  [SkewFieldPolynomial("(1 + -1 * b_1^1 * e_1^-1) / (1) * T^0"), SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0"), SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("0"), SkewFieldPolynomial("(-1 * b_1^1 * e_1^-1) / (1) * T^1 ++ (1 * b_1^1 * c_0^-1 * e_1^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^1) / (1) * T^0"), SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"), SkewFieldPolynomial("(-1 * a_3^1 * c_1^1) / (1) * T^0"),
                   SkewFieldPolynomial("0"), SkewFieldPolynomial("0")],
                  [SkewFieldPolynomial("(1 * a_4^-1 + -1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^1 ++ (1 + -1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^0"),
                   SkewFieldPolynomial("(1 * a_4^-1) / (1) * T^2 ++ (-1 * a_4^-1 * b_1^-1 * b_2^1 * d_3^1) / (1) * T^1"),
                   SkewFieldPolynomial("0"),SkewFieldPolynomial("(1) / (1) * T^1"),
                   SkewFieldPolynomial("(-1 * a_4^-1 * b_2^1 * d_3^1) / (1) * T^2"), SkewFieldPolynomial("0")],
                  [SkewFieldPolynomial("(1 + -1 * a_4^1 * d_2^-1) / (1) * T^0"), SkewFieldPolynomial("0"), SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1) / (1) * T^1 ++ (1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0"),
                   SkewFieldPolynomial("(1) / (1) * T^1"), SkewFieldPolynomial("(-1 * a_4^1 * d_2^-1 * e_0^-1) / (1) * T^0")],
                  [SkewFieldPolynomial("(1 + -1 * c_1^-1 * e_1^1) / (1) * T^0"), SkewFieldPolynomial("0"),
                   SkewFieldPolynomial("(-1 * c_1^-1 * e_1^1) / (1) * T^1 ++ (1 * c_1^-1 * e_1^1) / (1) * T^0"),
                   SkewFieldPolynomial("0"), SkewFieldPolynomial("0"), SkewFieldPolynomial("(1) / (1) * T^1")]]



if __name__ == "__main__":
    sys.exit(main())

