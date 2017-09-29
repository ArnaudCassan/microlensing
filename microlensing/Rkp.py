# -*- coding: utf-8 -*-

# ====================================================================
#   Methods of quadrupole() and hexadecapole() from publication:
#     Cassan, A. Fast computation of quadrupole and hexadecapole
#                approximations in microlensing with a single
#                point-source evaluation.
#     Mon. Not. R. Astron. 468, 3993 (2017)
# ====================================================================

import numpy as np
from copy import copy

def Q(p):
    '''Compute and display the Q_(p-n,n).
Usage: Q(p).'''
    M = p
    Q = np.zeros((M+1,M+2),dtype='S1000')
    Q[2,3] = '1 2'
    for m in range(3,M+1):
        tmp = str(m)+' '
        for k in range(1,m):
            tmp = tmp + str(k)
        Q[m,3] = Qpartial(m,Q[m-1,3])+','+tmp
        for n in range(4,m+1):
            Q[m,n] = Qpartial(m,Q[m-1,n])+','+Qproduct(m,Q[m-1,n-1])
        Q[m,m+1] = Qproduct(m,Q[m-1,m])
    for m in range(3,M+1):
        print '\033[35m Order p='+str(m)+'\033[30m'
        seq = 'b'*m
        lseq = [seq]
        for p in range(0,m):
            seq = seq[:p]+'a'+seq[p+1:]
            lseq.insert(0,seq)
        for p in range(0,m+1):
            curseq = lseq.pop(0)
            S = copy(Q)
            print '\n\033[36m   Q['+str(m-p)+','+str(p)+'] :\033[30m'
            for n in range(3,m+2):
                for k in range(0,m):
                    S[m,n] = S[m,n].replace(str(k+1),curseq[k])
                print '\n\033[34m     W'+str(n)+' * \033[30m(',
                C = S[m,n].split(',')
                for i in range(0,len(C)):
                    D =  C[i].split(' ')
                    for j in range(0,len(D)):
                        print 'a'+str(D[j].count('a'))+str(D[j].count('b')),
                        if (j < len(D)-1): print '*',
                        na = D[j].count('a')
                        nb = D[j].count('b')
                    if (i < len(C)-1): print '+',
                    else: print ')',
                if (n > m): print '\n'
    return

def Qpartial(m,Q):
    C = Q.split(',')
    F = C
    for l in range(0,len(C)):
        A = C[l].split(' ')
        AA = copy(A)
        for k in range(len(A)):
            G = copy(A)
            G.pop(k)
            AA[k] = str(m)+A[k]+' '+' '.join(G)
        F[l] = ','.join(AA)
    return ','.join(F)

def Qproduct(m,Q):
    C = Q.split(',')
    CC = copy(C)
    for l in range(0,len(C)):
        CC[l] = str(m)+' '+C[l]
    return ','.join(CC)



