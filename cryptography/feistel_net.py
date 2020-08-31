# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:35:00 2019

@author: Maksim
"""
#start_alphab = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz .,!:;?-'
inf_for_crypt ='Secret information: Feistel net is working'# input()

if len(inf_for_crypt) % 2 != 0:
    inf_for_crypt += '|'
my_rounds = 10 # input() количество раундов
half = int(len(inf_for_crypt)/2)
L0 =inf_for_crypt[:half]
R0 = inf_for_crypt[half:]

def Integer(l,r):
    l0, r0 = [], []
    for i in range(len(l)):
        l0.append(ord(l[i]))
        r0.append(ord(r[i]))
    return l0, r0
def Symbolic(a,b):
    res1, res2 = '', ''
    for i in range(len(a)):
        res1 += chr(a[i])
        res2 += chr(b[i]) 
    #print('full decryption: ', res1, res2, sep='')
    return res1+res2

def F(L, num_round):
    return L ^ num_round

def Feistel_Encrypt(L, R, rounds):
    for i in range(rounds):
        if i < rounds - 1:
            for k in range(len(L)):
                L[k], R[k] = (R[k] ^ F(L[k],i)) % 256, L[k]
        else:
            for k in range(len(L)):
                R[k] = (R[k] ^ F(L[k],i)) % 256
    return L, R
 
  
def Feistel_Decrypt(L,R,rounds):
    while rounds > 0:
        if rounds != 1:
            for k in range(len(L)):
                L[k], R[k] = (R[k] ^ F(L[k],rounds-1)) % 256, L[k]
        else:
            for k in range(len(L)):
                R[k] = (R[k] ^ F(L[k],rounds-1)) % 256
        rounds -= 1
    return L, R


left, right = Integer(L0, R0)
Ln,Rn = Feistel_Encrypt(left, right, my_rounds)
print('your message:',inf_for_crypt, '\n','Incrypted:',Symbolic(Ln, Rn))

L_start,R_start = Feistel_Decrypt(Ln, Rn, my_rounds)          

print('Decrypted:', Symbolic(L_start, R_start))
#Symbolic(L_start, R_start)
