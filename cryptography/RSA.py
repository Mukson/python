# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:42:29 2019

@author: Maksim
"""
import random
from math import gcd

inf_for_crypt ='RSA is working correctly'# input()

def Isprime_Ferma(x):
    for i in range(50):
        a = random.randint(1, x-1)
        if gcd(a,x) !=1:
            return False
        if a**(x-1) % x != 1: 
            return False
    return True
            
def p_q():
    while True:
        res = random.randint(1000,2000)
        if Isprime_Ferma(res) == True :
            return res

def Generate_e(_p, _q):
    fi = (_p-1)*(_q-1)
    while True:
        res = random.randint(5000,10000)
        if Isprime_Ferma(res) == True and gcd(res,fi) == 1 and res < fi:
            return res 

def Count_d(_p, _q, _e):
    res = 1
    while (_e*res) % ((_p-1)*(_q-1)) != 1:     
        res += 1
    return res

def Encryption(message, _e, _n):
    res = [(ord(i)**_e % _n) for i in message]    
    return res
    
def Decryption(encrypted_message, _d, _n):
    res = ''
    for i in encrypted_message:
        res += chr((i**_d % _n) )
    return res
    
p = p_q()
q = p_q()
n = p*q
e = Generate_e(p,q)
d = Count_d(p, q, e)
print('open key e,n:',e,n, 'secret key d, p, q:', d,p,q )
crypted = Encryption(inf_for_crypt, e, n)
print('Encypted:', crypted)
decrypted = Decryption(crypted, d, n)
print('Decrypted:', decrypted)
