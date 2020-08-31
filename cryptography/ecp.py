# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:57:12 2019

@author: Maksim
"""
# Электронная подпись ГОСТ 34.10-94
from math import gcd
import random
import hashlib

def Isprime_Ferma(x): # prime test by ferma
    for i in range(3):
        a = random.randint(1, x-1)
        if gcd(a,x) !=1:
            return False
        if a**(x-1) % x != 1: 
            return False
    return True

def Gen_p():
    while True:
        res = random.randint(1000, 10000)
        if Isprime_Ferma(res) == True :
            return res
def Gen_q(_p):
    while True:
        res = random.randint(100, _p-2)
        if Isprime_Ferma(res) == True and (_p-1) % res  == 0 :
            return res
        else:
            continue
        
def Count_a(_p,_q):
    res = 2 
    while  res < _p -1:
        if (res**_q) % _p !=1: 
            res += 1
        else:
            break
    return res


#Разбиение исходного текста на блоки по 16 бит 
#и выполнение над ними операции сложения по модулю 2
# и циклического сдвига на 1 разряд вправо при каждой операции XOR
def Hash(m): # do hash)) 
    blocks = ''
    count = 0
    for i in m:
        blocks += i 
        count +=1 
        if count == 2:
            blocks += '|' 
            count = 0 
    _16bit_block = blocks[:-1].split('|')
    flag = 0 
    #print(_16bit_block)
    for i in  _16bit_block:
        if flag == 0:
            res =  int(format(ord(i[0]), '08b')+format(ord(i[1]), '08b'),2)
            flag +=1
        else:
            res = (res ^ int(format(ord(i[0]), '08b')+format(ord(i[1]), '08b'),2)) >> 1
                    
    return res
def Gen_ecp(_p, _q, _a, _x, _hash):# Generation ecp
    
    k = random.randint(1, _q)
    r = _a**k %_p
    r1 = r % _p
    
    if r1 == 0 :
        print('r1 =0')
        return 1
    
    _s = (_x * r1 + k * _hash) % _q
    if _s == 0:
        print('s = 0')
        return 2
    print('ecp :','r1 = ', r1,'s = ', _s)
    return r1,_s

def Chek_ecp(_p, _q, _a, _y, _r1, _s, _hash): # проверка подписи
   
    v = (_hash**(_q-2)) % _q
    z1 = (_s*v) % _q
    z2 = (v*(_q - _r1)) % _q
    u = ((_a**z1 * _y**z2) % _p) % _p
    if u == _r1:
        print('ecp is correct')
    else:
        print('err 2: wrong ecp')
        
def make_sha1(str, encoding='utf-8'):
    return int.from_bytes(hashlib.sha1(str.encode(encoding)).digest(), "little")
    

        
m = 'test ecp'#input()

if len(m) % 2 !=0:
    m += '#' 
    
#my_hash = Hash(m)
my_hash =make_sha1(m)
print('hash =',hex(my_hash))
p = Gen_p()
print('p = ',p)
q = Gen_q(p)
print('q = ',q)
a= Count_a(p,q)
print('a = ',a)
x = random.randint(2,q - 1)
y = (a ** x) % p     
r1, s = Gen_ecp(p, q, a, x, my_hash)  
Chek_ecp(p, q, a, y, r1, s, my_hash)

r1 += 1
print('changed ecp:', 'r1 =', r1, 's=',s)
Chek_ecp(p, q, a, y, r1, s, my_hash)
