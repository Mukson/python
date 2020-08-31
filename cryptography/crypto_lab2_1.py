# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:47:32 2019

@author: Maksim
"""
A,C,b, M, T0 = 5, 3, 7, 128,7
T = []
T.append(T0) # T[0]
inf_for_crypt ='Secret information'# input()
start_alphab = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz .,!:;?-'
lenth = len(inf_for_crypt)
print(len(start_alphab), format(70,'07b'), f'{70:07b}')
#print(bin(1),bin(4), bin(1^4), 1^4,(1+0)%2)
def Gamma(A,C, T1, M, lenth):
    T1.append((A*T1[0]+C) % M)
    for i  in range(1,lenth):
        check_sum = bin(T1[i]).count('1')
        T1.append((A*check_sum + C) % M)
    return T1
def Incryption(arr_for_crypt, alphab, arr_of_gamma):
    res=''
    for i in range(len(arr_for_crypt)):
        new_index = (alphab.index(arr_for_crypt[i])^arr_of_gamma[i+1])% len(alphab)
        res += alphab[new_index]
       # print(i,bin(i),alphab[i],arr_of_gamma[i+1],bin(arr_of_gamma[i+1]),new_index, bin(new_index))
    return res
def Decryption(decrypt_arr,alphab, arr_of_gamma):
    res=''
    for i in range(len(decrypt_arr)):
        prev_index = (alphab.index(decrypt_arr[i])^arr_of_gamma[i+1])% len(alphab)
        res += alphab[prev_index]
    return res
Gamma(A,C,T, M, lenth)
print('Gamma:',T)
incrypted = Incryption(inf_for_crypt, start_alphab, T)
decrypted = Decryption(incrypted, start_alphab, T)
print('your message:',inf_for_crypt,'\n','Incrypted:', incrypted, '\n', 'Decrypted:', decrypted)


