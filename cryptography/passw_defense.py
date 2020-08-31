# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:01:55 2019

@author: Maksim
"""
import time

password = 'passwd'
num_try = 2
i = 0
print("Enter your password: ")
while True:
    ent_pwd = input()
    if ent_pwd == password:
        print('Welcome!')
        break
    else:
        print('Wrong password, try again...')
        i += 1
        if i == num_try:
            i = 0
            print ('will unblock in 10 sec')
            time.sleep(10)
            print("Enter your password: ")
    