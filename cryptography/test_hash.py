# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:07:02 2019

@author: Maksim
"""

import hashlib
str = "ecp"
def make_sha1(str, encoding='utf-8'):
    return int.from_bytes(hashlib.sha1(str.encode(encoding)).digest(), "little")
    
print("Hash:",hex((make_sha1(str, encoding='utf-8'))))