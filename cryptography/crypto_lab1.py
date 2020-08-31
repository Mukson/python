inf_for_crypt = input()
start_alphab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!:;?-'
crypto_alphab = 'CDABHIJEFGOPQRKLMNUVW:STZ XY;?-.,!'

def incryption(inf_for_crypt, start_alphab, crypto_alphab):
    res=''    
    for i in range(len(inf_for_crypt)):
        for j in range(len(start_alphab)):
          if inf_for_crypt[i] == start_alphab[j] or inf_for_crypt[i].upper() == start_alphab[j]:
              res +=crypto_alphab[j]
          else:
              continue    
    print('incrypted:',res)
    return(res)
    
def decryption(inf_for_decrypt, start_alphab, crypto_alphab):
    res=''    
    for i in range(len(inf_for_decrypt)):
        for j in range(len(crypto_alphab)):
          if inf_for_decrypt[i] == crypto_alphab[j]:
              res +=start_alphab[j]
          else:
              continue    
    print('decrypted:',res)
    return

inf_for_decrypt = incryption(inf_for_crypt, start_alphab, crypto_alphab)
decryption(inf_for_decrypt, start_alphab, crypto_alphab)