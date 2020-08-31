A,C,b, M = 5, 3, 7, 128
T = []
T.append(7) # T[0]
inf_for_crypt ='abc'# input()
start_alphab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!:;?-'
to_binary = ''
lenth = len(inf_for_crypt)
#перевод в бинарный вид 
for i in start_alphab:
    to_binary = to_binary + f'{start_alphab.index(i):07b}' + ' '
bin_arr = to_binary.split()

#подсчет контрольной суммы
#def Check_Sum(num):
 #   k = f'{num:07b}'.count('1')
  #  print('check sum:', k)
   # return k
def Gamma(A,C,b, T1, M, lenth):
    T1.append((A*T1[0]+C) % 128)
    #check_sum = f'{T[1]:07b}'.count('1')
    #print('check_sum', check_sum)
    for i  in range(1,lenth):
        check_sum = f'{T1[i]:07b}'.count('1')
        T1.append((A*check_sum + C) % 128)
    
        #print('i',i ,'check_sum', check_sum)
    print('T1:',T1)
    return T1
def Encryption(bin_arr_for_crypt, arr_of_bin_gamma):
    for i in bin_arr_for_crypt:
        for k in bin_arr_for_crypt[i]:
            (bin_arr_for_crypt[i][k] + arr_of_bin_gamma[i][k]) % 2
    
Gamma(A,C,b, T, M, lenth)
print('T:',T)
print(bin(3^2).count('1'), bin(3^2))