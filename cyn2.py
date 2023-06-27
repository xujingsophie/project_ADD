#第四章课后习题
#1、针对python的str对象，实现一个replace操作函数

#首先对整个串做一次匹配，确定代换串出现的位置和次数。然后替换
#import re
#def replace(self,str1,str2):
    #朴素匹配算法
    #只能实现一处替换，循环怎么写？？
def replace(self,str1,str2):
        m, n = len(str1) , len(self)
        i, j = 0 , 0
        while i <m and j<n :
            if str1[i] == self[j]:
                i = i+1
                j = j+1
            else:
                i = 0
                j = j - i+1
        if i == m:
            # return j - i  #j-i出处匹配到字符串1
            #取得匹配到字符串，用str2替换 ???怎么替换
            return substr(self,j-i,j-i+m,str2)
            #self[j-i,j-i+m]
        return -1
  #取得位置[a:b]的字串并用字串2替换
def substr(self,a,b,str2):
    self = self[:a] + str2 + self[b:]
    return self


#print(match('acaabcbabc','ab'))
print(replace('aaabbbccc','ab','11'))