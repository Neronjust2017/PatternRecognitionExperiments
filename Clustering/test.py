import numpy as np
li = [1 ,2 ,3]

j =len(li)
for i in range(j):
    print(i)
    if li[i] ==2 :
        li.remove(li[i])
        j = j-1
