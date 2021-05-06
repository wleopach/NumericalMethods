import  numpy as np
sup=100
i = 1
l = []
while i <= sup-1:
   i = i + 1
   for j in range(2, i):
        if i%j == 0:
           l.append(i)
           break
p = list(range(2, sup))

for s in p:
    if s not in l:
        print(s)

sq = [x**2 for x in range(sup)]

print(sq)