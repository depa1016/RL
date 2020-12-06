

dic = {}
dic2 = {}
#dic[0][0] = 1

dic[(0,0)] = 0
dic[(1,0)] = 1
dic[(2,0)] = 2
relevant_qs = [dic[(action, 0)] for action in range(0, 3)]
for i in range(0,3):
    print(dic[(i,0)])
