import dynet as dy
dy.renew_cg() # create a new computation graph
v1 = dy.inputVector([1,2,3,4])
v2 = dy.inputVector([5,6,7,8]) # v1 and v2 are expressions
v3 = v1 + v2
v4 = v3 * 2
v5 = v1 + 1
v6 = dy.concatenate([v1,v2,v3,v5])
print v6
print v6.npvalue()
