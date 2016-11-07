import dynet as dy
import random
data =[ ([0,1],0),
        ([1,0],0),
        ([0,0],1),
        ([1,1],1) ]
model = dy.Model()
pU = model.add_parameters((4,2))
pb = model.add_parameters(4)
pv = model.add_parameters(4)


trainer = dy.SimpleSGDTrainer(model)
closs = 0.0


for ITER in range(1000):
    for x,y in data:
        dy.renew_cg()
        U = dy.parameter(pU)
        b = dy.parameter(pb)
        v = dy.parameter(pv)
        x = dy.inputVector(x)
        yhat = dy.logistic(dy.dot_product(v,dy.tanh(U*x+b)))
        if y == 0:
            loss = -dy.log(1 - yhat)
        elif y == 1:
            loss = -dy.log(yhat)
        closs += loss.scalar_value() # forward
        loss.backward()
        trainer.update()
    if ITER > 0 and ITER % 100 == 0:
        trainer.status()
        print "Iter:",ITER,"loss:", closs/400
        closs = 0
