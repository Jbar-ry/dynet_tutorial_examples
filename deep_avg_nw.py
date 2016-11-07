import dynet as dy

model=dy.Model()
V=10000
EDIM=200
model.add_lookup_parameters((V,EDIM))
trainer = dy.SimpleSGDTrainer(model)
closs = 0.0


for ITER in range(1000):
    for x,y in data:
        print x,y
