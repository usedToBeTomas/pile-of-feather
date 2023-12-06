from pileoffeather import nn, data_loader
import numpy as np
import time

def correct():
    print("\033[92m" + "## Passed ##" + "\033[0m")

def notcorrect():
    print("\033[91m" + "## Not Passed ##" + "\033[0m")


print("__________________________________________START_TEST_______________________________________________")
print("nn.create()")
try:
    model = nn.create(layers = [[400, 'input'], [30, 'relu'], [10, 'relu'], [1, 'sigmoid']], name = "test1")
except Exception as e:
    print(e)
    notcorrect()
else:
    correct()

print("_____________________________________________")
print("model.layers")
if model.layers == [[400, 'input'], [30, 'relu'], [10, 'relu'], [1, 'sigmoid']]:
    correct()
else:
    notcorrect()

print("_____________________________________________")
print("dataLoader.load()")
try:
    ones = data_loader.load(data_type = "image", color = "grayscale", folder = "../examples/image_classifier/ones", resize = (20,20))
    zeros = data_loader.load(data_type = "image", color = "grayscale", folder = "../examples/image_classifier/zeros", resize = (20,20))
    input = np.vstack((ones, zeros))
    output = np.concatenate((np.ones(500), np.zeros(500)))
except Exception as e:
    print(e)
    notcorrect()
else:
    correct()

print("_____________________________________________")
print("nn.mbgd() - no errors")
try:
    st = time.time()
    nn.mbgd(model, input, output, batch_size = 16, epoch_number = 10, rate = 0.6)
    et = time.time()
    print("Time = " + str(et - st))
except Exception as e:
    print(e)
    notcorrect()
else:
    correct()

print("_____________________________________________")
print("model.run()")
try:
    st = time.time()
    for i in range(500):
        model.run(ones[i])
    for i in range(500):
        model.run(zeros[i])
    et = time.time()
    print("Time = " + str(et - st))
except Exception as e:
    print(e)
    notcorrect()
else:
    correct()

print("_____________________________________________")
print("nn.mbgd(), model.run() - working correctly")
try:
    counter = 0
    for i in range(500):
        if round(model.run(ones[i])[0],3) > .5:
            counter += 1
    for i in range(500):
        if round(model.run(zeros[i])[0],3) < .5:
            counter += 1

    print(str(counter/10) + "% Accuracy")
    if counter/10 > 99:
        correct()
    else:
        notcorrect()
except Exception as e:
    print(e)
    notcorrect()

print("_____________________________________________")
print("model.pop(), model.insert()")
try:
    model1 = nn.load(name = "test1")
    model.pop(2)
    model1.pop(0)
    model1.pop(0)
    model1.pop()
    model.insert(model1,2)
    counter = 0
    for i in range(500):
        if round(model.run(ones[i])[0],3) > .5:
            counter += 1
    for i in range(500):
        if round(model.run(zeros[i])[0],3) < .5:
            counter += 1

    print(str(counter/10) + "% Accuracy")
    if counter/10 > 99:
        correct()
    else:
        notcorrect()
except Exception as e:
    print(e)
    notcorrect()
