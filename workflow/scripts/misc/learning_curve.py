import re
from matplotlib import pyplot as plt
import os
import sys
print("usage: python learning_curve.py sample.train.log output.prefix(e.g. res/sample.t)")
log_file = open(sys.argv[1])
oname=sys.argv[2]
oname1=oname+"LearningCurve.loss.pdf"
oname2=oname+"LearningCurve.F1.pdf"
print("for", log_file)
print("will output", oname1, oname2)

# Read and parse log
flag = False
epoch = []
loss_val = []
loss_train = []
f1_val = []
f1_train = []

for position, line in enumerate(log_file):
    if line.startswith("Epoch 1"):
        flag = True
    if line.startswith("Restoring model weights from the end of the best epoch"):
        flag = False
        
    if flag:
        #print(line)
        if line.startswith("Epoch"):
            epoch.append(int(re.findall("(\d+)/", line)[0]))
        else:
            loss_train.append(float(re.findall("- loss: ([\d.]+)", line)[0]))
            loss_val.append(float(re.findall("- val_loss: ([\d.]+)", line)[0]))  
            f1_train.append(float(re.findall("- F1: ([\d.]+)", line)[0]))
            f1_val.append(float(re.findall("- val_F1: ([\d.]+)", line)[0]))


# Loss learning curve
plt.plot(epoch, loss_train, label = "loss_train")
plt.plot(epoch, loss_val, label = "loss_val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy")
plt.legend()
plt.savefig(oname1)
plt.close()


# F1 learning curve
plt.plot(epoch, f1_train, label = "F1_train")
plt.plot(epoch, f1_val, label = "F1_val")
plt.xlabel("Epoch")
plt.ylabel("F1 score")
plt.legend()
plt.savefig(oname2)
plt.close()

