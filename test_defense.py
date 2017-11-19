## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import keras
from setup_mnist import MNIST
from setup_cifar import CIFAR
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator, Attack, DBDetector
from l2_attack import CarliniL2
import utils

dataset = "CIFAR"

keras.backend.set_learning_phase(False)
sess = keras.backend.get_session()

attacking = False

prefix = "O_" if attacking else ""

if dataset == "MNIST":
    detector_I = [AEDetector("./defensive_models/"+prefix+"PAE_MNIST_I_"+str(i), p=1) for i in range(8)]
    if attacking:
        detector_I += [AEDetector("./defensive_models/MNIST_I_"+str(i), p=1) for i in range(8)]
else:
    detector_I = []

if dataset == "MNIST":
    detector_II = [AEDetector("./defensive_models/"+prefix+"PAE_MNIST_II_"+str(i), p=2) for i in range(8)]
    if attacking:
        detector_II += [AEDetector("./defensive_models/MNIST_II_"+str(i), p=2) for i in range(8)]
else:
    detector_II = [AEDetector("./defensive_models/"+prefix+"PAE_CIFAR_II_"+str(i), p=1) for i in range(8)]

if dataset == "MNIST":
    reformer = [SimpleReformer("./defensive_models/"+prefix+"PAE_MNIST_I_"+str(i)) for i in range(8)]
    if attacking:
        reformer += [SimpleReformer("./defensive_models/MNIST_I_"+str(i)) for i in range(24)]
else:
    reformer = [SimpleReformer("./defensive_models/"+prefix+"PAE_CIFAR_II_"+str(i)) for i in range(8)]

id_reformer = IdReformer()

if dataset == "MNIST":
    classifier = Classifier("./models/example_classifier")
else:
    classifier = Classifier("./models/cifar_example_classifier")

if dataset == "MNIST":
    detector_JSD = []
else:
    detector_JSD = [DBDetector(id_reformer, ref, classifier, T=10) for i,ref in enumerate(reformer)]
    detector_JSD += [DBDetector(id_reformer, ref, classifier, T=40) for i,ref in enumerate(reformer)]

detector_dict = dict()
for i,det in enumerate(detector_I):
    detector_dict["I"+str(i)] = det
for i,det in enumerate(detector_II):
    detector_dict["II"+str(i)] = det
for i,det in enumerate(detector_JSD):
    detector_dict["JSD"+str(i)] = det

if dataset == "MNIST":
    data = MNIST()
else:
    data = CIFAR()

operator = Operator(data, classifier, detector_dict, reformer[0])

if dataset == "MNIST":
    dr = dict([("I"+str(i),.001) for i in range(100)]+[("II"+str(i),.001) for i in range(100)])
else:
    dr = dict([("II"+str(i),.005) for i in range(100)]+[("JSD"+str(i),.01) for i in range(100)])

idx = [np.where(np.argmax(data.test_labels,axis=1)==i)[0][0] for i in range(10)]
dat = np.array([data.test_data[i] for i in idx for j in range(10)])
lab = sess.run(tf.one_hot(np.array([list(range(10))]*10).flatten(), depth=10))

def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten())*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

class Pred2:
    image_size = 28 if dataset == "MNIST" else 32
    num_labels = 10
    num_channels = 1 if dataset == "MNIST" else 3
    def __init__(self, reformer):
        self.reformer = reformer
    def predict(self, x):
        return classifier.model(self.reformer.model(x))

if attacking:
    thrs = operator.get_thrs(dict((k,v*4) for k,v in dr.items()))

    attack = CarliniL2(sess, [Pred2(x) for x in reformer], detector_dict, thrs, batch_size=100,
                       binary_search_steps=4, learning_rate=1e-2,
                       max_iterations=10000, targeted=True,
                       initial_const=1, confidence=1,
                       boxmin=0, boxmax=1)
    
    adv = attack.attack(dat, lab)
    np.save("/tmp/"+dataset+".npy",adv)
else:
    adv = np.load("/tmp/"+dataset+".npy")
print('mean distortion', np.mean(np.sum((adv-dat)**2,axis=(1,2,3))**.5))

for i,ref in enumerate(reformer):
    print('reformer',i)
    predicted = np.argmax(classifier.model.predict(ref.model.predict(adv)),axis=1)
    print(np.mean(predicted==np.argmax(lab,axis=1)))#,predicted)

predicted = np.argmax(classifier.model.predict(adv),axis=1)
print('without reformer')
print(np.mean(predicted==np.argmax(lab,axis=1)),predicted)
    
thrs = operator.get_thrs(dr)
print(thrs)
passes, _ = operator.filter(adv, thrs)
print('rate of passing fooling', float(len(passes))/len(dat))#, passes)

exit(0)
for e in adv:
    show(e)

detector0 = detector_dict['I0']
detector1 = detector_dict['II0']

print(detector0.mark(dat))
print(detector1.mark(dat))
print(detector0.mark(adv))
print(detector1.mark(adv))
exit(0)


idx = utils.load_obj("example_idx")
_, _, Y = prepare_data(data, idx)
f = "example_carlini_0.0"
testAttack = AttackData(f, Y, "Carlini L2 0.0")

Attack(operator)
exit(0)
evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate=dr)

