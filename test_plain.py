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

dataset = "MNIST"

if dataset == "MNIST":
    classifier = Classifier("./models/example_classifier")
else:
    classifier = Classifier("./models/cifar_example_classifier")

if dataset == "MNIST":
    data = MNIST()
else:
    data = CIFAR()

class Pred2:
    image_size = 28 if dataset == "MNIST" else 32
    num_labels = 10
    num_channels = 1 if dataset == "MNIST" else 3
    def predict(self, x):
        return classifier.model(x)


keras.backend.set_learning_phase(False)
sess = keras.backend.get_session()
attack = CarliniL2(sess, [Pred2()], {}, {}, batch_size=100,
                   binary_search_steps=4, learning_rate=1e-2,
                   max_iterations=10000, targeted=True,
                   initial_const=1, confidence=1,
                   boxmin=0, boxmax=1)

idx = [np.where(np.argmax(data.test_labels,axis=1)==i)[0][0] for i in range(10)]
dat = np.array([data.test_data[i] for i in idx for j in range(10)])
lab = sess.run(tf.one_hot(np.array([list(range(10))]*10).flatten(), depth=10))

adv = attack.attack(dat, lab)
print('mean distortion', np.mean(np.sum((adv-dat)**2,axis=(1,2,3))**.5))

