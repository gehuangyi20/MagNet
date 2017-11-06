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
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator, Attack
from l2_attack import CarliniL2
import utils


sess = keras.backend.set_learning_phase(False)

detector_I = [AEDetector("./defensive_models/PAE_MNIST_I_"+str(i), p=2) for i in range(3)]
detector_II = [AEDetector("./defensive_models/PAE_MNIST_II_"+str(i), p=1) for i in range(3)]
reformer = [SimpleReformer("./defensive_models/PAE_MNIST_I_"+str(i)) for i in range(3)]

id_reformer = IdReformer()
classifier = Classifier("./models/example_classifier")

detector_dict = dict()
for i,det in enumerate(detector_I):
    detector_dict["I"+str(i)] = det
for i,det in enumerate(detector_II):
    detector_dict["II"+str(i)] = det

mnist = MNIST()     
operator = Operator(mnist, classifier, detector_dict, reformer[0])

idx = utils.load_obj("example_idx")
_, _, Y = prepare_data(mnist, idx)
f = "example_carlini_0.0"
testAttack = AttackData(f, Y, "Carlini L2 0.0")

dr = dict([("I"+str(i),.001) for i in range(10)]+[("II"+str(i),.001) for i in range(10)])

dat = mnist.test_data[:16]
print(dat.min(), dat.max())

def show(img):
    remap = "  .*#"+"#"*100
    img = (img.flatten())*3
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

detector0 = detector_dict['I0']
detector1 = detector_dict['II0']

thrs = operator.get_thrs(dr)

def pred2(x):
    probs = classifier.model(reformer[0].model(x))
    
    #prob = tf.reduce_sum([d.tf_mark(x) for d in detector_dict.values()],axis=1)
    #print(prob)
    #res = tf.stack([prob, 1-prob], axis=1)
    #print(res.shape)
    return probs

class Pred2:
    image_size = 28
    num_labels = 10
    num_channels = 1
    def predict(self, x):
        return pred2(x)

sess = keras.backend.get_session()
#attack = cleverhans.attacks.CarliniWagnerL2(pred2, sess=sess)
#adv = attack.generate_np(dat, y_target=np.array([[0, 1]]*len(dat)))
attack = CarliniL2(sess, Pred2(), detector_dict, thrs, batch_size=16,
                   binary_search_steps=5, learning_rate=1e-1,
                   max_iterations=1000, targeted=True,
                   initial_const=1, confidence=1,
                   boxmin=0, boxmax=1)

lab = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]*16)
adv = attack.attack(dat, lab)
np.save("/tmp/q.npy",adv)
#adv = np.load("/tmp/q.npy")

#class Adv:
#    data = adv
#    labels = np.argmax(mnist.train_labels,axis=1)[:10]

for ref in reformer:
    print(np.argmax(classifier.model.predict(ref.model.predict(adv)),axis=1))
print(np.argmax(classifier.model.predict(adv),axis=1))
    
print(thrs)
passes, _ = operator.filter(adv, thrs)
print(passes)

for e in adv:
    show(e)
exit(0)

print(detector0.mark(dat))
print(detector1.mark(dat))
print(detector0.mark(adv))
print(detector1.mark(adv))
exit(0)


Attack(operator)
exit(0)
evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate=dr)

