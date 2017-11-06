## train_defense.py
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST
from defensive_models import DenoisingAutoEncoder as DAE
from defensive_models import PackedAutoEncoder as PAE

poolings = ["average", "max"]

shape = [28, 28, 1]
combination_I = [3, "average", 3]
combination_II = [3]
activation = "sigmoid"
reg_strength = 1e-9
epochs = 30

data = MNIST()

"""
AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation,
           reg_strength=reg_strength)
AE_I.train(data, "MNIST_I", num_epochs=epochs)

AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation,
            reg_strength=reg_strength)
AE_II.train(data, "MNIST_II", num_epochs=epochs)
"""

AE_I = PAE(shape, combination_I, data, v_noise=0.1, activation=activation,
            n_pack=8)
AE_I.train(data, "PAE_MNIST_I", alpha=.2, num_epochs=epochs)

AE_II = PAE(shape, combination_II, data, v_noise=0.1, activation=activation,
            n_pack=8)
AE_II.train(data, "PAE_MNIST_II", alpha=.2, num_epochs=epochs)
