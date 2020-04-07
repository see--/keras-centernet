import torch
from keras.models import *

from keras_centernet.models.decode import CtDetDecode
from keras_centernet.models.networks.hourglass import HourglassNetwork

K.set_learning_phase(0)

def convert_weights_to_keras(keras_model,py_state_dict):
    for i, layer in enumerate(keras_model.layers):
        if len(layer.weights) > 0:
            if 'conv' in layer.name or 'skip.0' in layer.name or 'cnvs_.0.0' in layer.name or 'inters_.0.0' in layer.name:
                layer_weights_name="{}.weight".format(layer.name)

                weight_size = len(py_state_dict['state_dict'][layer_weights_name].size())
                transpose_dims = []

                for j in range(weight_size):
                    transpose_dims.append(weight_size - j - 1)
                '''exchange the  first two dim,o: [out_c, in_c, k_h, k_w]
                 Keras: [k_h, k_w, in_c, out_c]
                 '''
                transpose_dims[0],transpose_dims[1] = transpose_dims[1],transpose_dims[0]

                weights=py_state_dict['state_dict'][layer_weights_name].numpy().transpose(transpose_dims)
                if  layer.use_bias:
                    layer_bias_name = "{}.bias".format(layer.name)
                    bias = py_state_dict['state_dict'][layer_bias_name].numpy()
                    keras_model.layers[i].set_weights([weights,bias])
                else:
                    keras_model.layers[i].set_weights([weights])
                print("load {} weights".format(layer.name))
            elif 'bn' in layer.name or 'skip.1' in layer.name or 'cnvs_.0.1' in layer.name or 'inters_.0.1' in layer.name:
                layer_weight_name="{}.weight".format(layer.name)
                layer_bias_name="{}.bias".format(layer.name)
                layer_mean_name="{}.running_mean".format(layer.name)
                layer_var_name="{}.running_var".format(layer.name)
                keras_model.layers[i].set_weights([py_state_dict['state_dict'][layer_weight_name].numpy(),
                                                   py_state_dict['state_dict'][layer_bias_name].numpy(),
                                                   py_state_dict['state_dict'][layer_mean_name].numpy(),
                                                   py_state_dict['state_dict'][layer_var_name].numpy()])
                print("load {} weights".format(layer.name))
            elif 'hm' in layer.name or 'wh' in layer.name or 'reg' in layer.name:
                layer_weight_name = "{}.weight".format(layer.name)
                layer_bias_name = "{}.bias".format(layer.name)

                weight_size = len(py_state_dict['state_dict'][layer_weight_name].size())
                transpose_dims = []

                for j in range(weight_size):
                    transpose_dims.append(weight_size - j - 1)

                transpose_dims[0], transpose_dims[1] = transpose_dims[1], transpose_dims[0]

                keras_model.layers[i].set_weights([py_state_dict['state_dict'][layer_weight_name].numpy().transpose(transpose_dims),
                                               py_state_dict['state_dict'][layer_bias_name].numpy()])
                print("load {} weights".format(layer.name))


"""
Create hourglass_model
"""


kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': None,
    'inres': (512,512),
}
heads = {
    'hm': 1,
    'reg': 2,
    'wh': 2
}
keras_model = HourglassNetwork(heads=heads, **kwargs)
keras_model = CtDetDecode(keras_model)

keras_model.summary()
"""
load the weights from model_best.pth,I trained the centernet with official PyTorch code,and the classes is 1
"""
py_model = torch.load("model_best.pth",map_location=torch.device('cpu'))

"""
convert and load weitghts to the keras model
"""
convert_weights_to_keras(keras_model,py_model)

# Save the weights of the converted keras model for later use
keras_model.save_weights("cenetnet_hg.h5")




