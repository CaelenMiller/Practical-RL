from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
import torch

'''This is an example custom model for use with RLlib. Look at [TrainGridWorld.py] for example usage'''
class CustomModel(TorchModelV2, nn.Module):
    #The inputs need to be exactly this or RLlib will die. 
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = kwargs["obs_size"]
        self.action_size = kwargs["action_size"]
        self.layers = kwargs["num_layers"]

        self.input_layer = nn.Linear(self.obs_size, 256)

        self.net = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256)
            ) for i in range(self.layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(256, self.action_size)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Assuming the input dict contains 'obs' key with the observation
        obs = input_dict["obs"]

        self._features =  self.net(self.input_layer(obs))

        self._output = self.output_layer(self._features)

        return self._output, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward first!"
        return torch.reshape(torch.mean(self._features, -1), [-1])