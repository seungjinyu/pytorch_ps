import torch

class ParameterServer:
    def __init__(self):
        self.weights = torch.ones(10)  # 간단한 예시

    def get_weights(self):
        return self.weights

    def update_weights(self, gradients):
        self.weights -= gradients
