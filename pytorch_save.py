import sys

import torch
import torchvision
from torch.jit import ScriptModule, script_method, trace
# An instance of your model.
model = torchvision.models.resnet18()

#
# # An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)
#
# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('model.pt')
input_names='INPUT__0'
output_names='OUTPUT__1'
torch.onnx.export(model, example,"model.onnx",  verbose=True)
# torch.onnx.export(model, example,"model.onnx",  verbose=True, input_names=input_names, output_names=output_names)

sys.exit(0)


import torch.nn as nn
dtype=torch.float

class SimplePT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(16, 16)
        self.linear1 = nn.Linear(16, 16)

    def forward(self, INPUT0, INPUT1):
        OUTPUT0 = self.linear0(INPUT1)
        OUTPUT1 = self.linear1(INPUT0)
        return OUTPUT0, OUTPUT1

simplept = SimplePT()
INPUT0 = torch.rand(2, 16,dtype=dtype)
INPUT1 = torch.rand(2, 16,dtype=dtype)
model=simplept.eval()

INPUT0 = torch.rand( 16,dtype=dtype)
INPUT1 = torch.rand( 16,dtype=dtype)

# class SimplePT(torch.jit.ScriptModule):
#     def __init__(self):
#         super().__init__()
#         self.linear0 = trace(nn.Linear(16, 16),INPUT0)
#         self.linear1 = trace(nn.Linear(16, 16),INPUT1)
#
#     @torch.jit.script_method
#     def forward(self, INPUT0, INPUT1):
#         OUTPUT0 = self.linear0(INPUT1)
#         OUTPUT1 = self.linear1(INPUT0)
#         return OUTPUT0, OUTPUT1



simplept.eval()
OUTPUT0, OUTPUT1=simplept(INPUT0,INPUT1)
# traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1))
traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1),(OUTPUT0, OUTPUT1))
# traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1),example_outputs=(OUTPUT0, OUTPUT1))
traced_script_module.eval()
OUTPUT0, OUTPUT1=traced_script_module(INPUT0,INPUT1)
traced_script_module.save("model.pt",)

# model = torch.jit.load("model.pt", map_location='cpu')
print(model,OUTPUT0, OUTPUT1)
a=0

# dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
input_names = [ 'INPUT__0','INPUT__1' ]
output_names = ['OUTPUT__0','OUTPUT__0' ]
# ,example_outputs=(OUTPUT0,OUTPUT1)

torch.onnx.export(model, (INPUT0,INPUT1),"model.onnx",  verbose=True)
# torch.onnx.export(model, (INPUT0,INPUT1),(OUTPUT0,OUTPUT1),"model.onnx",  verbose=True)
# torch.onnx.export(model, (INPUT0,INPUT1),"model.onnx",example_outputs=(OUTPUT0,OUTPUT1),  verbose=True, input_names=input_names, output_names=output_names)
