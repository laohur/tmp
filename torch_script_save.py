import sys

import torch
# import torchvision
from torch.jit import ScriptModule, script_method, trace
# An instance of your model.
# model = torchvision.models.resnet18()
#
# #
# # # An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)
# #
# # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save('model.pt')
# input_names='INPUT__0'
# output_names='OUTPUT__1'
# torch.onnx.export(model, example,"model.onnx",  verbose=True)
# # torch.onnx.export(model, example,"model.onnx",  verbose=True, input_names=input_names, output_names=output_names)
#
# sys.exit(0)


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
# my_module.save("my_module_model.pt")




import torch.nn as nn
dtype=torch.float

class SimplePT(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear0 = nn.Linear(16, 16)
        # self.linear1 = nn.Linear(16, 16)
    # def forward(self, INPUT0):
    #     OUTPUT0 = self.linear0(INPUT0)
    #     return OUTPUT0

    def forward(self, INPUT0, INPUT1):
        # OUTPUT0 = self.linear0(INPUT1)
        # OUTPUT1 = self.linear1(INPUT0)
        # return OUTPUT0, OUTPUT1
        return INPUT1+INPUT0,INPUT0-INPUT1

simplept = SimplePT()
# INPUT0 = torch.rand(2, 16,dtype=dtype)
# INPUT1 = torch.rand(2, 16,dtype=dtype)
# model=simplept.eval()

INPUT0 = torch.rand( 16,dtype=dtype)
INPUT1 = torch.rand( 16,dtype=dtype)

# class SimplePT(torch.jit.ScriptModule):
#     def __init__(self):
#         super().__init__()
#         # self.linear0 = trace(nn.Linear(16, 16),INPUT0)
#         # self.linear1 = trace(nn.Linear(16, 16),INPUT1)
#
#     @torch.jit.script_method
#     def forward(self, INPUT0, INPUT1):
#         # OUTPUT0 = self.linear0(INPUT1)
#         # OUTPUT1 = self.linear1(INPUT0)
#         return INPUT1+INPUT0,INPUT0-INPUT1



# simplept.eval()
# OUTPUT0=simplept(INPUT0)
# OUTPUT0, OUTPUT1=simplept(INPUT0,INPUT1)
# OUTPUT0, OUTPUT1=SimplePT((INPUT0,INPUT1))
# traced_script_module = torch.jit.trace(simplept, INPUT0 )
traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1))
# traced_script_module = torch.jit.script(simplept)
# traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1),(OUTPUT0, OUTPUT1))
# traced_script_module = torch.jit.trace(simplept, (INPUT0, INPUT1),example_outputs=(OUTPUT0, OUTPUT1))
# traced_script_module.eval()
# OUTPUT0=traced_script_module(INPUT0)
OUTPUT0, OUTPUT1=traced_script_module(INPUT0,INPUT1)
traced_script_module.save("model.pt",)
model=simplept
# model = torch.jit.load("model.pt", map_location='cpu')
print(model,OUTPUT0)
a=0

# dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
input_names = [ 'INPUT__0','INPUT__1' ]
output_names = ['OUTPUT__0','OUTPUT__0' ]
# ,example_outputs=(OUTPUT0,OUTPUT1)

# torch.onnx.export(model, INPUT0,"model.onnx",  verbose=True)
torch.onnx.export(model, (INPUT0,INPUT1),"model.onnx",  verbose=True)
# torch.onnx.export(model, (INPUT0,INPUT1),(OUTPUT0,OUTPUT1),"model.onnx",  verbose=True)
# torch.onnx.export(model, (INPUT0,INPUT1),"model.onnx",example_outputs=(OUTPUT0,OUTPUT1),  verbose=True, input_names=input_names, output_names=output_names)
