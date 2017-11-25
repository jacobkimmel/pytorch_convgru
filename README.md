# Convolutional Gated Recurrent Unit (ConvGRU) in PyTorch

These modules implement an individual `ConvGRUCell` and the corresponding multi-cell `ConvGRU` wrapper in [PyTorch](https://pytorch.org).

The ConvGRU is implemented as described in [Ballas *et. al.* 2015: Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/abs/1511.06432).

The `ConvGRUCell` was largely borrowed from [@halochou](https://github.com/halochou). The `ConvGRU` wrapper is based on the [PyTorch RNN source](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#RNN).

# Usage

```python

from convgru import ConvGRU

# Generate a ConvGRU with 3 cells
# input_size and hidden_sizes reflect feature map depths.
# Height and Width are preserved by zero padding within the module.
model = ConvGRU(input_size=8, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3)

x = Variable(torch.FloatTensor(1,8,64,64))
output = model(x)

# output is a list of sequential hidden representation tensors
print(type(output)) # list

# final output size
print(output[-1].size()) # torch.Size([1, 16, 64, 64])
```

## Development

This tool is a product of the [Laboratory of Cell Geometry](https://cellgeometry.ucsf.edu/) at the [University of California, San Francisco](https://ucsf.edu).
