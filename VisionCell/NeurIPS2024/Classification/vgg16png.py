import torch
import torchvision.models as models
import hiddenlayer as hl

# Build the VGG16 model
vgg16 = models.vgg16()

# Build HiddenLayer graph
hl_graph = hl.build_graph(vgg16, torch.zeros([1, 3, 224, 224]))

# Save the graph to a file
hl_graph.save("./vgg16_hiddenlayer", format="png")
