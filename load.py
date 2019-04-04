from models.optimized.resnet_new import load_resnet as res2
from models.baseline.resnet import load_resnet

a = load_resnet()
b = res2()
kn = list(b.modules())[1].state_dict().keys()
ko = list(a.state_dict().keys())
kn1 = [k.replace('__', '.') for k in kn]

