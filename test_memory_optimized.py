import pdb
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest, time, sys

import models.optimized.densenet_new as densenet_optim
import models.optimized.resnet_new as resnet_optim


class TestMemoryOptimized(unittest.TestCase):

    def test_resnet_optim(self):
        N = 20
        # N = 51
        total_iters = 20    # (warmup + benchmark)
        iterations = 1
        chunks = 6

        target = torch.ones(N).type("torch.LongTensor")
        x = torch.ones(N, 3, 224, 224, requires_grad=True)
        # x = torch.ones(N, 3, 32, 32, requires_grad=True)
        # model = resnet_optim.resnet200()
        # model = resnet_optim.resnet101()
        model = resnet_optim.resnet50()
        # model = resnet_optim.resnet1001()

        # switch the model to train mode
        model.train()

        # convert the model and input to cuda
        model = model.cuda()
        input_var = x.cuda()
        target_var = target.cuda()

        # declare the optimizer and criterion
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
        optimizer.zero_grad()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with cudnn.flags(enabled=True, benchmark=True):
            for i in range(total_iters):
                start.record()
                start_cpu = time.time()
                for j in range(iterations):
                    output = model(input_var, chunks=chunks)
                    loss = criterion(output, target_var)
                    loss.backward()
                    optimizer.step()

                end_cpu = time.time()
                end.record()
                torch.cuda.synchronize()
                gpu_msec = start.elapsed_time(end)
                print("Optimized resnet ({:2d}): ({:8.3f} usecs gpu) ({:8.3f} usecs cpu)".format(
                    i, gpu_msec * 1000, (end_cpu - start_cpu) * 1000000,
                    file=sys.stderr))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

if __name__ == '__main__':
    unittest.main()
