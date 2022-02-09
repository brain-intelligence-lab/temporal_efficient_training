% This code is from https://github.com/aa-samad/conv_snn
% Please follow their preprocess steps
clc
clear
a = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
% a = {'airplane'};
for folder0 = 1:10
    for file0 = 0:999
        if mod(file0, 10) == 0
           fprintf('step: %s %d\n', a{folder0}, file0)
        end
        addr = sprintf('raw-DVS-CIFAR10\\%s\\cifar10_%s_%d.aedat', a{folder0}, a{folder0}, file0);
        out1 = dat2mat(addr);
        save(sprintf('dvs-cifar10\\%s\\%d.mat', a{folder0}, file0), 'out1')
    end
end