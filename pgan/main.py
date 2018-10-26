from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import models

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--trainRateG', type=int, default=1, help='train rate of G:D')
parser.add_argument('--trainRateD', type=int, default=1, help='train rate of D:G')
parser.add_argument('--activatePG', action='store_true', help='activate Generator with PNN')
parser.add_argument('--activatePD', action='store_true', help='activate Discriminator with PNN')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--activateWGAN', action='store_true', help='activate WassersteinGAN')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
if opt.dataset == 'mnist':
    nc = 1
else:
    nc = 3

if opt.activatePG:
    print ('activate Generator with PNN...')
    #netG = models.P_Generator(ngpu, nz, ngf, nc).to(device)
    netG = models.noisegenerator101(nchannels=nc, nfilters=opt.nz, nclasses=1)
else:
    print ('activate Generator with CNN...')
    netG = models.Generator(ngpu, nz, ngf, nc).to(device)
netG = netG.cuda()
netG.apply(models.weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
#print(netG)

if opt.activatePD:
    print ('activate Discriminator with PNN...')
    netD = models.noiseresnet18(nchannels=nc, nfilters=128, nclasses=1 , pool=2)
else:
    print ('activate Discriminator with CNN...')
    netD = models.Discriminator(ngpu, nz, ndf, nc).to(device)
    #netD = naiveresnet.noiseresgenerator18(nchannels=3, nfilters=128, nclasses=1)
netD.apply(models.weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
#print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
#one = torch.FloatTensor([1])
#one = torch.tensor(1.0)
one = torch.FloatTensor(torch.ones([opt.batchSize]))
mone = one * -1
real_label = 1
fake_label = 0

#print ('update parameters:',list(filter(lambda p: p.requires_grad,netG.parameters())))

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    print ('optimizer:adam')
    if opt.activatePG:
        optimizerG = optim.Adam(filter(lambda p: p.requires_grad,netG.parameters()), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=1e-8)
    else:
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=1e-8)

    if opt.activatePD:
        optimizerD = optim.Adam(filter(lambda p: p.requires_grad,netD.parameters()), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=1e-8)
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=1e-8)
else:
    print ('optimizer:RMSprop')
    if opt.activatePG:
        optimizerG = optim.RMSprop(filter(lambda p: p.requires_grad,netG.parameters()), lr=opt.lrG)
    else:
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    if opt.activatePD:
        optimizerD = optim.RMSprop(filter(lambda p: p.requires_grad,netD.parameters()), lr=opt.lrD)
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)

# START TRAINING
if opt.activateWGAN:
    print ('Algorithm : WGAN')
    # ---WGAN Training---
    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                netD.zero_grad()

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                print (real_cpu.size())
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = netD(inputv)
                #errD_real.backward(one)
                errD_real.backward(one[0:inputv.size()[0]])
                print (one[0:inputv.size()[0]].size())

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD(inputv)
                #errD_fake.backward(mone)
                errD_fake.backward(mone[0:inputv.size()[0]])
                print (mone[0:inputv.size()[0]].size())

                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            if gen_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.outf))
                fake = netG(Variable(fixed_noise, volatile=True))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.outf, gen_iterations))
    # ---WGAN Training End---
else:
    print ('Algorithm : DCGAN')
    criterion = nn.BCELoss()
    # ---DCGAN Training---
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            for j in range(opt.trainRateD):
                # train with real
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                #batch_size = opt.batchSize
                label = torch.full((batch_size,), real_label, device=device)

                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

            for j in range(opt.trainRateG):
                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % opt.outf,
                            normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d_%04d.png' % (opt.outf, epoch, i),
                            normalize=True)
    # ---DCGAN Training End---

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
