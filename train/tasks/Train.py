import numpy as np
import torch
import torch.optim as optim
import yaml
from torch import nn

from common.options import args
from train.tasks.segmented.dataset.kitti.parser import Parser
from train.tasks.segmented.modules.Discriminator import PixelDiscriminator
from train.tasks.segmented.modules.TITAN import Titan
from train.tasks.segmented.trainer import Trainer

if __name__ == "__main__":
    if args.epoch_c != 0:
        print("Restarting Training from Epoch:", args.epoch_c)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    discriminator = PixelDiscriminator().cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=list(map(int,args.cuda.split(',')))).cuda()

    generator = Titan().cuda()
    generator = nn.DataParallel(generator, device_ids=list(map(int,args.cuda.split(',')))).cuda()

    ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    parser = Parser(root=args.dataset,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=DATA["split"]["test"],
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True,
                    transform=False)

    optim_gen = optim.Adam(generator.parameters(), lr=ARCH["train"]["lr"], betas=(0.5, 0.999))

    optim_disc = optim.Adam(discriminator.parameters(), lr=ARCH["train"]["lr"], betas=(0.5, 0.999))

    optimizers = {'generator': optim_gen,
                  'discriminator': optim_disc}

    epsilon_w = ARCH["train"]["epsilon_w"]
    content = torch.zeros(parser.get_n_classes(), dtype=torch.float)

    train = Trainer(generator, discriminator, optimizers, parser, args)
    print(train.trainer().run(parser.trainloader, args.epochs))
