from PIL import Image
from torchvision import transforms
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.engine import Engine
import numpy as np
from ignite.metrics import RunningAverage
from common.checkpoint import ModelCheckpoint
from ignite.handlers import Timer
import os
from torch import nn
import torch
from torch.autograd import Variable
from train.tasks.segmented.modules.Lovasz_Softmax import Lovasz_softmax
from torchvision.utils import save_image
from SSIM_PIL import compare_ssim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from train.tasks.segmented.modules.ioueval import iouEval

palette = [0, 0, 0,  # 0 outlier
           100, 150, 245,  # 1 car
           100, 230, 245,  # 2 bicycle
           30, 60, 150,  # 3 motorcycle
           80, 30, 180,  # 4 truck
           0, 0, 250,  # 5 other-vehichle
           255, 30, 30,  # 6 person
           255, 0, 255,  # 7 road
           75, 0, 75,  # 8 sidewalk
           255, 200, 0,  # 9 building
           255, 120, 50,  # 10 fence
           0, 175, 0,  # 11 vegetation
           150, 240, 80,  # 12 terrain
           255, 240, 150,  # 13 pole
           255, 0, 0  # 14 traffic-sign
           ]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_gradient_penalty(D, real_images, fake_images, rgb):
    eta = torch.Tensor(1, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(1, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated, rgb)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class Trainer():
    def __init__(self, generator, discriminator, optimizer, parser, args):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.parser = parser
        self.args = args
        self.timer = Timer(average=True)
        self.criterion_gan = nn.MSELoss(reduction='mean')
        self.criterion_pixel = nn.L1Loss(reduction='mean')
        self.validDataLoader = self.parser.validloader
        self.trainDataLoader = self.parser.trainloader
        self.testDataLoader = self.parser.testloader
        self.epoch_c = args.epoch_c
        self.ls = Lovasz_softmax(ignore=0).cuda()
        self.scaler = GradScaler()
        self.evaluator = iouEval(15, "cuda", [0])
        self.info = {"g_loss": 0,
                     "d_loss": 0,
                     "guiding_loss": 0,
                     "wdistance": 0,
                     "lr": 0,
                     "ssim": 0,
                     "psnr": 0,
                     'miou': 0,
                     'acc': 0}

        if self.epoch_c != 0:
            self.generator.load_state_dict(
                torch.load(args.output_dir + "/checkpoints/training_generator_{}.pth".format(
                    self.epoch_c)))
            self.discriminator.load_state_dict(
                torch.load(
                    args.output_dir + "/checkpoints/training_discriminator_{}.pth".format(
                        self.epoch_c)))
            self.optimizer['discriminator'].load_state_dict(torch.load(
                args.output_dir + "/checkpoints/training_optimizer_D_{}.pth".format(
                    self.epoch_c)))
            self.optimizer['generator'].load_state_dict(torch.load(
                args.output_dir + "/checkpoints/training_optimizer_G_{}.pth".format(
                    self.epoch_c)))

        @autocast()
        def step(engine, batch):
            mone = torch.tensor([-1]).cuda()
            rgb, rgbseg, lidarseg, rgb_labels, proj_labels, proj_mask, proj, rgb_labels_one_hot, proj_labels_one_hot = batch
            proj = proj.cuda(non_blocking=True)
            rgb_labels_one_hot = rgb_labels_one_hot.cuda(non_blocking=True)
            proj_labels_one_hot = proj_labels_one_hot.cuda(non_blocking=True)
            rgb_labels = rgb_labels.unsqueeze(1).cuda(non_blocking=True)

            for p in self.discriminator.module.parameters():
                p.requires_grad = True

            self.discriminator.zero_grad()
            fake = self.generator(proj, proj_labels_one_hot)
            fake_detach = fake.detach()

            dloss_real = self.discriminator(rgb_labels_one_hot, proj_labels_one_hot)
            dloss_real = dloss_real.mean()
            self.scaler.scale(dloss_real).backward(mone, retain_graph=True)

            dloss_fake = self.discriminator(fake_detach, proj_labels_one_hot)
            dloss_fake = dloss_fake.mean()
            self.scaler.scale(dloss_fake).backward(-1 * mone, retain_graph=True)

            gp = compute_gradient_penalty(self.discriminator, rgb_labels_one_hot.data, fake_detach.data,
                                          proj_labels_one_hot.data)
            self.scaler.scale(gp).backward(retain_graph=True)
            self.scaler.step(self.optimizer['discriminator'])

            d_loss = dloss_fake - dloss_real + 10 * gp
            wdistance = dloss_real - dloss_fake
            for p in self.discriminator.module.parameters():
                p.requires_grad = False

            self.generator.zero_grad()

            fake = self.generator(proj, proj_labels_one_hot)
            g_loss = self.discriminator(fake, proj_labels_one_hot)
            g_loss = g_loss.mean()
            self.scaler.scale(g_loss).backward(mone, retain_graph=True)

            guiding_loss = (self.ls(fake, rgb_labels))
            self.scaler.scale(guiding_loss).backward()
            self.scaler.step(self.optimizer['generator'])

            if engine.state.iteration % self.args.print_freq == 0:
                directory = os.path.dirname(self.args.output_dir + "/examples/epoch_{}/".format(engine.state.epoch))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                colorize_mask(rgb_labels[0][0].cpu().numpy()).save(
                    args.output_dir + "/examples/epoch_{}/real_rbgbseg.png".format(engine.state.epoch))

                lidarseg_img = lidarseg[0]
                save_image(lidarseg_img,
                           self.args.output_dir + "/examples/epoch_{}/real_lidarseg.png".format(engine.state.epoch))
                gpc_fake_argmax = fake.argmax(dim=1, keepdim=True)
                colorize_mask(gpc_fake_argmax[0][0].cpu().numpy()).save(
                    self.args.output_dir + "/examples/epoch_{}/generated_camseg.png".format(engine.state.epoch))
            self.scaler.update()
            return {'g_loss': -g_loss.mean().item(),
                    'd_loss': d_loss.mean().item(),
                    'guiding_loss': guiding_loss.mean().item(),
                    'wdistance': wdistance.mean().item(),
                    'lr': self.optimizer['discriminator'].param_groups[0]['lr']}

        trainer = Engine(step)
        self.t = trainer

        checkpoint_handler = ModelCheckpoint(self.args.output_dir + '/checkpoints/', 'training',
                                             save_interval=self.args.checkpoint_interval,
                                             n_saved=self.args.epochs, require_empty=False, iteration=self.args.epoch_c)

        monitoring_metrics = ['g_loss', 'd_loss', 'wdistance', 'guiding_loss', 'lr']
        RunningAverage(alpha=0.98, output_transform=lambda x: x['g_loss']).attach(trainer, 'g_loss')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['d_loss']).attach(trainer, 'd_loss')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['wdistance']).attach(trainer, 'wdistance')
        RunningAverage(alpha=0.98, output_transform=lambda x: x['guiding_loss']).attach(trainer, 'guiding_loss')
        RunningAverage(alpha=0.01, output_transform=lambda x: x['lr']).attach(trainer, 'lr')

        self.pbar = ProgressBar()
        self.pbar.attach(trainer, metric_names=monitoring_metrics)

        trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                  to_save={"discriminator": self.discriminator,
                                           "generator": self.generator,
                                           "optimizer_D": self.optimizer['discriminator'],
                                           "optimizer_G": self.optimizer['generator'],
                                           })

        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def evaluator(engine):
            acc = AverageMeter()
            iou = AverageMeter()
            generator.eval()
            discriminator.eval()
            self.evaluator.reset()
            ssim = np.array([])
            directory = os.path.dirname(self.args.output_dir + "/examples/epoch_{}/".format(engine.state.epoch))
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.dirname(
                self.args.output_dir + "/examples/epoch_{}/validation/".format(engine.state.epoch))
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i, (rgb, rgbseg, lidarseg, rgb_labels, proj_labels, proj_mask, proj, rgb_labels_one_hot,
                    proj_labels_one_hot) in enumerate(parser.validloader):
                proj = proj.cuda(non_blocking=True)

                proj_labels_one_hot = proj_labels_one_hot.cuda(non_blocking=True)
                rgb_labels = rgb_labels.cuda(non_blocking=True).long()
                generated_img = self.generator(proj, proj_labels_one_hot)
                generated_argmax = generated_img.argmax(dim=1, keepdim=True)
                self.evaluator.addBatch(generated_argmax[0], rgb_labels)
                generated_argmax = colorize_mask(generated_argmax[0][0].cpu().numpy()).convert(mode='RGB')
                rgbseg = transforms.ToPILImage()(rgbseg[0][0]).convert(mode='RGB')
                generated_argmax.save(
                    self.args.output_dir + "/examples/epoch_{}/validation/generated_camseg_valid_{}.png".format(
                        engine.state.epoch, i))
                ssim = np.append(ssim, compare_ssim(rgbseg, generated_argmax))



            accuracy = self.evaluator.getacc()
            jaccard, class_jaccard = self.evaluator.getIoU()
            acc.update(accuracy.item(), rgb.size(0))
            iou.update(jaccard.item(), rgb.size(0))
            print("Mean SSIM:", np.mean(ssim))
            print("mIoU:", iou.avg)
            print("Acc:", accuracy)
            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=i, jacc=jacc))
            self.info['ssim'] = np.mean(ssim)
            self.info['miou'] = iou.avg
            self.info['acc'] = acc.avg

            generator.train()
            discriminator.train()

        @trainer.on(Events.EPOCH_COMPLETED)
        def print_times(engine):
            self.pbar.log_message(
                'Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, self.timer.value()))
            self.timer.reset()

        @trainer.on(Events.ITERATION_COMPLETED(every=args.print_freq))
        def print_logs(engine):
            columns = engine.state.metrics.keys()
            values = [str(round(value, 5)) for value in engine.state.metrics.values()]
            self.info['g_loss'] = engine.state.output['g_loss']
            self.info['d_loss'] = engine.state.output['d_loss']
            self.info['guiding_loss'] = engine.state.output['guiding_loss']
            self.info['wdistance'] = engine.state.output['wdistance']
            self.info['lr'] = engine.state.output['lr']

            i = (engine.state.iteration % len(self.trainDataLoader))
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=self.args.epochs,
                                                                  i=i,
                                                                  max_i=len(self.trainDataLoader))
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            self.pbar.log_message(message)

        @trainer.on(Events.EPOCH_COMPLETED)
        def print_times(engine):
            self.pbar.log_message(
                'Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, self.timer.value()))
            self.timer.reset()

        @trainer.on(Events.STARTED)
        def loaded(engine):
            if self.epoch_c != 0:
                engine.state.epoch = self.epoch_c
                engine.state.iteration = self.args.epoch_c * len(self.trainDataLoader)

    def trainer(self):
        return self.t
