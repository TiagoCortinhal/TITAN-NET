import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_workers',
                    type=int, default=10,
                    help='number of data loading workers')

parser.add_argument('--checkpoint_interval',
                    type=int, default=1,
                    help='Checkpoint internal (in epoch)')

parser.add_argument('--batch-size',
                    type=int, default=4,
                    help='input batch size')

parser.add_argument('--arch_cfg', '-ac',
                    type=str,
                    required=True,
                    help='Architecture yaml cfg file. See /config/arch for sample. No default!')

parser.add_argument('--data_cfg', '-dc',
                    type=str,
                    required=False,
                    default='config/labels/semantic-kitti.yaml',
                    help='Classification yaml cfg file. See /config/labels for sample. No default!')

parser.add_argument('--epochs',
                    type=int, default=300,
                    help='number of epochs to train for')

parser.add_argument('--n_classes',
                    type=int, default=20,
                    help='number of classes')

parser.add_argument('--learning_rate',
                    type=float, default=1e-4,
                    help='Learning Rate')

parser.add_argument('--dataset', '-d',
                    type=str,
                    required=True,
                    help='Dataset to train with. No Default')

parser.add_argument('--cuda',
                    type=str,
                    help='GPUs to use. Ex. 1,3,6')

parser.add_argument('--lidar_pretrained',
                    type=str,
                    required=False,
                    help='path to pretrained lidar segmentation network')

parser.add_argument('--rgb_pretrained',
                    type=str,
                    required=False,
                    help='path to pretrained rgb segmentation network')

parser.add_argument('--output-dir',
                    default='./out/v2/',
                    help='directory to output images and model checkpoints')

parser.add_argument('--seed',
                    type=int, default=23,
                    help='manual seed')

parser.add_argument('--print_freq',
                    type=int, default=50,
                    help='output frequency')

parser.add_argument('--eval', action='store_true')

parser.add_argument('--eval-epoch', type=int, default=0)

parser.add_argument('--theta', type=int, default=1)

parser.add_argument('--gamma', type=int, default=10)

parser.add_argument('--epoch-c', type=int, default=0)

args = parser.parse_args()
