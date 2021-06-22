import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from PIL import Image
import os
import sys
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR100


class calibrateCIL:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                        (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                       (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        for epoch in range(self.epochs):
            scheduler.step()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                ########### cutout #############
                images_rotate = images
                mask_size = 16
                mask_size_half = 8
                size, c, h, w = images.shape
                cxmin, cxmax = mask_size_half, w - mask_size_half
                cymin, cymax = mask_size_half, h - mask_size_half
                for i in range(size):
                    cx = np.random.randint(cxmin, cxmax)
                    cy = np.random.randint(cymin, cymax)
                    xmin = cx - mask_size_half
                    ymin = cy - mask_size_half
                    xmax = xmin + mask_size
                    ymax = ymin + mask_size
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmax)
                    ymax = min(h, ymax)
                    images_rotate[i, :, ymin:ymax, xmin:xmax] = torch.zeros(3, mask_size, mask_size)
                images_rotate = images_rotate.to(self.device)
                ########### cutout based #############

                loss = self._compute_loss(images, target) + self._compute_loss(images_rotate, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target):
        cls_criterion = nn.CrossEntropyLoss()
        output = self.model(imgs)
        if self.old_model is None:
            return cls_criterion(output, target)
        else:
            old_output = self.old_model(imgs)
            old_task_size = old_output.shape[1]
            soft_target = F.softmax(old_output.detach() / 2, dim=1)
            logp = F.log_softmax(output[:, :old_task_size] / 2, dim=1)
            loss_kd = -torch.mean(torch.sum(soft_target * logp, dim=1))

            target = target - old_task_size
            loss2 = cls_criterion(output[:, old_task_size:(old_task_size + 1 * self.task_size)], target)
            alpha = float(old_task_size) / float(old_task_size + 1 * self.task_size)
            return alpha * loss_kd + (1 - alpha) * loss2

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

