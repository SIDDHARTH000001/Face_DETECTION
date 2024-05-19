import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchvision import models


class EfficientNetEmbedding(nn.Module):
    def __init__(self,input_shape, num_layers_to_unfreeze = 0):
        super(EfficientNetEmbedding, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        # base_model = nn.Sequential(*list(base_model.children())[:-3])
        base_model.classifier = nn.Identity()
        base_model.input_shape = (3,input_shape,input_shape)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

        # layers = list(base_model.modules())
        # for i in range(len(layers)-num_layers_to_unfreeze):
        #     layers[i].required_grad = False

        self.base_model = base_model
        self.flatten = nn.Flatten()
        self.embedding = nn.Sequential(
            self.base_model,
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        emb = self.embedding(x)
        return emb







class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor, positive, negative):
        ap_similarity = F.cosine_similarity(anchor, positive, dim=-1)
        an_similarity = F.cosine_similarity(anchor, negative, dim=-1)
        return ap_similarity, an_similarity

class SiameseNet(nn.Module):
    def __init__(self, embedding_net, input_shape):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.distance_layer = DistanceLayer()

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)
        return self.distance_layer(anchor_embedding, positive_embedding, negative_embedding)


class SiameseModel(nn.Module):

    def __init__(self, siamese_net, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.loss_tracker = MeanMetric()
        self.accuracy_tracker = MeanMetric()


    def forward(self, anchor, positive, negative):
        ap_similarity, an_similarity = self.siamese_net( anchor, positive, negative)
        return ap_similarity, an_similarity

    def training_step(self, anchor, positive, negative):
        ap_similarity, an_similarity = self.siamese_net(anchor, positive, negative)
        loss = torch.max(torch.tensor(0.0), self.margin - ap_similarity + an_similarity)
        return loss.mean()
        # return 1 - ap_similarity + an_similarity
        # losses = F.relu(ap_similarity - an_similarity + self.margin).mean()
        #
        # l2_regularization = 0
        # l1_regularization = 0
        # for param in self.siamese_net.parameters():
        #     l2_regularization += torch.norm(param, p=2)**2
        #     l1_regularization += torch.norm(param, p=1)

        #
        # lambda_l1 = 0.0001  # adjust L1 regularization strength
        # lambda_l2 = 0.0001  # adjust L2 regularization strength
        # total_loss = loss + lambda_l1 * l1_regularization + lambda_l2 * l2_regularization
        #
        # return total_loss


    def validation_step(self, anchor, positive, negative):
        ap_similarity, an_similarity = self.siamese_net(anchor, positive, negative)
        loss = torch.max(torch.tensor(0.0), self.margin - ap_similarity + an_similarity)
        return loss.mean()
        losses = F.relu(ap_similarity - an_similarity + self.margin).mean()

        l2_regularization = 0
        l1_regularization = 0
        for param in self.siamese_net.parameters():
            l2_regularization += torch.norm(param, p=2)**2
            l1_regularization += torch.norm(param, p=1)


        # Total loss including regularization
        lambda_l1 = 0.0001  # adjust L1 regularization strength
        lambda_l2 = 0.0001  # adjust L2 regularization strength
        total_loss = losses + lambda_l1 * l1_regularization + lambda_l2 * l2_regularization

        return total_loss


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        ap_similarity, an_similarity = self.siamese_net(*inputs)

        loss = self.compute_loss(ap_similarity, an_similarity)

        self.log('test_loss', loss)
        return loss

    def compute_loss(self, ap_similarity, an_similarity):
        loss = F.relu(ap_similarity - an_similarity + self.margin)
        return loss.mean()

    def compute_accuracy(self, ap_similarity, an_similarity):
        accuracy = (ap_similarity < an_similarity).float().mean()
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



