"""ff_model.py: NN model implementing the forward-forward algorithm."""

import torch
import torch.nn as nn
import math
from src import utils

class FFModel(nn.Module):
    def __init__(self, opt):
        super(FFModel, self).__init__()
        self.opt = opt  # opt contains all the model configuration
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers   # hidden dimensions

        # Define the model
        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        self.ff_loss = nn.BCEWithLogitsLoss()  # loss function

        self.act_fn = ReLU_full_grad()  # activation function

        # Initialize the downstream classification model
        channels_for_classification = sum(
            [self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)]
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification, self.opt.data.num_classes, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        self._init_weights()  # initialize weights


    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(m.weight.shape[0]))
                nn.init.zeros_(m.bias)
        
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
        
    def _normalize_layer(self, z, tol=1e-8):
        """Normalize the layer. Does so by dividing by the L2 norm of the layer."""
        return z / (torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True)) + tol)
    
    def _compute_ff_loss(self, z, y):
        """Compute the forward-forward loss."""
        sum_squares = torch.sum(z ** 2, dim=1)
        logits = sum_squares - z.shape[1]  # logits are the sum of the squares minus the dimensionality
        ff_loss = self.ff_loss(logits, y.float())  # compute the loss

        with torch.no_grad():
            ff_accuracy = torch.sum((torch.sigmoid(logits) > 0.5) == y).float() / z.shape[0]
        
        return ff_loss, ff_accuracy.item()

    def forward(self, x, y):
        outputs = {
            'loss': torch.zeros(1, device=self.opt.device)
        }
        
        # Concatenate the positive and negative images
        z = torch.cat([x['pos_images'], x['neg_images']], dim=0)  
        # Create labels for the positive and negative images
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)  # flatten the images
        z = self._normalize_layer(z)  # normalize the layer

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            ff_loss, ff_accuracy = self._compute_ff_loss(z, posneg_labels)
            outputs[f'loss_layer_{idx}'] = ff_loss
            outputs[f'accuracy_layer_{idx}'] = ff_accuracy
            outputs['loss'] += ff_loss

            z.detach()  # detach the tensor to avoid backpropagating through the normalization
            z = self._normalize_layer(z)  # normalize the layer before the next layer
        
        outputs = self.forward_downstream_classification_model(x, y, outputs)

        return outputs
    
    def forward_downstream_classification_model(self, x, y, outputs=None):
        if outputs is None:
            outputs = {
                'loss': torch.zeros(1, device=self.opt.device)
            }
        
        z = x['neutral_sample']
        z = z.reshape(z.shape[0], -1)
        z = self._normalize_layer(z)

        classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._normalize_layer(z)
                
                if idx >= 1:
                    classification_model.append(z)
        
        classification_model = torch.cat(classification_model, dim=-1)

        output = self.linear_classifier(classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, y['class_labels'])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, y['class_labels']
        )
        outputs['loss'] += classification_loss
        outputs['accuracy'] = classification_accuracy
        outputs['classification_loss'] = classification_loss

        return outputs
                


class ReLU_full_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()