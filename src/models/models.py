import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.models import resnet18, ResNet18_Weights

from typing import List
import math

from avalanche.models import FeatureExtractorModel
from avalanche.models.dynamic_modules import MultiHeadClassifier,\
                    MultiTaskModule, IncrementalClassifier

from avalanche.models.bic_model import BiasLayer

'''
Layer Definitions
'''
class L2NormalizeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Flatten
        return torch.nn.functional.normalize(x, p=2, dim=1)
    

class FeatAvgPoolLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """ This format to be compatible with OpenSelfSup and the classifiers expecting a list."""
        # Pool
        assert x.dim() == 4, \
            "Tensor must has 4 dims, got: {}".format(x.dim())
        x = self.avg_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        return x

    

'''
Backbones
'''
class SimpleCNNFeat(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNNFeat, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            #nn.Dropout(p=0.25)
        )

        self.feature_size = self.calc_feature_size(input_size)
    
    def calc_feature_size(self, input_size):
        with torch.no_grad():
            self.feature_size = self.features(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    

class MLPfeat(nn.Module):
    def_hidden_size = 400

    def __init__(self, nonlinear_embedding: bool, input_size=28 * 28,
                 hidden_sizes: tuple = None, nb_layers=2):
        """
        :param nonlinear_embedding: Include non-linearity on last embedding layer.
        This is typically True for Linear classifiers on top. But is false for embedding based algorithms.
        :param input_size:
        :param hidden_size:
        :param nb_layers:
        """
        super().__init__()
        assert nb_layers >= 2
        if hidden_sizes is None:
            hidden_sizes = [self.def_hidden_size] * nb_layers
        else:
            assert len(hidden_sizes) == nb_layers
        self.feature_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes

        # Need at least one non-linear layer
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_sizes[0]),
                                 nn.ReLU(inplace=True)
                                 ))

        for layer_idx in range(1, nb_layers - 1):  # Not first, not last
            layers.add_module(
                f"fc{layer_idx}", nn.Sequential(
                    *(nn.Linear(hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]),
                      nn.ReLU(inplace=True)
                      )))

        # Final layer
        layers.add_module(
            f"fc{nb_layers}", nn.Sequential(
                *(nn.Linear(hidden_sizes[nb_layers - 2],
                            hidden_sizes[nb_layers - 1]),
                  )))

        # Optionally add final nonlinearity
        if nonlinear_embedding:
            layers.add_module(
                f"final_nonlinear", nn.Sequential(
                    *(nn.ReLU(inplace=True),)))

        self.features = nn.Sequential(*layers)
        # self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        # x = self.classifier(x)
        return x
    

#########
#Slim ResNet (Avalanche Copy)
#########
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SlimResnet18Feat(nn.Module):
    def __init__(self, input_size, block, num_blocks, nf, global_pooling=False):
        super(SlimResnet18Feat, self).__init__()
        self.in_planes = nf
        self.global_pooling = global_pooling

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        #self.linear = nn.Linear(nf * 8 * block.expansion, 100) # num_classes

        self.feature_size = self.calc_feature_size(input_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def calc_feature_size(self, input_size):
        with torch.no_grad():
            self.feature_size = self.forward(torch.zeros(1, *input_size)).view(1, -1).size(1)
        return self.feature_size

    def forward(self, x):
        #bsz = x.size(0)
        #out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.global_pooling:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out


#########
# ResNet (PyTorch)
######### 
# TODO
    

'''
Classifier
'''
class BiCClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = torch.nn.Linear(in_features=in_features, 
                                    out_features=num_classes)  
        self.bias_layer = None  # NOTE: will be set by bic plugin

    def forward(self, x):
        x = self.classifier(x)
        
        if not self.training and self.bias_layer is not None:
            x = self.bias_layer(x)
        return x



'''
Model Factory
'''
def _get_feat_extr(args, input_size):
    """ Get embedding network. """
    if args.backbone == "mlp":  # MNIST MLP
        nonlin_embedding = args.classifier in ['linear']
        feat_extr = MLPfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, 
                            input_size=math.prod(input_size))
    elif args.backbone == 'simple_cnn':
        feat_extr = SimpleCNNFeat(input_size=input_size)
    elif args.backbone == 'SlimResNet18':
        feat_extr = SlimResnet18Feat(input_size=input_size,
                                     block=BasicBlock, 
                                     num_blocks=[2, 2, 2, 2],
                                     nf=20,
                                     global_pooling=args.use_GAP)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")
    return feat_extr

def _get_classifier(classifier_type: str, n_classes: int, feat_size: int, 
                    initial_out_features: int, task_incr: bool, 
                    lin_bias: bool = True): 
    """ 
    Get classifier head. For embedding networks this is normalization or identity layer.
    
    feat_size: Input to the linear layer
    initial_out_features: (Initial) output of the linear layer. Potenitally growing with task.
    """
    # No prototypes, final linear layer for classification
    if classifier_type == 'linear':  # Linear layer
        print("_get_classifier::feat_size: ", feat_size)
        if task_incr:
            classifier = MultiHeadClassifier(in_features=feat_size,
                                    initial_out_features=initial_out_features,
                                    )#use_bias=lin_bias # NOTE: False by default
        else:
            classifier = torch.nn.Linear(in_features=feat_size, 
                                    out_features=n_classes, 
                                    ) #bias=lin_bias  
    # Prototypes held in strategy
    elif classifier_type == 'norm_embed':  # Get feature normalization
        classifier = L2NormalizeLayer()
    elif classifier_type == 'identity':  # Just extract embedding output
        classifier = torch.nn.Flatten()
    else:
        raise NotImplementedError()
    return classifier




def get_model(args, n_classes, input_size, initial_out_features, 
              backbone_weights=None, model_weights=None):
    """ 
    Build model from feature extractor and classifier.
    
    n_classes: total number of classes in the dataset
    initial_out_features: number of classes for the first task
    """

    feat_extr = _get_feat_extr(args, input_size=input_size)  # Feature extractor
    classifier = _get_classifier(args.classifier, 
                                 n_classes, 
                                 feat_extr.feature_size, 
                                 initial_out_features, 
                                 task_incr=args.task_incr,
                                 lin_bias=args.lin_bias)  # Classifier
    
    # Load weights for only the feature extractor (backbone)
    if backbone_weights:
        print("Loading backbone weights from: ", backbone_weights)
        state_dict = torch.load(backbone_weights)
        for key in list(state_dict.keys()):
            new_key = key.replace("encoder", "features")
            new_key = key.replace("feature_extractor.", "")
            state_dict[new_key] = state_dict.pop(key)
        feat_extr.load_state_dict(state_dict, strict=False)


    model = FeatureExtractorModel(
        feature_extractor=feat_extr,
        train_classifier=classifier
    )

    # Load weights for the entire model (backbone + heads)
    if model_weights:
        state_dict = torch.load(model_weights)
        print("state_dict")
        for key in state_dict:
            print(key)
        print("Loading pretrained model weights from: ", model_weights)
        model.load_state_dict(torch.load(model_weights), strict=True)

    return model
