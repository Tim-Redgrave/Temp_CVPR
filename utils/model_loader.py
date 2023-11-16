import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
import torchvision.transforms as transforms

class ResNet50_with_preprocessing(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for key,module in base_model._modules.items():
            self.add_module(key,module)

    def forward(self,x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class InceptionV3_with_preprocessing(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for key,module in base_model._modules.items():
            self.add_module(key,module)

    def forward(self,x):
        x = self.preprocess(x)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseNet_with_preprocessing(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for key,module in base_model._modules.items():
            self.add_module(key,module)

    def forward(self,x):
        x = self.preprocess(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def load_model(model_architecture,pretraining,num_classes,weights_path=None):
    match model_architecture:
        case "resnet":
            model = torchvision_models.resnet50(weights=pretraining)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
            model = ResNet50_with_preprocessing(model)
            if weights_path is not None and os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path)['state_dict'])
            return model
        case "inception":
            model = torchvision_models.inception_v3(weights=pretraining)
            model.aux_logits = False
            model.AuxLogits = None
            model.name = "inception"
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            model = InceptionV3_with_preprocessing(model)
            if weights_path is not None and os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path)['state_dict'])
            return model
        case "densenet":
            model = torchvision_models.densenet121(weights=pretraining)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            model = DenseNet_with_preprocessing(model)
            if weights_path is not None and os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path)['state_dict'])
            return model
        case _:
            raise ValueError("Invalid Architecture Choice")
