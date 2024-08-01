import torch
import torch.nn
from torchvision.models import vgg19

class ImplicitLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features.cuda()
        # self.feature_extractor = torch.nn.Sequential(*list(vgg.features)[:35]).cuda()
        #self.feature_extractor.eval()

        # Define the downsampling layer
        self.pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2))

    @property
    def name(self,):
        return "implicit"

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0':'conv1_1',
                      '5':'conv2_1',
                      '10':'conv3_1',
                      '19':'conv4_1',
                      '21':'conv4_2', # content repr
                      '28':'conv5_1',}

        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
            
        return features

    def to_rgb(self, x):
        x_mono = x.mean(dim=-1, keepdim=True)  # Take the mean along the channel dimension to get a single-channel image
        
        x_rgb = x_mono.repeat(1, 1, 1, 3)  # Repeat the single-channel image along the channel dimension to get a three-channel image
        
        x_rgb = x_rgb.permute(0, 3, 1, 2)
        # Apply the pooling operation to downsample the data
        x_rgb = self.pool(x_rgb)

        return x_rgb

    def gram_matrix(self, tensor):
        n, d, h, w = tensor.size()
        print(tensor.size())
        tensor = tensor.view(d, n*h*w)
        return torch.mm(tensor, tensor.t()) #gram
    
    def forward(self, x, y):

        # Normalize along the batch dimension
        # x = x / torch.max(torch.abs(x), dim=0, keepdim=True).values
        # y = y / torch.max(torch.abs(y), dim=0, keepdim=True).values

        # Compute the feature maps
        x = self.to_rgb(x)
        y = self.to_rgb(y)

        # get features
        x_features = self.get_features(x, self.vgg)
        y_features = self.get_features(y, self.vgg)

        # calculate content loss
        content_loss = torch.mean((x_features['conv2_1'] - y_features['conv2_1'])**2)
        # # calculate grams
        # style_grams = {layer:self.gram_matrix(y_features[layer]) for layer in y_features}

        # # weights for style layers 
        # style_weights = {'conv1_1':1.,
        #                 'conv2_1':1.,
        #                 'conv3_1':1.,
        #                 'conv4_1':1.,
        #                 'conv5_1':1.}
		# # calculate style loss
        # style_loss = 0

        # for layer in style_weights:
        #     target_feature = x_features[layer]
        #     _, d, h, w = target_feature.shape

        #     # target gram
        #     target_gram = self.gram_matrix(target_feature)

        #     # style gram
        #     style_gram = style_grams[layer]

        #     # style loss for curr layer
        #     layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        #     style_loss += layer_style_loss / (d * h * w)

        return content_loss#+style_loss