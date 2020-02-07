from _imports import *
from autoencoder import AutoEncoder
from _util import nop, linear
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

class Efficient_AutoEncoder(AutoEncoder):
    def __init__(self, nh, version = "efficientnet-b0", advprop = False, use_pool=False):
        super().__init__()
        self.conv = EfficientNet.from_pretrained(version, advprop=advprop)
        self.pool = nn.AdaptiveMaxPool2d(1) if use_pool else nop()
        self.encoder = nn.Sequential(linear(nh,512),
                                     linear(512,256),
                                     linear(256, 64))
        self.decoder = nn.Sequential(linear(64, 256),
                                     linear(256,512),
                                     linear(512,nh))

    def forward(self, x):
        fm = torch.flatten(self.pool(self.conv.extract_features(x)), start_dim=1)
        enc = self.encoder(fm)
        return enc

    def get_feature_map(self, x):
        with torch.no_grad():
            fm= self.conv.extract_features(x)
        return torch.flatten(fm,start_dim=1)

    def decode(self, x):
        fm = self.get_feature_map(x)
        enc = self.encoder(torch.flatten(self.pool(fm), start_dim=1))
        dec = self.decoder(enc)
        return torch.sigmoid(dec)

if __name__ == "__main__":
    from torchvision.datasets import CIFAR10 as load_data
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    # Test run on CIFAR 10

    nh = 1280*4*4
    model = Efficient_AutoEncoder(1280*4*4).cuda()

    convert = transforms.Compose([transforms.Pad((128-32)//2),transforms.ToTensor()])
    path = load_data(root='./data', train=True, download=True, transform=convert)
    tdl = torch.utils.data.DataLoader(path, batch_size=128, shuffle=True)
    path = load_data(root='./data', train=False, download=True, transform=convert)
    vdl = torch.utils.data.DataLoader(path, batch_size=128, shuffle=True)
    
    model.fit_encoder(tdl, vdl, num_epochs=10, lr=1e-3, device = 'cuda')

    plt.plot(model.training_losses)
    plt.show()
    
    plt.plot(model.validation_losses)
    plt.show()

    import pickle

    with open("./trained_models/efficient_ae_cifar10.pkl",'wb') as f:
        pickle.dump(model, f)
