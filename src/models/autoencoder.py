from _imports import *
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, criterion=nn.BCELoss(),
    			optimizer = optim.Adam):
    	super().__init__()
    	self.criterion = criterion
    	self.optimizer = optimizer
    	self.encoder = None
    	self.decoder = None
    	self.training_losses = []
    	self.validation_losses = []

    def forward(self, x):
        pass
    def get_feature_map(self, x):
        pass
    def decode(self, x):
        pass

    def fit_encoder(self, tdl, vdl=None,
    				num_epochs=1, lr = 1e-6, device = 'cpu', **kwargs):
    	#Training autoencoder based on feature map.

        pb = tqdm(range(num_epochs))
        self.optim = self.optimizer([*self.encoder.parameters(),*self.decoder.parameters()],
        							lr=lr, **kwargs)
        for e in pb:
            self.train()
            for x, _ in tdl:
                x = x.to(device)
                optim.zero_grad()
                y = self.get_feature_map(x)
                yhat = self.decode(x)
                loss = self.criterion(yhat,y)
                loss.backward()
                optim.step()
                pb.set_description(f"Loss: {round(loss.item(),2)}")
            
            self.evaluate(tdl, training=True)
            self.evaluate(vdl, validation=True)
            
    def evaluate(self, dl, training=False, validation=False):
        self.eval()
        with torch.no_grad():
            for x, y in dl:
                x,y = x.cuda(), y.cuda()
                y = self.get_feature_map(x)
                yhat = self._forward(x)
                loss = self.criterion(yhat,y)
                
                if validation: self.validation_losses.append(loss.item())
                if training: self.training_losses.append(loss.item())