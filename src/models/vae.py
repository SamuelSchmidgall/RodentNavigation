from _imports import *
import torch.optim as optim
import torch.nn.functional as F
from autoencoder import AutoEncoder

def ELBO(yhat, y, mu, var):
	pass

class VAE(AutoEncoder):
    def __init__(self):
    	super().__init__(criterion=ELBO)

    def reparameterize(self, mu, var):
    	return mu + torch.exp(0.5*var) * torch.normal(0,1)

    def fit_encoder(self, tdl, vdl=None,
    				num_epochs=1, lr = 1e-6, device = 'cpu', **kwargs):
    	#Training vae based on feature map.

        pb = tqdm(range(num_epochs))
        self.optim = self.optimizer([*self.encoder.parameters(),*self.decoder.parameters()],
        							lr=lr, **kwargs)
        for e in pb:
            self.train()
            for x, _ in tdl:
                x = x.to(device)
                optim.zero_grad()
                y = self.get_feature_map(x)
                mu, var = self.encode(x)
                z = self.reparameterize(mu,var)
                yhat = self.decode(z)
                loss = self.criterion(yhat,y)
                loss.backward()
                optim.step()
                pb.set_description(f"Loss: {round(loss.item(),2)}")
            
            self.evaluate(tdl, training=True)
            self.evaluate(vdl, validation=True)
