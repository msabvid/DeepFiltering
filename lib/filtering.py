import torch
import torch.nn as nn
import signatory
import torchcde
#from sklearn.linear_model import LinearRegression
from torch import optim
from tqdm import tqdm

from lib.nrde import NeuralCDE
from lib.networks import LinearRegression, FFN
from lib.utils import to_numpy, toggle






def augment_with_time(t, *args):
    """
    Augment all the paths in args with time
    """
    for x in args:
        ts = t.reshape(1,-1,1).repeat(x.shape[0],1,1)
        yield torch.cat([ts, x],2)



class Filter(nn.Module):
    
    def __init__(self, depth: int, x_real_obs: torch.Tensor, x_real_state: torch.Tensor, t: torch.Tensor, window_length: int):
        super().__init__()

        self.depth = depth
        self.x_real_obs = x_real_obs
        self.x_real_state = x_real_state
        self.t = t
        self.window_length = window_length
        
        # 1. Neural RDE to model (X,Y)
        logsig_channels = signatory.logsignature_channels(in_channels=x_real_obs.shape[-1]+1, depth=depth) # +1 because of time
        self.nrde = NeuralCDE(input_channels=logsig_channels, hidden_channels=8, output_channels=1, interpolation="linear")
        
        self.loss_xy = []
       

    
    def _filtering(self, x_real_obs: torch.Tensor):
        x_real_obs_t, = augment_with_time(self.t, x_real_obs)
        obs_logsig = torchcde.logsig_windows(x_real_obs_t, self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)
        _, pred = self.nrde(obs_coeffs, t='grid_points')
        return pred
    
    def _train_filtering(self, num_epochs: int, batch_size: int):
        print("Training Neural RDE to model (X,Y)")
        x_real_obs_t, = augment_with_time(self.t, self.x_real_obs)
        obs_logsig = torchcde.logsig_windows(x_real_obs_t, self.depth, window_length=self.window_length)
        obs_coeffs = torchcde.linear_interpolation_coeffs(obs_logsig)

        train_dataset = torch.utils.data.TensorDataset(obs_coeffs, self.x_real_state)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)

        optimizer = torch.optim.Adam(self.nrde.parameters(), lr=0.002)
        loss_fn = nn.MSELoss()

        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            for i, (batch_coeffs, batch_y) in enumerate(train_dataloader):
                pbar.write("batch {} of {}".format(i, len(train_dataloader)))
                optimizer.zero_grad()
                _, pred = self.nrde(batch_coeffs, t='grid_points')
                loss = loss_fn(pred, batch_y[:,::self.window_length])
                loss.backward()
                optimizer.step()
                self.loss_xy.append(loss.item())
            pbar.update(1)
            pbar.write("loss:{:.4f}".format(loss.item()))


    def fit(self, num_epochs, mc_samples: int):
        batch_size = 400

        # 1. Solving Filtering problem
        self._train_filtering(num_epochs=num_epochs, batch_size=batch_size)

        return 0
                

