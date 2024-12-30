import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class Simple(nn.Module):
    """
    A simple linear PyTorch model with batch normalization and dropout.
    Uses ReLU activation in hidden layers and maps final output to [0,1] range
    using tanh transformation.

    Parameters:
        hidden_size (int): Number of neurons per hidden layer
        num_hidden_layers (int): Number of hidden layers in the network
        init_size (int): Input feature dimension
        dropout_rate (float): Dropout probability (default: 0.2)
    """
    def __init__(
			self,
			hidden_size: int = 100,
			num_hidden_layers: int = 7,
			init_size: int = 2,
			dropout_rate: float = 0.2
		):
        super(Simple, self).__init__()
        
        # Initialize layer containers
        self.layers = nn.ModuleList()
        
        # Input layer with batch norm
        self.layers.extend([
            nn.Linear(init_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            self.layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))
        
        # Final activation
        self.tanh = nn.Tanh()
        
        # Create sequential model
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor mapped to range [0,1]
        """
        # Pass through network and map to [0,1] range
        return (self.tanh(self.seq(x)) + 1) / 2


class SkipConn(nn.Module):
    """
    A PyTorch model with skip connections between hidden layers and input concatenation.
    Each hidden layer receives the original input and previous layer outputs via skip connections.
    Uses LeakyReLU activations and maps final output to [0,1] range using tanh transformation.

    Parameters:
        hidden_size (int): Number of non-skip parameters per hidden layer
        num_hidden_layers (int): Number of hidden layers in the network
        init_size (int): Input feature dimension
        dropout_rate (float): Dropout probability (default: 0.2)
        linmap (object, optional): Linear mapping transform for input data
        leaky_slope (float): Negative slope for LeakyReLU (default: 0.01)
    """
    def __init__(
        	self,
			hidden_size: int = 100,
            num_hidden_layers: int = 7,
            init_size: int = 2, 
			dropout_rate: float = 0.2,
			linmap=None,
			leaky_slope: float = 0.01
		):
        super(SkipConn, self).__init__()
        
        # Store configuration
        self.hidden_size = hidden_size
        self.init_size = init_size
        self._linmap = linmap
        
        # Initial layer
        self.inLayer = nn.Sequential(
            nn.Linear(init_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=leaky_slope),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers with skip connections
        self.hidden = nn.ModuleList()
        for i in range(num_hidden_layers):
            # Input size includes current features, previous layer output, and original input
            in_size = hidden_size*2 + init_size if i > 0 else hidden_size + init_size
            
            self.hidden.append(nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(negative_slope=leaky_slope),
                nn.Dropout(dropout_rate)
            ))
        
        # Output layer
        self.outLayer = nn.Linear(hidden_size*2 + init_size, 1)
        
        # Final activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor mapped to range [0,1]
        """
        # Apply optional linear mapping
        if self._linmap is not None:
            x = self._linmap.map(x)
        
        # Initial layer
        cur = self.inLayer(x)
        prev = torch.empty(x.size(0), 0, device=x.device)
        
        # Process through hidden layers with skip connections
        for layer in self.hidden:
            # Combine current features, previous layer output, and original input
            combined = torch.cat([cur, prev, x], dim=1)
            prev = cur
            cur = layer(combined)
        
        # Final layer with skip connections
        y = self.outLayer(torch.cat([cur, prev, x], dim=1))
        
        # Map output to [0,1] range using tanh transformation
        return (self.tanh(y) + 1) / 2



class Fourier(nn.Module):
	def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
		""" 
		Linear torch model that adds Fourier Features to the initial input x as \
		sin(x) + cos(x), sin(2x) + cos(2x), sin(3x) + cos(3x), ...
		These features are then inputted to a SkipConn network.

		Parameters: 
		fourier_order (int): number fourier features to use. Each addition adds 4x\
		 parameters to each layer.
		hidden_size (float): number of non-skip parameters per hidden layer (SkipConn)
		num_hidden_layers (float): number of hidden layers (SkipConn)
		"""
		super(Fourier,self).__init__()
		self.fourier_order = fourier_order
		self.inner_model = SkipConn(hidden_size, num_hidden_layers, fourier_order*4 + 2)
		self._linmap = linmap
		self.orders = torch.arange(1, fourier_order + 1).float().to(device)

	def forward(self,x):
		if self._linmap:
			x = self._linmap.map(x)
		x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
		fourier_features = torch.cat([torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1)
		fourier_features = fourier_features.view(x.shape[0], -1)  # flatten the last two dimensions
		return self.inner_model(fourier_features)


class Fourier2D(nn.Module):
    def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
        super(Fourier2D,self).__init__()
        self.fourier_order = fourier_order
        self.inner_model = SkipConn(hidden_size, num_hidden_layers, (fourier_order*fourier_order*4) + 2)
        self._linmap = linmap
        self.orders = torch.arange(0, fourier_order).float().to(device)

    def forward(self,x):
        if self._linmap:
            x = self._linmap.map(x)
        features = [x]
        for n in self.orders:
            for m in self.orders:
                features.append((torch.cos(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.cos(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
        fourier_features = torch.cat(features, 1)
        return self.inner_model(fourier_features)


class CenteredLinearMap():
	def __init__(self, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1, x_size=None, y_size=None):
		if x_size is not None:
			x_m = x_size/(xmax - xmin)
		else: 
			x_m = 1.
		if y_size is not None:
			y_m = y_size/(ymax - ymin)
		else: 
			y_m = 1.
		x_b = -(xmin + xmax)*x_m/2 - 1 # TODO REMOVE!
		y_b = -(ymin + ymax)*y_m/2
		self.m = torch.tensor([x_m, y_m], dtype=torch.float)
		self.b = torch.tensor([x_b, y_b], dtype=torch.float)


	def map(self, x):
		m = self.m.to(device)
		b = self.b.to(device)
		return m*x + b


# Taylor features, x, x^2, x^3, ...
# surprisingly terrible
class Taylor(nn.Module):
	def __init__(self, taylor_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
		super(Taylor,self).__init__()
		self.taylor_order = taylor_order
		self._linmap = linmap
		self.inner_model = SkipConn(hidden_size, num_hidden_layers, taylor_order*2 + 2)

	def forward(self,x):
		if self._linmap:
			x = self._linmap.map(x)
		series = [x]
		for n in range(1, self.taylor_order+1):
			series.append(x**n)
		taylor = torch.cat(series, 1)
		return self.inner_model(taylor)

