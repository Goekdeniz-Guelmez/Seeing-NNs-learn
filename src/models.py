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
    """
    Neural network that augments input with Fourier features before processing through a SkipConn network.
    Adds sin(nx) + cos(nx) features for n=1...fourier_order.
    
    Parameters:
        fourier_order (int): Number of Fourier features to generate (default: 4)
        hidden_size (int): Number of neurons per hidden layer in SkipConn (default: 100)
        num_hidden_layers (int): Number of hidden layers in SkipConn (default: 7)
        dropout_rate (float): Dropout probability (default: 0.2)
        linmap (object, optional): Linear mapping transform for input data
    """
    def __init__(
			self,
			fourier_order: int = 4,
			hidden_size: int = 100,
			num_hidden_layers: int = 7, 
			dropout_rate: float = 0.2,
			linmap=None
		):
        super(Fourier, self).__init__()
        
        # Configuration
        self.fourier_order = fourier_order
        self._linmap = linmap
        
        # Calculate input size for inner model (2 features per order (sin+cos) * fourier_order + original 2D input)
        inner_input_size = fourier_order * 4 + 2
        
        # Initialize inner SkipConn model with dropout
        self.inner_model = SkipConn(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            init_size=inner_input_size,
            dropout_rate=dropout_rate,
            linmap=None  # We handle mapping here
        )
        
        # Register orders as buffer to move with model to correct device
        self.register_buffer('orders', torch.arange(1, fourier_order + 1, dtype=torch.float))

    def forward(self, x):
        """
        Forward pass computing Fourier features and processing through SkipConn.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Output tensor mapped to range [0,1]
        """
        # Apply optional linear mapping
        if self._linmap is not None:
            x = self._linmap.map(x)
        
        # Add dimension for broadcasting with orders
        x = x.unsqueeze(-1)  # Shape: (batch_size, 2, 1)
        
        # Compute Fourier features
        fourier_features = torch.cat([
            torch.sin(self.orders * x),  # sin(nx)
            torch.cos(self.orders * x),  # cos(nx)
            x  # original input
        ], dim=-1)
        
        # Flatten features
        fourier_features = fourier_features.reshape(x.shape[0], -1)
        
        # Process through inner model
        return self.inner_model(fourier_features)


class Fourier2D(nn.Module):
    """
    Neural network that augments 2D input with 2D Fourier features before processing through a SkipConn network.
    Computes products of sin/cos features for both dimensions.
    
    Parameters:
        fourier_order (int): Number of Fourier features per dimension (default: 4)
        hidden_size (int): Number of neurons per hidden layer in SkipConn (default: 100)
        num_hidden_layers (int): Number of hidden layers in SkipConn (default: 7)
        dropout_rate (float): Dropout probability (default: 0.2)
        linmap (object, optional): Linear mapping transform for input data
    """
    def __init__(
			self,
			fourier_order: int = 4,
			hidden_size: int = 100,
			num_hidden_layers: int = 7, 
			dropout_rate: float = 0.2,
			linmap=None
		):
        super(Fourier2D, self).__init__()
        
        # Configuration
        self.fourier_order = fourier_order
        self._linmap = linmap
        
        # Calculate input size for inner model (4 features per order combination + original 2D input)
        inner_input_size = (fourier_order * fourier_order * 4) + 2
        
        # Initialize inner SkipConn model with dropout
        self.inner_model = SkipConn(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            init_size=inner_input_size,
            dropout_rate=dropout_rate,
            linmap=None  # We handle mapping here
        )
        
        # Register orders as buffer to move with model to correct device
        self.register_buffer('orders', torch.arange(0, fourier_order, dtype=torch.float))

    def compute_2d_fourier_features(self, x):
        """
        Compute 2D Fourier features as products of sin/cos terms.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Fourier features tensor
        """
        features = [x]  # Start with original input
        
        # Compute product terms for all order combinations
        for n in self.orders:
            for m in self.orders:
                # Compute trig products using broadcasting
                cos_n_x = torch.cos(n * x[:, 0])
                sin_n_x = torch.sin(n * x[:, 0])
                cos_m_y = torch.cos(m * x[:, 1])
                sin_m_y = torch.sin(m * x[:, 1])
                
                # Add all product combinations
                features.extend([
                    (cos_n_x * cos_m_y).unsqueeze(-1),  # cos(nx)cos(my)
                    (cos_n_x * sin_m_y).unsqueeze(-1),  # cos(nx)sin(my)
                    (sin_n_x * cos_m_y).unsqueeze(-1),  # sin(nx)cos(my)
                    (sin_n_x * sin_m_y).unsqueeze(-1)   # sin(nx)sin(my)
                ])
        
        return torch.cat(features, dim=1)

    def forward(self, x):
        """
        Forward pass computing 2D Fourier features and processing through SkipConn.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Output tensor mapped to range [0,1]
        """
        # Apply optional linear mapping
        if self._linmap is not None:
            x = self._linmap.map(x)
        
        # Compute 2D Fourier features
        fourier_features = self.compute_2d_fourier_features(x)
        
        # Process through inner model
        return self.inner_model(fourier_features)


import torch

class CenteredLinearMap:
    """
    A utility class that performs centered linear mapping of 2D coordinates.
    Maps input coordinates from one range to another while centering the transformation.
    
    Parameters:
        xmin (float): Minimum x-coordinate of input range (default: -2.5)
        xmax (float): Maximum x-coordinate of input range (default: 1.0)
        ymin (float): Minimum y-coordinate of input range (default: -1.1)
        ymax (float): Maximum y-coordinate of input range (default: 1.1)
        x_size (float, optional): Target x range size. If None, maintains scale (default: None)
        y_size (float, optional): Target y range size. If None, maintains scale (default: None)
    """
    def __init__(
			self,
			xmin: float = -2.5,
			xmax: float = 1.0,
			ymin: float = -1.1,
			ymax: float = 1.1,
			x_size=None,
			y_size=None
		):
        # Validate input ranges
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Max values must be greater than min values")
            
        # Calculate x scaling factor
        if x_size is not None:
            if x_size <= 0:
                raise ValueError("x_size must be positive")
            x_scale = x_size / (xmax - xmin)
        else:
            x_scale = 1.0
            
        # Calculate y scaling factor
        if y_size is not None:
            if y_size <= 0:
                raise ValueError("y_size must be positive")
            y_scale = y_size / (ymax - ymin)
        else:
            y_scale = 1.0
            
        # Calculate centering offsets
        x_offset = -(xmin + xmax) * x_scale / 2
        y_offset = -(ymin + ymax) * y_scale / 2
        
        # Store transformation parameters as tensors
        self.scale = torch.tensor([x_scale, y_scale], dtype=torch.float)
        self.offset = torch.tensor([x_offset, y_offset], dtype=torch.float)
        
        # Store original parameters for reference
        self.input_range = {
            'x': (xmin, xmax),
            'y': (ymin, ymax)
        }
        self.target_size = {
            'x': x_size,
            'y': y_size
        }

    def map(self, x):
        """
        Apply the centered linear transformation to input coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Transformed coordinates tensor of same shape as input
        """
        # Move transformation parameters to same device as input
        scale = self.scale.to(x.device)
        offset = self.offset.to(x.device)
        
        # Apply linear transformation: scale * x + offset
        return scale * x + offset
    
    def inverse_map(self, x):
        """
        Apply the inverse transformation to mapped coordinates.
        
        Args:
            x (torch.Tensor): Transformed coordinates tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Original coordinates tensor of same shape as input
        """
        # Move transformation parameters to same device as input
        scale = self.scale.to(x.device)
        offset = self.offset.to(x.device)
        
        # Apply inverse transformation: (x - offset) / scale
        return (x - offset) / scale
    
    @property
    def transformation_params(self):
        """
        Get the current transformation parameters.
        
        Returns:
            dict: Dictionary containing scale and offset tensors
        """
        return {
            'scale': self.scale,
            'offset': self.offset
        }
        
    def __repr__(self):
        """
        String representation of the transformation.
        
        Returns:
            str: Formatted string describing the transformation
        """
        return (f"CenteredLinearMap(\n"
                f"  Input range x: [{self.input_range['x'][0]}, {self.input_range['x'][1]}]\n"
                f"  Input range y: [{self.input_range['y'][0]}, {self.input_range['y'][1]}]\n"
                f"  Target size x: {self.target_size['x']}\n"
                f"  Target size y: {self.target_size['y']}\n"
                f"  Scale: {self.scale.tolist()}\n"
                f"  Offset: {self.offset.tolist()}\n)")
    
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

