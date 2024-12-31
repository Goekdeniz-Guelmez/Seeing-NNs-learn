import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    Neural network that augments input with Taylor series terms (polynomial features)
    before processing through a SkipConn network. Computes terms up to x^n for
    each input dimension.
    
    Parameters:
        taylor_order (int): Highest order of polynomial terms to compute (default: 4)
        hidden_size (int): Number of neurons per hidden layer in SkipConn (default: 100)
        num_hidden_layers (int): Number of hidden layers in SkipConn (default: 7)
        dropout_rate (float): Dropout probability (default: 0.2)
        linmap (object, optional): Linear mapping transform for input data
    """
    def __init__(
			self,
            taylor_order: int = 4,
            hidden_size: int = 100,
            num_hidden_layers: int = 7,
			dropout_rate: float = 0.2,
            linmap=None
		):
        super(Taylor, self).__init__()
        
        # Configuration
        self.taylor_order = taylor_order
        self._linmap = linmap
        
        # Calculate input size for inner model
        # For each input dimension, we have terms: x, x^2, x^3, ..., x^n
        # Plus the original 2D input
        inner_input_size = taylor_order * 2 + 2
        
        # Initialize inner SkipConn model with dropout
        self.inner_model = SkipConn(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            init_size=inner_input_size,
            dropout_rate=dropout_rate,
            linmap=None  # We handle mapping here
        )
        
        # Register powers for computing Taylor terms
        self.register_buffer(
            'powers',
            torch.arange(2, taylor_order + 1, dtype=torch.float)
        )

    def compute_taylor_terms(self, x):
        """
        Compute polynomial terms up to specified order for each input dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Tensor containing polynomial terms up to specified order
        """
        terms = [x]  # Start with linear term
        
        # Pre-compute x^2 once as it's needed for all higher powers
        x_squared = x ** 2
        terms.append(x_squared)
        
        # Use accumulated multiplication for higher powers to avoid numerical issues
        if self.taylor_order > 2:
            curr_power = x_squared
            for power in range(3, self.taylor_order + 1):
                curr_power = curr_power * x  # More stable than x ** power
                terms.append(curr_power)
        
        # Concatenate all terms
        return torch.cat(terms, dim=1)

    def forward(self, x):
        """
        Forward pass computing Taylor series terms and processing through SkipConn.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
            
        Returns:
            torch.Tensor: Output tensor mapped to range [0,1]
        """
        # Apply optional linear mapping
        if self._linmap is not None:
            x = self._linmap.map(x)
        
        # Compute Taylor series terms
        taylor_features = self.compute_taylor_terms(x)
        
        # Process through inner model
        return self.inner_model(taylor_features)
    
    def extra_repr(self) -> str:
        """
        Additional information to be displayed in string representation.
        
        Returns:
            str: String containing model configuration
        """
        return f'taylor_order={self.taylor_order}'
    

class BasicAutoencoder(nn.Module):
    """
    Basic fully-connected autoencoder for image reconstruction.
    
    Parameters:
        input_size (int): Size of flattened input image (width * height * channels)
        hidden_size (int): Size of the bottleneck representation
        dropout_rate (float): Dropout probability
    """
    def __init__(
            self,
            input_size: int = 784,
            hidden_size: int = 128,
            dropout_rate: float = 0.2
		):
        super(BasicAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size * 4, input_size),
            nn.Sigmoid()  # For image pixel values in [0,1]
        )
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Encode then decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape to original dimensions
        return decoded.view(x.size())


class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional autoencoder for image reconstruction with skip connections.
    
    Parameters:
        in_channels (int): Number of input image channels
        base_channels (int): Base number of convolutional channels
        latent_dim (int): Size of the latent representation
        input_size (tuple): Input image dimensions (height, width)
    """
    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 32,
            latent_dim: int = 128,
            input_size: tuple = (28, 28)
		):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Calculate feature map size after encodings
        self.feature_size = (input_size[0] // 4, input_size[1] // 4)
        
        # Encoder
        self.encoder = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Layer 2
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        ])
        
        # Flatten and map to latent space
        self.to_latent = nn.Linear(
            base_channels * 2 * self.feature_size[0] * self.feature_size[1], 
            latent_dim
        )
        
        # Map from latent space to features
        self.from_latent = nn.Linear(
            latent_dim,
            base_channels * 2 * self.feature_size[0] * self.feature_size[1]
        )
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                 kernel_size=2, stride=2),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU()
            ),
            # Layer 2
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 3, base_channels,
                                 kernel_size=2, stride=2),
                nn.BatchNorm2d(base_channels),
                nn.ReLU()
            ),
            # Final layer
            nn.Sequential(
                nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, x):
        # Store skip connections
        skips = []
        
        # Encoding
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skips.append(x)
        
        # Flatten and map to latent space
        x = x.view(x.size(0), -1)
        x = self.to_latent(x)
        
        # Map back to feature space
        x = self.from_latent(x)
        x = x.view(x.size(0), -1, *self.feature_size)
        
        # Decoding with skip connections
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            x = decoder_layer(x)
            skip = skips[-(i+1)]
            x = torch.cat([x, skip], dim=1)
        
        # Final layer without skip connection
        x = self.decoder[-1](x)
        return x


class VariationalAutoencoder(nn.Module):
    """
    Variational autoencoder (VAE) for image reconstruction.
    
    Parameters:
        in_channels (int): Number of input image channels
        base_channels (int): Base number of convolutional channels
        latent_dim (int): Size of the latent representation
        input_size (tuple): Input image dimensions (height, width)
    """
    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 32,
            latent_dim: int = 128,
            input_size: tuple = (28, 28)
		):
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_size = (input_size[0] // 4, input_size[1] // 4)
        feature_dims = base_channels * 4 * self.feature_size[0] * self.feature_size[1]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1
			),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1
			),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1
			),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU()
        )
        
        # Latent space mapping
        self.fc_mu = nn.Linear(feature_dims, latent_dim)
        self.fc_var = nn.Linear(feature_dims, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feature_dims)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2, 
				kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
			),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
				kernel_size=3,
                stride=1,
                padding=1
			),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                base_channels,
                in_channels,
				kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
			),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        # Encode input
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Map from latent space
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, *self.feature_size)
        
        # Decode
        return self.decoder(x)
    
    def forward(self, x):
        # Encode and get latent parameters
        mu, log_var = self.encode(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, log_var)
        
        # Decode
        return self.decode(z), mu, log_var
    

class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.
    
    Parameters:
        img_size (tuple): Input image size (H, W)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(
            self,
            img_size: tuple = (224, 224),
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768
        ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Create patch embeddings
        x = self.proj(x)  # (B, embed_dim, grid_size[0], grid_size[1])
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    
    Parameters:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout rate
        attn_dropout (float): Attention dropout rate
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            dropout: float = 0.,
            attn_dropout: float = 0.
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head self-attention
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image processing.
    
    Parameters:
        img_size (tuple): Input image size (default: (224, 224))
        patch_size (int): Size of image patches (default: 16)
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output channels (default: 3 for RGB reconstruction)
        embed_dim (int): Embedding dimension (default: 768)
        depth (int): Number of transformer blocks (default: 12)
        num_heads (int): Number of attention heads (default: 12)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim (default: 4.0)
        dropout (float): Dropout rate (default: 0.0)
        attn_dropout (float): Attention dropout rate (default: 0.0)
    """
    def __init__(
        self,
        img_size: tuple = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        attn_dropout: float = 0.
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * num_classes)
        
        # Store parameters for reconstruction
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Process output
        x = self.norm(x)
        x = x[:, 1:]  # Remove cls token
        x = self.head(x)
        
        # Reshape to image format
        B = x.shape[0]
        x = x.reshape(B, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], self.patch_size, self.patch_size, self.num_classes)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.num_classes, self.img_size[0], self.img_size[1])
        
        return torch.sigmoid(x)  # Output in range [0,1]

    def extra_repr(self) -> str:
        return f'img_size={self.img_size}, patch_size={self.patch_size}'