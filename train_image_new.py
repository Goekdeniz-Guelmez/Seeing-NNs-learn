import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch import optim, nn
from src.models import (
    Simple,
    Fourier,
    BasicAutoencoder,
    ConvolutionalAutoencoder,
    VariationalAutoencoder,
    CenteredLinearMap,
    VisionTransformer)
from torch.utils.data import DataLoader
from src.imageDataset import ImageDataset
from tqdm import tqdm
from src.videomaker import renderModel

def train_model(model_name, config, dataset, device):
    """
    Training function that handles different model architectures.
    
    Args:
        model_name (str): Name of the model to train
        config (dict): Configuration parameters
        dataset: Dataset instance
        device: torch device
    """
    # Create data loader
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Get image dimensions
    resx, resy = dataset.width, dataset.height
    input_size = resx * resy * 3  # for RGB images
    
    # Create coordinate grid
    linspace = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, resx),
            torch.linspace(1, -1, resy),
            indexing='ij'
        ),
        dim=-1
    ).to(device)
    linspace = torch.rot90(linspace, 1, (0, 1))
    
    # Initialize model based on architecture
    if model_name == 'SkipConn':
        model = Simple(
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers']
        ).to(device)
    elif model_name == 'Fourier':
        linmap = CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi)
        model = Fourier(
            fourier_order=32,
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers'],
            linmap=linmap
        ).to(device)
    elif model_name == 'BasicAutoencoder':
        model = BasicAutoencoder(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate']
        ).to(device)
    elif model_name == 'ConvolutionalAutoencoder':
        model = ConvolutionalAutoencoder(
            in_channels=3,
            base_channels=32,
            latent_dim=config['hidden_size'],
            input_size=(resx, resy)
        ).to(device)
    elif model_name == 'VariationalAutoencoder':
        model = VariationalAutoencoder(
            in_channels=3,
            base_channels=32,
            latent_dim=config['hidden_size'],
            input_size=(resx, resy)
        ).to(device)
    elif model_name == 'VisionTransformer':
        model = VisionTransformer(
            in_channels=3,
            num_classes=3,
            num_heads=6,
            depth= 12,
            embed_dim=config['hidden_size'],
            img_size=input_size,
            mlp_ratio=3.,
            dropout=config['dropout_rate'],
            attn_dropout=config['dropout_rate'],
            patch_size=16,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Set up loss function
    if model_name == 'VariationalAutoencoder':
        def loss_func(y_pred, y):
            recon_x, mu, log_var = y_pred
            recon_loss = F.binary_cross_entropy(recon_x, y, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            return recon_loss + config['kld_weight'] * kld_loss
    else:
        loss_func = nn.MSELoss()
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler_step'],
        gamma=0.5
    )
    
    # Training loop
    iteration, frame = 0, 0
    os.makedirs(f'./frames/{config["proj_name"]}', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        model.train()
        
        for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            if model_name == 'VariationalAutoencoder':
                y_pred, mu, log_var = model(x)
                loss = loss_func((y_pred, mu, log_var), y)
            elif model_name == 'VisionTransformer':
                x = model.process_input(x)
                y_pred = model(x)
                loss = loss_func(y_pred, y.permute(0, 3, 1, 2))  # Adjust target format
            else:
                y_pred = model(x)
                loss = loss_func(y_pred.squeeze(), y)
            
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            
            # Save visualization
            if iteration % config['save_every_n_iterations'] == 0:
                save_visualization(
                    model,
                    config['proj_name'],
                    frame,
                    resx,
                    resy,
                    linspace,
                    device
                )
                frame += 1
            iteration += 1
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(loader)}')
    
    # Create video from frames
    create_video(config['proj_name'])
    return model

def save_visualization(model, proj_name, frame, resx, resy, linspace, device, model_name='SkipConn'):
    """Save current model output as image"""
    model.eval()
    with torch.no_grad():
        if model_name == 'VisionTransformer':
            # For transformer, we need to create a proper image input
            coords = linspace.unsqueeze(0)  # Add batch dimension
            # Create a 3-channel input by repeating coordinates
            coords = coords.repeat(1, 3, 1, 1)
            output = model(coords)
            # Take first image from batch
            output = output[0].permute(1, 2, 0).cpu().numpy()
        else:
            output = renderModel(model, resx=resx, resy=resy, linspace=linspace)
        
        plt.imsave(
            f'./frames/{proj_name}/{frame:05d}.png',
            output,
            cmap='magma' if model_name != 'VisionTransformer' else None,
            origin='lower'
        )
    model.train()

def create_video(proj_name):
    """Create video from saved frames using ffmpeg"""
    os.system(
        f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png '
        f'-c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p '
        f'./frames/{proj_name}/{proj_name}.mp4'
    )

config = {
    'image_path': 'DatasetImages/evg.jpg',
    'hidden_size': 200,
    'num_hidden_layers': 30,
    'batch_size': 32,
    'lr': 0.001,
    'num_epochs': 50,
    'proj_name': 'Evangelion_wp',
    'save_every_n_iterations': 1,
    'scheduler_step': 3,
    'dropout_rate': 0.2,
    'kld_weight': 0.005  # for VAE
}

transformer_config = {
    'image_path': 'DatasetImages/gg.png',
    'hidden_size': 768,  # Embedding dimension
    'batch_size': 8,     # Smaller batch size due to memory requirements
    'lr': 0.0001,        # Lower learning rate for stability
    'num_epochs': 100,
    'proj_name': 'Vision_Transformer_Test',
    'save_every_n_iterations': 5,
    'scheduler_step': 5,
    'dropout_rate': 0.1,
    'patch_size': 16,
    'num_heads': 8,
    'transformer_depth': 12,
    'mlp_ratio': 4.0
}

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = ImageDataset(config['image_path'])
    
    # Train model (choose architecture)
    model_name = "VariationalAutoencoder"
    trained_model = train_model(model_name, config, dataset, device)