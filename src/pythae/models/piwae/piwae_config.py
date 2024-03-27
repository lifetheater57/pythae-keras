from pydantic.dataclasses import dataclass

from ..vae import VAE_PTConfig


@dataclass
class PIWAEConfig(VAE_PTConfig):
    """Partially IWAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_gradient_estimates (int): Number of (M-)estimates to use for the gradient
            estimate for the encoder. Default: 5
        number_samples (int): Number of samples to use on the Monte-Carlo estimation. Default: 10
    """

    number_gradient_estimates: int = 5
    number_samples: int = 10
