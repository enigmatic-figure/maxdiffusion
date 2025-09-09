from typing import Dict
import jax
import jax.numpy as jnp
from jax import random


def build_sgn_latents(
    rng: jax.Array,
    wildcard_embed: Dict,
    pipeline,
    alpha: float = 0.2,
    lowres_steps: int = 1,
    lowres_size: int = 256,
    target_size: int = 1024,
):
    """Construct SGN (structured Gaussian noise) latents.

    Steps:
      1. Run a low-resolution Hyper-SDXL denoising pass using ``wildcard_embed``
         and capture the latent tensor before VAE decode.
      2. Upsample to the target latent grid and normalize each channel to unit
         variance.
      3. Mix with white Gaussian noise using ``alpha``.
    """
    # Low-resolution latent pass
    latent_low = pipeline.generate_latents_only(
        rng=rng,
        prompt_embeds=wildcard_embed,
        num_inference_steps=lowres_steps,
        height=lowres_size,
        width=lowres_size,
    )

    gh = target_size // 8
    latent_up = jax.image.resize(latent_low, (latent_low.shape[0], 4, gh, gh), method="bilinear")
    mean = latent_up.mean(axis=(2, 3), keepdims=True)
    std = latent_up.std(axis=(2, 3), keepdims=True) + 1e-6
    latent_norm = (latent_up - mean) / std

    rng, sub = random.split(rng)
    z = random.normal(sub, latent_norm.shape, dtype=latent_norm.dtype)
    return jnp.sqrt(alpha) * latent_norm + jnp.sqrt(1.0 - alpha) * z
