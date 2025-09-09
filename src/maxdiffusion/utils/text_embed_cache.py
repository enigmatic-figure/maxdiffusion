import torch
import jax.numpy as jnp

def _to_jax(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)

def load_precomputed_embed_pt(pt_path: str):
    """Load precomputed SDXL text embeddings from a ``.pt`` file.

    The file is expected to be created by ``torch.save`` with the following
    structure::

        {
            "id": <prompt id>,
            "clip_l": {
                "prompt_embeds": Tensor[1, 77, 768],
                "pooled_prompt_embeds": Tensor[1, 768],
                "neg_prompt_embeds": Tensor[1, 77, 768],
                "neg_pooled_prompt_embeds": Tensor[1, 768],
            },
            "clip_g": {
                "pooled_prompt_embeds": Tensor[1, Dg],
                "neg_pooled_prompt_embeds": Tensor[1, Dg],
            }
        }

    Returns a dictionary of JAX arrays with keys ``clip_l``, ``clip_g``,
    ``neg_clip_l`` and ``neg_clip_g``. The CLIP-L entries contain a tuple of
    ``(prompt_embeds, pooled_prompt_embeds)``.  The OpenCLIP-bigG entries only
    include pooled embeddings as SDXL uses the pooled representation for the
    second encoder.
    """
    blob = torch.load(pt_path, map_location="cpu")
    clip_l = blob["clip_l"]
    clip_g = blob["clip_g"]
    return {
        "clip_l": (
            _to_jax(clip_l["prompt_embeds"]),
            _to_jax(clip_l["pooled_prompt_embeds"]),
        ),
        "clip_g": (None, _to_jax(clip_g["pooled_prompt_embeds"])),
        "neg_clip_l": (
            _to_jax(clip_l["neg_prompt_embeds"]),
            _to_jax(clip_l["neg_pooled_prompt_embeds"]),
        ),
        "neg_clip_g": (None, _to_jax(clip_g["neg_pooled_prompt_embeds"])),
    }
