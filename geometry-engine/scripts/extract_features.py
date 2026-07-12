import os
import torch
import numpy as np
import argparse
from PIL import Image
from diffusers import DiffusionPipeline
import sys

# Add parent directory to path to import hooks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.feature_extractor import DiTFeatureExtractor

def save_features(features, log_dir):
    """Saves the extracted features to the hierarchical log directory."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Save double blocks
    for timestep, layers in features.get("double_blocks", {}).items():
        for layer_idx, data in layers.items():
            # Save latent (img) stream
            if data["img"] is not None:
                img_path = os.path.join(log_dir, f"double_L{layer_idx}_t{timestep:.2f}_img.npy")
                np.save(img_path, data["img"].numpy())
            # Save condition (txt) stream
            if data["txt"] is not None:
                txt_path = os.path.join(log_dir, f"double_L{layer_idx}_t{timestep:.2f}_txt.npy")
                np.save(txt_path, data["txt"].numpy())
                
    # Save single blocks
    for timestep, layers in features.get("single_blocks", {}).items():
        for layer_idx, tensor in layers.items():
            path = os.path.join(log_dir, f"single_L{layer_idx}_t{timestep:.2f}.npy")
            np.save(path, tensor.numpy())
            
    print(f"✅ Features saved to {log_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract intermediate features from Hunyuan3D-2 DiT")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-path", type=str, default="tencent/Hunyuan3D-2", help="HuggingFace model ID or local path")
    parser.add_argument("--out-dir", type=str, default="logs/features", help="Output directory for saved features")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9], help="Timesteps to extract")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading pipeline from {args.model_path} on {device}...")
    
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(args.model_path, trust_remote_code=True)
    pipe = pipe.to(device)
    
    # Locate the DiT model inside the pipeline
    # For Hunyuan3D-2, it's typically named 'transformer' or 'model'
    dit_model = None
    if hasattr(pipe, 'transformer'):
        dit_model = pipe.transformer
    elif hasattr(pipe, 'model'):
        dit_model = pipe.model
    else:
        raise ValueError("Could not locate the DiT model inside the pipeline. Please inspect the pipeline object.")

    print("🔌 Attaching feature extractor...")
    extractor = DiTFeatureExtractor(dit_model)
    
    # Register hooks on all Double and Single stream blocks
    # We assume 16 double blocks and 32 single blocks based on Hunyuan3D-2 config
    num_double = len(dit_model.double_blocks) if hasattr(dit_model, 'double_blocks') else 0
    num_single = len(dit_model.single_blocks) if hasattr(dit_model, 'single_blocks') else 0
    
    extractor.register_hooks(
        double_indices=list(range(num_double)),
        single_indices=list(range(num_single))
    )
    extractor.set_target_timesteps(args.timesteps)

    # Load image
    print(f"🖼️ Loading image: {args.image}")
    input_image = Image.open(args.image).convert("RGB")

    print(f"⏳ Running inference... (Extracting at timesteps: {args.timesteps})")
    # Run a short inference pass (e.g., 20 steps is enough to hit the relative timesteps)
    # We set output_type to avoid trimesh/pymeshlab dependencies crashing
    with torch.no_grad():
        # Inference is run for its capture side-effects (hidden states via hooks);
        # the returned latents themselves are not consumed here.
        pipe(
            image=input_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            output_type='latent' # Return latents, skip marching cubes to save time/avoid errors
        )
        
    print("💾 Saving extracted features...")
    save_features(extractor.features, args.out_dir)
    
    # Verification
    n_features = 0
    for block_type in ["double_blocks", "single_blocks"]:
        for t in extractor.features[block_type]:
            n_features += len(extractor.features[block_type][t])
            
    print(f"✅ Successfully extracted {n_features} feature tensors.")
    
    # Clean up
    extractor.clear_hooks()

if __name__ == "__main__":
    main()
