import os
import torch
import numpy as np
from PIL import Image
from basicsr.utils import imwrite
from swinfir.archs import SwinFIR
from swinfir.utils import load_model

def load_image(path):
    """Load and preprocess an image for SwinFIR."""
    img = Image.open(path).convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    img = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # CHW-RGB -> BCHW-RGB
    return img

def main(model_dir, input_path, output_path, device):
    # 1. Configure model programmatically to match original YAML settings
    opt = {
        'name': 'SwinFIR_SRx4',
        'model': 'SwinFIR',
        'scale': 4,
        'dataroot_lq': input_path,
        'results_dir': output_path,
        'path': {
            'pretrain_model': model_dir
        },
        'device': device,
        'network_g': {
            'type': 'SwinFIR',
            'upscale': 4,
            'in_chans': 3,
            'img_size': 60,
            'window_size': 12,
            'img_range': 1.0,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': 'transformattention'
        }
    }

    # 2. Initialize model with original architecture config
    model = SwinFIR(
        img_size=60,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=12,
        mlp_ratio=2,
        upscale=4,
        img_range=1.0,
        upsampler='pixelshuffle',
        resi_connection='transformattention'
    )
    model = model.to(device)
    load_model(model, opt['path']['pretrain_model'], strict=True)
    model.eval()

    # 3. Process images matching original workflow
    os.makedirs(output_path, exist_ok=True)
    img_list = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    with torch.no_grad():
        for img_name in img_list:
            img_path = os.path.join(input_path, img_name)
            
            # Load and prepare image
            lq = load_image(img_path).to(device)
            
            # Pad input image to be a multiple of window_size
            _, _, h, w = lq.size()
            window_size = 12
            mod_pad_h = (h // window_size + 1) * window_size - h
            mod_pad_w = (w // window_size + 1) * window_size - w
            lq = torch.cat([lq, torch.flip(lq, [2])], 2)[:, :, :h + mod_pad_h, :]
            lq = torch.cat([lq, torch.flip(lq, [3])], 3)[:, :, :, :w + mod_pad_w]
            
            # Inference
            output = model(lq)
            output = output[..., :h * 4, :w * 4]  # Remove padding
            
            # Post-process and save
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :])  # RGB->BGR for OpenCV
            output = (output * 255.0).round().astype(np.uint8)
            
            # Save with original filename
            basename = os.path.splitext(img_name)[0]
            save_path = os.path.join(output_path, f"{basename}.png")
            imwrite(output, save_path)

    print(f"Results saved to {output_path}")