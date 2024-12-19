#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/unet_latent_16_diffusion.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4"
