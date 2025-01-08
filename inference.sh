#!/bin/bash

# 创建输出目录
mkdir -p output

# 遍历input目录下的所有视频文件
for video in input/[gm]-*.mp4; do
    if [ -f "$video" ]; then
        # 获取文件名（不含路径和扩展名）
        filename=$(basename "$video" .mp4)
        
        # 检查对应的音频文件是否存在
        audio="input/${filename}.wav"
        if [ -f "$audio" ]; then
            echo "Processing: $filename"
            
            # 根据文件名前缀设置不同的guidance_scale
            if [[ $filename == g-* ]]; then
                guidance_scale=1.5  # 女性音色参数
            else
                guidance_scale=1.5  # 男性音色参数
            fi
            
            # 运行推理脚本
            python -m scripts.inference \
                --unet_config_path "configs/unet/second_stage.yaml" \
                --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
                --guidance_scale $guidance_scale \
                --video_path "$video" \
                --audio_path "$audio" \
                --video_out_path "output/${filename}_out.mp4"
        else
            echo "Warning: No matching audio file found for $video"
        fi
    fi
done
