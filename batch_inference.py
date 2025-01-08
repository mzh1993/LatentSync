import os
import glob
from typing import List, Tuple, Dict
import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from latentsync.whisper.audio2feature import Audio2Feature


def get_gender_pairs(input_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """获取男性和女性的音频-视频对应关系
    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]: 
        (女性音频到视频的映射, 男性音频到视频的映射)
        每个字典的格式: {音频路径: [视频路径1, 视频路径2, ...]}
    """
    female_pairs = {}  # 音频到视频列表的映射
    male_pairs = {}
    
    # 获取所有音频和视频文件
    audio_files = glob.glob(os.path.join(input_dir, "[gm]-*.wav"))
    video_files = glob.glob(os.path.join(input_dir, "[gm]-*.mp4"))
    
    # 对文件名进行排序以确保处理顺序一致
    audio_files.sort()
    video_files.sort()
    
    # 处理音频文件
    for audio_path in audio_files:
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        audio_prefix = basename.split('-')[0]  # 获取g或m前缀
        audio_id = basename.split('-')[1]  # 获取音频ID
        
        # 查找匹配的视频文件
        matching_videos = []
        for video_path in video_files:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            video_prefix = video_basename.split('-')[0]
            
            # 如果前缀相同（同性别），则添加到匹配列表
            if video_prefix == audio_prefix:
                matching_videos.append(video_path)
        
        # 根据性别存储音频-视频对应关系
        if matching_videos:  # 只在有匹配视频时添加
            if audio_prefix == 'g':
                female_pairs[audio_path] = matching_videos
            else:
                male_pairs[audio_path] = matching_videos
        else:
            print(f"警告: 未找到与音频 {audio_path} 匹配的视频文件")
    
    return female_pairs, male_pairs


def init_pipeline(config, args):
    """初始化推理pipeline"""
    print("正在初始化模型...")
    
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim必须是768或384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path, 
        device="cuda", 
        num_frames=config.data.num_frames
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=torch.float16
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=torch.float16)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    return pipeline


def process_batch(pipeline, pairs_dict, config, args, gender):
    """处理一批音视频文件
    Args:
        pairs_dict: Dict[str, List[str]] 音频到视频列表的映射
    """
    print(f"\n开始处理{gender}音视频文件 (共{len(pairs_dict)}个音频)")
    
    for audio_path, video_paths in pairs_dict.items():
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"\n处理音频: {audio_basename}")
        print(f"音频路径: {audio_path}")
        
        # 对每个匹配的视频进行处理
        for video_path in video_paths:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(args.output_dir, f"{video_basename}__{audio_basename}_out.mp4")
            mask_path = os.path.join(args.output_dir, f"{video_basename}__{audio_basename}_mask.mp4")
            
            print(f"\n处理视频: {video_basename}")
            print(f"视频路径: {video_path}")
            
            try:
                pipeline(
                    video_path=video_path,
                    audio_path=audio_path,
                    video_out_path=output_path,
                    video_mask_path=mask_path,
                    num_frames=config.data.num_frames,
                    num_inference_steps=config.run.inference_steps,
                    guidance_scale=args.guidance_scale,
                    weight_dtype=torch.float16,
                    width=config.data.resolution,
                    height=config.data.resolution,
                )
                print(f"处理完成: {output_path}")
            except Exception as e:
                print(f"处理失败 {video_basename} with {audio_basename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input', help='输入目录路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录路径')
    parser.add_argument('--unet_config_path', type=str, default='configs/unet/second_stage.yaml', help='UNet配置文件路径')
    parser.add_argument('--inference_ckpt_path', type=str, default='checkpoints/latentsync_unet.pt', help='模型检查点路径')
    parser.add_argument('--guidance_scale', type=float, default=1.5, help='guidance scale参数')
    parser.add_argument('--seed', type=int, default=1247, help='随机种子')
    args = parser.parse_args()

    if args.seed != -1:
        torch.manual_seed(args.seed)
    print(f"随机种子: {torch.initial_seed()}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    config = OmegaConf.load(args.unet_config_path)
    
    # 获取音频-视频对应关系
    female_pairs, male_pairs = get_gender_pairs(args.input_dir)
    
    if not (female_pairs or male_pairs):
        print("错误: 未找到任何有效的音视频对!")
        return

    # 显示匹配信息
    print("\n音视频匹配情况:")
    for gender, pairs in [("女性", female_pairs), ("男性", male_pairs)]:
        if pairs:
            print(f"\n{gender}音频匹配情况:")
            for audio, videos in pairs.items():
                print(f"音频 {os.path.basename(audio)} 匹配到 {len(videos)} 个视频:")
                for video in videos:
                    print(f"  - {os.path.basename(video)}")

    pipeline = init_pipeline(config, args)
    
    if female_pairs:
        process_batch(pipeline, female_pairs, config, args, "女性")
    
    if male_pairs:
        process_batch(pipeline, male_pairs, config, args, "男性")

    print("\n所有处理完成!")


if __name__ == '__main__':
    main() 