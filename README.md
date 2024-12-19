# LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv_paper-2412.09262-b31b1b)](https://arxiv.org/abs/2412.09262)

</div>

## 📖 Introduction

<p align="center">
<img src="assets/frameworks_comparison.png" width=60%>
<p>

We present LatentSync, an end-to-end lip sync framework based on audio conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip sync methods based on pixel space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/ff3a84da-dc9b-498a-950f-5c54f58dd5c5 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/150e00fd-381e-4421-a478-a9ea3d1212a8 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/32c830a9-4d7d-4044-9b33-b184d8e11010 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/84e4fe9d-b108-44a4-8712-13a012348145 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7510a448-255a-44ee-b093-a1b98bd3961d controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/6150c453-c559-4ae0-bb00-c565f135ff41 controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/7ce04d50-d39f-4154-932a-ec3a590a8f64 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/70bde520-42fa-4a0e-b66c-d3040ae5e065 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/) and [EMO](https://humanaigc.github.io/emote-portrait-alive/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [ ] Data processing pipeline
- [ ] Training code

## Setting up the Environment

Install the required packages and download the checkpoints via:

```bash
source setup_env.sh
```

If the download is successful, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- latentsync_syncnet.pt
|-- whisper
|   `-- tiny.pt
|-- auxiliary
|   |-- 2DFAN4-cd938726ad.zip
|   |-- i3d_torchscript.pt
|   |-- koniq_pretrained.pkl
|   |-- s3fd-619a316812.pth
|   |-- sfd_face.pth
|   |-- syncnet_v2.model
|   |-- vgg16-397923af.pth
|   `-- vit_g_hybrid_pt_1200e_ssv2_ft.pth
```

These already include all the checkpoints required for latentsync training and inference. If you only want to try inference, you only need to download `latentsync_unet.pt` and `tiny.pt` from our [HuggingFace repo](https://huggingface.co/chunyu-li/LatentSync)

## Inference

Run the script for inference, which requires about 6.5GB GPU memory.

```bash
./inference.sh
```
