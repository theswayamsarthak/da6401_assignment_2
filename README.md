# DA6401 Assignment 2 — Visual Perception Pipeline

Implementation of a multi-task visual perception pipeline on the Oxford-IIIT Pet Dataset using PyTorch.

## Links

- WandB Report: [DA6401 Assignment - 2 by Swayam Sarthak ee23b079](https://wandb.ai/theswayamsarthak-iitmaana/da6401-assignment2/reports/Visual-Perception-Pipeline--VmlldzoxNjQ5MzkyMg?accessToken=tp85rhw7mnzdn91mbza524qz8qy7z84z73cxztjct28vjjzhtq6phzcbe6chxxem)
- GitHub Repo: YOUR_GITHUB_LINK_HERE

## Tasks

- Task 1: VGG11 classification with CustomDropout and BatchNorm across 37 breed classes
- Task 2: Object localization with custom IoU loss predicting [x_center, y_center, width, height]
- Task 3: U-Net semantic segmentation with transposed convolution decoder
- Task 4: Unified multi-task model with shared VGG11 backbone across all three heads

## Results

| Task           | Metric         | Value |
|----------------|----------------|-------|
| Classification | Val Accuracy   | 33.0% |
| Localization   | Train IoU Loss | 0.375 |
| Segmentation   | Val Dice Score | 0.629 |

## Project Structure

    .
    checkpoints/
        checkpoints.md
    data/
        pets_dataset.py
    inference.py
    losses/
        __init__.py
        iou_loss.py
    models/
        __init__.py
        classification.py
        layers.py
        localization.py
        multitask.py
        segmentation.py
        vgg11.py
    multitask.py
    README.md
    requirements.txt
    train.py

## Setup

    pip install -r requirements.txt

## Training

    python train.py --task classification --epochs 30 --lr 3e-4 --batch_size 64 --data_root data
    python train.py --task localization   --epochs 50 --lr 1e-4 --batch_size 64 --freeze_encoder --data_root data
    python train.py --task segmentation   --epochs 30 --lr 1e-4 --batch_size 64 --freeze_encoder --data_root data

## Inference

    from inference import run_inference
    result = run_inference("path/to/image.jpg")
    print(result["class_id"])
    print(result["bbox"])
    print(result["seg_mask"])

## Checkpoints

Checkpoints are automatically downloaded from Google Drive when
MultiTaskPerceptionModel is initialized. No manual download needed.

## Dependencies

torch, numpy, pillow, albumentations, wandb, scikit-learn, gdown
