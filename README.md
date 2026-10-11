


# 3D Cell Tracking Pipeline (BioHub)

##Overview
This project implements an input - output deep learning pipeline designed to detect and track cells in 3D microscopy time-series data. Built to handle the massive data constraints of 3D biological imaging, this pipeline utilizes a custom 3D U-Net architecture to predict probabilistic heatmaps of cell locations, forming the foundational 'cell detection' step for temporal cell tracking.

The system is optimized for high-performance computing, emphasizing memory-efficient data loading, mixed-precision training, and  spatial feature extraction.

##Key Engineering Highlights

Memory-Efficient Lazy Loading: Implemented a custom PyTorch Dataset using Zarr to stream massive 3D volume chunks on-the-fly, bypassing standard RAM limitations for handling continuous time-series microscopy data.

Target Heatmap Generation: Engineered a custom label-generation system using ###scipy.ndimage.gaussian_filter### to convert sparse 3D spatial coordinates( X,Y,Z)  into smooth probability heatmaps, providing gradient soft targets for the model ( 0 being background and 1 indicating a cell ) .

Optimized 3D Architecture: Built a custom 3D U-Net in PyTorch with Instance Normalization and max-pooling bottlenecks to efficiently capture volumetric spatial relationships.

Custom Loss Formulation: Implemented Adaptive Wing Loss from scratch to heavily penalize errors in heatmap peaks while remaining tuned  to background noise, improving the localization accuracy of tight cell clusters.

High-Performance Training: Leveraged PyTorch Automatic Mixed Precision (AMP) and gradient scaling to maximize GPU utilization and batch  during training.

Tech Stack

Deep Learning: PyTorch, TorchAMP

Medical/3D Imaging: MONAI (Medical Open Network for AI)

Data Processing & Scientific Computing: Zarr, SciPy, NumPy, Pandas

Environment: CUDA, Google Colab

Pipeline Architecture
1. Data Augmentation
Utilizes the MONAI library to apply dynamic, 3D-aware augmentations to prevent overfitting on complex volumetric data.

Transformations include RandZoomd (trilinear interpolation), RandRotate90d, and NormalizeIntensityd.

2. Ground Truth & Data Streaming
Microscopy datasets are very very  large(87GB) . The Custom PyTorch Dataset class addresses this by lazily fetching specific 16×Z×Y×X chunks directly from compressed .zarr directories.

Ground truth coordinates are dynamically converted into 3D Gaussian heatmaps during the __getitem__ call to ensure labels are perfectly aligned with the augmented input chunks.

3. Model Architecture (3D U-Net)
A fully custom 3D U-Net designed for volumetric regression:

Encoder: Cascading ConvBlock modules with 3D Convolutions, Instance Normalization, and ReLU activations, followed by 1×2×2 Max Pooling.

Decoder: Trilinear Upsampling combined with skip connections from the encoder to retain high-frequency spatial details.

Output: A 1×1×1 3D Convolution outputting the final predicted probability map.

4. Training Loop & Validation
Utilizes a highly optimized PyTorch training loop with torch.amp.autocast for mixed-precision training, drastically reducing VRAM usage and speeding up step times.

On-the-fly validation includes dynamic padding and Gaussian blurring using MONAI's GaussianFilter to smooth predictions before calculating the validation loss against the normalized ground truth peaks.

Next Steps / Future Implementation
Temporal Tracking: Integrate a linear assignment algorithm (e.g., Jonker-Volgenant / lapjv) to draw edges between the predicted 3D cell nodes across sequential time points, completing the full spatial-temporal tracking objective.

Inference Optimization: Export the trained weights and build an inference pipeline optimized for batch-processing test sets to finalize submission datasets.
