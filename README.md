


# 3D Cell Tracking Pipeline (BioHub)

<h2>Overview</h2>
This project implements an input - output deep learning pipeline designed to detect and track cells in 3D microscopy time-series data. Built to handle the massive data constraints of 3D biological imaging, this pipeline utilizes a custom 3D U-Net architecture to predict probabilistic heatmaps of cell locations, forming the foundational 'cell detection' step for temporal cell tracking.

The system is optimized for high-performance computing, emphasizing memory-efficient data loading, mixed-precision training, and  spatial feature extraction.

<h2>Key Engineering Highlights</h2>

<b>Memory-Efficient Lazy Loading:</b> Implemented a custom PyTorch Dataset using Zarr to stream massive 3D volume chunks on-the-fly, bypassing standard RAM limitations for handling continuous time-series microscopy data.

<b>Target Heatmap Generation:</b> Engineered a custom label-generation system using ###scipy.ndimage.gaussian_filter### to convert sparse 3D spatial coordinates( X,Y,Z)  into smooth probability heatmaps, providing gradient soft targets for the model ( 0 being background and 1 indicating a cell ) .

<b>Optimized 3D Architecture:</b> Built a custom 3D U-Net in PyTorch with Instance Normalization and max-pooling bottlenecks to efficiently capture volumetric spatial relationships.

<b>Custom Loss Formulation:</b> Implemented Adaptive Wing Loss from scratch to heavily penalize errors in heatmap peaks while remaining tuned  to background noise, improving the localization accuracy of tight cell clusters.

<b>High-Performance Training:</b> Leveraged PyTorch Automatic Mixed Precision (AMP) and gradient scaling to maximize GPU utilization and batch  during training.

<h2>Tech Stack</h2>

<b>Deep Learning</b>: PyTorch, TorchAMP</b>

<b>Medical/3D Imaging: MONAI (Medical Open Network for AI)</b>

<b>Data Processing & Scientific Computing: Zarr, SciPy, NumPy, Pandas</b>

<b>Environment: CUDA, Google Colab</b>

Pipeline Architecture
<b>1. Data Augmentation</b>
Utilizes the MONAI library to apply dynamic, 3D-aware augmentations to prevent overfitting on complex volumetric data.

Transformations include RandZoomd (trilinear interpolation), RandRotate90d, and NormalizeIntensityd.

<b>2. Ground Truth & Data Streaming</b>
Microscopy datasets are very very very did i say very.. large(87GB) . The Custom PyTorch Dataset class addresses this by lazily fetching specific 16×Z×Y×X chunks directly from compressed .zarr directories.

Ground truth coordinates are dynamically converted into 3D Gaussian heatmaps during the __getitem__ call to ensure labels are perfectly aligned with the augmented input chunks.

<b>3. Model Architecture (3D U-Net)</b>
A fully custom 3D U-Net designed for volumetric regression:

<b>Encoder: Cascading ConvBlock modules with 3D Convolutions, Instance Normalization, and ReLU activations, followed by 1×2×2 Max Pooling.</b>

<b>Decoder: Trilinear Upsampling combined with skip connections from the encoder to retain high-frequency spatial details.</b>

<b>Output: A 1×1×1 3D Convolution outputting the final predicted probability map.</b>

<b>4. Training Loop & Validation</b>
Utilizes a highly optimized PyTorch training loop with torch.amp.autocast for mixed-precision training, drastically reducing VRAM usage and speeding up step times.

On-the-fly validation includes dynamic padding and Gaussian blurring using MONAI's GaussianFilter to smooth predictions before calculating the validation loss against the normalized ground truth peaks.

<h2>Linear Tracking</h2>
Integrated a linear assignment algorithm lapjv to draw edges between the predicted 3D cell nodes across sequential time points.



<h2> Results : </h2>

Link to the csv file ( I promise its safe to click, i am not a weirdo)  : https://docs.google.com/spreadsheets/d/1Tb96eQm5c0gNjqTsyYBxdKdh99rejJeaV4rOzehNTro/edit?usp=sharing

