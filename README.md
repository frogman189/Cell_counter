# <h1 align="center">ECE 046211 - Technion - Deep Learning - Project </h1> 
## <h2 align="center"> "Cell Counter" - A Deep Learning Analysis of cell counting in microscopic image </h2>

## Abstract
In this project we tackle image-level cell counting on the LIVECell dataset: given a single phase-contrast image, predict one number—the total cells. We benchmark three practical routes used in microscopy: (1) instance segmentation with Mask R-CNN (count high-confidence detections; anchors retuned for tiny, crowded cells), (2) density regression with a U-Net-style FCN (predict a non-negative density map and sum it), and (3) global regression with a ConvNeXt backbone (map the whole image directly to a non-negative count). All models share consistent preprocessing (ImageNet normalization, light flips/rotations) for a fair comparison.

We evaluate with accuracy within ±k cells (k ∈ {0,1,3,5,10,20}) and standard MAE/MSE to reflect real lab tolerance. The study highlights where each approach shines: instance segmentation is interpretable (see misses vs. false positives), density maps are robust in confluent fields with ambiguous boundaries, and global regression is fast and simple when you only need the number.

Alongside the code, we provide ready-to-run training and evaluation scripts, clear configs, and swappable heads/backbones. The goal is a practical baseline for microscopy workflows—reducing annotation burden, handling dense scenes, and delivering reliable counts that plug into growth curves, confluence monitoring, and high-throughput screening.

<!-- <div align="center">
  <img src="./data/readme/yolo_video.gif" alt="Pothole Detection" width="600">
</div> -->
