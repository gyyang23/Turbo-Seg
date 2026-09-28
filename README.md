# Turbo-Seg: Fully Convolutional Open-Vocabulary Semantic Segmentation with Turbo Decoding

by [Guoyu Yang](https://github.com/gyyang23), Xiaojing Wei, and Daming Shi (Shenzhen University)

## News🔥
- Turbo-Seg is accepted by **IEEE Transactions on Multimedia (TMM)**.

<p align="center">
  <img src="assets/Turbo-Seg Visualization Results.png" width="100%">
  <br>
  <em>Fig. 1. Segmentation results and inference speed of Turbo-Seg on smartphone-captured images, along with its performance-speed trade-off across the A-847, PC-459, A-150, and PAS-20 benchmarks.</em>
</p>

## Introduction

Open-vocabulary semantic segmentation (OVSS) leverages vision-language models (VLMs) like CLIP to generalize to categories that are unseen during training. Recent methods improve accuracy via cost aggregation mechanisms, but they often suffer from quadratic computational complexity with respect to the number of categories (leading to slow inference), or rely on acceleration techniques that degrade performance.

**Turbo-Seg** addresses these limitations with a fully convolutional design that combines:

- **A convolution-based VLM encoder** (CLIP ConvNeXt), which better preserves spatial information and maintains linear complexity with respect to input size.
- **Two cascaded Turbo Aggregation Decoders**, each rapidly refining category-specific masks through three synergistic aggregation mechanisms:
  - **Spatial Aggregation (SA)**: captures local spatial context via depthwise separable convolution and MLP.
  - **Text-Driven Aggregation (TDA)**: incorporates text features into the mask.
  - **Vision-Driven Aggregation (VDA)**: integrates multi-level vision features for finer details.

By replacing the Transformer's attention with convolutional operations, Turbo-Seg achieves linear complexity with respect to the number of categories, without relying on performance-degrading acceleration mechanisms (e.g., SED's Category Early Rejection).

<p align="center">
  <img src="assets/Turbo-Seg Framework.png" width="100%">
  <br>
  <em>Fig. 2. The overall framework of Turbo-Seg. A convolution-based VLM encoder (CLIP ConvNeXt) extracts vision and text features, and two cascaded Turbo Aggregation Decoders (each composed of SA, TDA, and VDA) progressively refine the category-specific masks.</em>
</p>

## Installation
Please follow [installation](INSTALL.md).

## Data Preparation
Please follow [dataset preparation](datasets/README.md).

## Demo
We provide a demo script to run Turbo-Seg on your own images.

```bash
python demo/demo.py --config-file ./configs/convnextB.yaml \
  --input /path/to/your/image.jpg \
  --output /path/to/output.png \
  --opts MODEL.WEIGHTS /path/to/weights.pth
```

## Training
We provide a shell script for training and evaluation. `run.sh` trains the model with the given config and evaluates the model after training.

To train or evaluate the model in different environments, modify the given shell script and config files accordingly.

```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For ConvNeXt-B variant
sh run.sh configs/convnextB.yaml 4 output/
# For ConvNeXt-L variant
sh run.sh configs/convnextL.yaml 4 output/
```

### Implementation details
Following the paper, the models are trained on COCO-Stuff (171 categories, 118,287 images) with the AdamW optimizer, a weight decay of 0.0001, and the cosine annealing learning rate schedule:

| Model | Input size | Iterations | VLM LR | Decoder LR | Batch size (per GPU) |
|---|---|---|---|---|---|
| CLIP ConvNeXt-B | 768×768 | 80,000 | 2×10⁻⁶ | 2×10⁻⁴ | 4 |
| CLIP ConvNeXt-L | 640×640 | 80,000 | 1×10⁻⁶ | 1×10⁻⁴ | 4 |

Note that the model's best performance does not occur at the final checkpoint but rather at an intermediate one.

## Evaluation
`run.sh` automatically evaluates the model following the evaluation protocol, using the weights in the output directory if not specified. To evaluate a specific checkpoint, pass it via `MODEL.WEIGHTS`:

```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

sh run.sh configs/convnextB.yaml 4 output/ MODEL.WEIGHTS /path/to/weights.pth
```

## Pretrained Models
We provide the benchmark results of Turbo-Seg. After the paper was accepted, we refactored and optimized the codebase and re-trained the models, yielding the weights below. All models are evaluated with 4 NVIDIA A100 GPUs. mIoU is reported.

| Name | VLM | Params | A-847 | PC-459 | A-150 | PC-59 | PAS-20 | Download |
|---|---|---|------|------|------|------|------|---|
| Turbo-Seg (B) | ConvNeXt-B | 180.6M | 12.4 | 19.0 | 32.9 | 55.0 | 95.4 | [download](https://drive.google.com/file/d/1xl4K4y3EynlFfc00WksGLJbS85VFW-e6/view?usp=sharing) |
| Turbo-Seg (L) | ConvNeXt-L | 353.1M | 16.0 | 23.0 | 35.7 | 57.9 | 97.1 | [download](https://drive.google.com/file/d/1umC-WWc31rX3lp3dfwPSdegWLm97httw/view?usp=sharing) |

Inference latency (ms, per image on an NVIDIA A100):

| Model | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|---|---|---|---|---|---|
| Turbo-Seg (ConvNeXt-B) | 132.1 | 77.4 | 33.5 | 21.0 | 15.4 |
| Turbo-Seg (ConvNeXt-L) | 159.1 | 93.0 | 40.2 | 24.3 | 18.1 |

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) and [SED](https://github.com/xb534/SED/tree/main), whose code has been utilized in this repository.

## Citing Turbo-Seg
```BibTeX
@article{yang2026turbo,
  title={Turbo-Seg: Fully Convolutional Open-Vocabulary Semantic Segmentation with Turbo Decoding},
  author={Yang, Guoyu and Wei, Xiaojing and Shi, Daming},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```
