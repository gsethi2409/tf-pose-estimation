# Trained Models & Performances

## Models

I have tried multiple variations of models to find optmized network architecture. Some of them are below and checkpoint files are provided for research purpose.

- cmu
  - the model based VGG pretrained network which described in the original paper.
  - I converted Weights in Caffe format to use in tensorflow.
  - [pretrained weight download](https://www.dropbox.com/s/xh5s7sb7remu8tx/openpose_coco.npy?dl=0)

- dsconv
  - Same architecture as the cmu version except for the **depthwise separable convolution** of mobilenet.
  - I trained it using 'transfer learning', but it provides not-enough speed and accuracy.

- mobilenet
  - Based on the mobilenet paper, 12 convolutional layers are used as feature-extraction layers.
  - To improve on small person, **minor modification** on the architecture have been made.
  - Three models were learned according to network size parameters.
    - mobilenet
      - 368x368 : [checkpoint weight download](https://www.dropbox.com/s/09xivpuboecge56/mobilenet_0.75_0.50_model-388003.zip?dl=0)
    - mobilenet_fast
    - mobilenet_accurate
  - I published models which is not the best ones, but you can test them before you trained a model from the scratch.

- mobilenet v2
  - Similar to mobilenet, but using improved version of it.

| Name                 | Feature Layers      | Configuration                   |
|----------------------|---------------------|---------------------------------|
| cmu                  | VGG16               | OpenPose                        |
| mobilenet_thin       | Mobilenet           | width=0.75 refine-width=0.75    |
| mobilenet_v2_large   | Mobilenet v2 (582M) | width=1.40 refine-width=1.00    |
| mobilenet_v2_small   | Mobilenet v2 (97M)  | width=0.50 refine-width=0.50    |

## Performance on COCO Datasets

| Set         | Model               | Scale | Resolution | AP         | AP 50      | AP 75      | AP medium  | AP large   | AR         | AR 50      | AR 75      | AR medium  | AR large   |
|-------------|---------------------|-------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| 2014 Val    | Original Paper      | 3     | Image      |      0.584 |      0.815 |      0.626 |      0.544 |      0.651 |            |            |            |            |            |
| | | | | | | | | | | | | |
| 2014 Val    | CMU(openpose)       | 1     | Image      |     0.5067 |     0.7660 |     0.5377 |     0.4927 |     0.5309 |     0.5614 |     0.7900 |     0.5903 |     0.5089 |     0.6347 |
| 2014 Val    | VGG(openpose, our)  | 1     | Image      |     0.5067 |     0.7660 |     0.5377 |     0.4927 |     0.5309 |     0.5614 |     0.7900 |     0.5903 |     0.5089 |     0.6347 |
| 2017 Val    | VGG(openpose, our)  | 1     | Image      |     0.496  |     0.759  |     0.521  |     0.493  |     0.497  |     0.562  |     0.7830 |     0.590  |     0.506  |     0.644  |
| | | | | | | | | | | | | |
| 2014 Val    | Mobilenet thin      | 1     | Image      |     0.2806 |     0.5577 |     0.2474 |     0.2802 |     0.2843 |     0.3214 |     0.5840 |     0.2997 |     0.2946 |     0.3587 |
| 2014 Val    | Mobilenet-v2 Large  | 1     | Image      |     0.3130 |     0.5846 |     0.2940 |     0.2622 |     0.3850 |     0.3680 |     0.6101 |     0.3637 |     0.2765 |     0.4912 |
| 2014 Val    | Mobilenet-v2 Small  | 1     | Image      |     0.1730 |     0.4062 |     0.1240 |     0.1501 |     0.2105 |     0.2207 |     0.4505 |     0.1876 |     0.1601 |     0.3020 |

- I also ran keras & caffe models to verify single-scale version's performance, they matched this result.

## Computation Budget & Latency

| Model               | mAP@COCO2014 | GFLOPs | Latency(432x368)<br/>(Macbook 15' 2.9GHz i9, tf 1.12) | Latency(432x368)<br/>(V100 GPU) |
|---------------------|-------------:|--------|------------------------------------------------------:|-------------------------------:|
| CMU, VGG(OpenPose)  |              |        | 0.8589s | 0.0570s |
| Mobilenet thin      | 0.2806       |        | 0.1701s | 0.0217s |
| Mobilenet-v2 Large  | 0.3130       |        | 0.2066s | 0.0214s |
| Mobilenet-v2 Small  | 0.1730       |        | 0.1290s | 0.0210s |

Optimized Tensorflow was built before run this experiment. This may varies between environments, images and other factors.
