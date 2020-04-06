## Training

### Coco Dataset 

You should download COCO Dataset from http://cocodataset.org/#download

Also, you need to install cocoapi for easy parsing : https://github.com/cocodataset/cocoapi


```
$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI
$ python3 setup.py build_ext --inplace
$ python3 setup.py build_ext install
```

### Augmentation

CMU Perceptual Computing Lab has modified Caffe to provide data augmentation. See : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

I implemented the augmentation codes as the way of the original version, See [pose_dataset.py](pose_dataset.py) and [pose_augment.py](pose_augment.py). This includes scaling, rotation, flip, cropping.

This process can be a bottleneck for training, so if you have enough computing resources, please see [Run for Faster Training]() Section

### Run

```
$ python3 train.py --model=cmu --datapath={datapath} --batchsize=64 --lr=0.001 --modelpath={path-to-save}

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

If you want to reproduce the original paper's result, the following setting is recommended.

- model : vgg
- lr : 0.0001 or 0.00004
- input-width = input-height = 368x368 or 432x368
- batchsize : 10 (I trained with batchsizes up to 128, they are trained well)

| Heatmap Loss                              | PAFmap(Part Affinity Field) Loss         |
|-------------------------------------------|------------------------------------------|
| ![train_loss_cmu](/etcs/loss_ll_heat.png) | ![train_loss_cmu](/etcs/loss_ll_paf.png) |  

As you can see from the table above, training loss was converged at the almost same trends with the original paper.
 
The mobilenet versions has slightly poor loss value compared to the original one. Training losses are 3 to 8% larger, though validation losses are 5 to 14% larger.


### Run for Faster Training

If you have enough computing resources in multiple nodes, you can launch multiple workers on nodes to help data preparation.
 
```
worker-node1$ python3 pose_dataworker.py --master=tcp://host:port
worker-node2$ python3 pose_dataworker.py --master=tcp://host:port
worker-node3$ python3 pose_dataworker.py --master=tcp://host:port
...
```

After above preparation, you can launch training script with 'remote-data' arguments.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

Also, You can quickly train with multiple gpus. This automatically splits batch into multiple gpus for forward/backward computations.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port --gpus=8

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

I trained models within a day with 8 gpus and multiple pre-processing nodes with 48 core cpus.

### Model Optimization for Inference

After trained a model, I optimized models by folding batch normalization to convolutional layers and removing redundant operations.  

Firstly, the model should be frozen.

```bash
$ python3 -m tensorflow.python.tools.freeze_graph \
  --input_graph=... \
  --output_graph=... \
  --input_checkpoint=... \
  --output_node_names="Openpose/concat_stage7"
```

And the optimization can be performed on the frozen model via graph transform provided by tensorflow. 

```bash
$ bazel build tensorflow/tools/graph_transforms:transform_graph
$ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=./tmp/graph_frozen.pb \
    --out_graph=./tmp/graph_opt.pb \
    --inputs='image:0' \
    --outputs='Openpose/concat_stage7:0' \
    --transforms='
    strip_unused_nodes(type=float, shape="1,368,368,3")
    fold_old_batch_norms
    fold_batch_norms
    fold_constants(ignoreError=False)
    remove_nodes(op=Identity, op=CheckNumerics)'
```

Also, It is promising to quantize neural network in 8 bit to get futher improvement for speed. In my case, this will make inference less accurate and take more time on Intel's CPUs.
 
```
$ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/Users/ildoonet/repos/tf-openpose/tmp/cmu_640x480/graph_opt.pb \
  --out_graph=/Users/ildoonet/repos/tf-openpose/tmp/cmu_640x480/graph_q.pb \
  --inputs='image' \
  --outputs='Openpose/concat_stage7:0' \
  --transforms='add_default_attributes strip_unused_nodes(type=float, shape="1,360,640,3")
    remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)
    fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes
    strip_unused_nodes sort_by_execution_order'
```
