from __future__ import absolute_import

import tensorflow as tf

from tf_pose import network_base
from tf_pose.mobilenet import mobilenet_v2
from tf_pose.network_base import layer


class Mobilenetv2Network(network_base.BaseNetwork):
    def __init__(self, inputs, trainable=True, conv_width=1.0, conv_width2=1.0):
        self.conv_width = conv_width
        self.refine_width = conv_width2
        network_base.BaseNetwork.__init__(self, inputs, trainable)

    @layer
    def base(self, input, name):
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
            net, endpoints = mobilenet_v2.mobilenet_base(input, self.conv_width, finegrain_classification_mode=(self.conv_width < 1.0))
            for k, tensor in sorted(list(endpoints.items()), key=lambda x: x[0]):
                self.layers['%s/%s' % (name, k)] = tensor
                # print(k, tensor.shape)
            return net

    def setup(self):
        depth2 = lambda x: int(x * self.refine_width)

        self.feed('image').base(name='base')

        # TODO : add more feature with downsample?
        # self.feed('base/layer_4/output').max_pool(2, 2, 2, 2, name='base/layer_4/output/downsample')
        # self.feed('base/layer_4/output').avg_pool(2, 2, 2, 2, name='base/layer_4/output/downsample')
        self.feed('base/layer_14/output').upsample(factor='base/layer_7/output', name='base/layer_14/output/upsample')
        (self.feed(
            'base/layer_7/output',
            'base/layer_14/output/upsample',
            # 'base/layer_4/output/downsample'
        ).concat(3, name='feat_concat'))

        feature_lv = 'feat_concat'
        with tf.variable_scope(None, 'Openpose'):
            prefix = 'MConv_Stage1'
            (self.feed(feature_lv)
             # .se_block(name=prefix + '_L1_se', ratio=8)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L1_4')
             .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

            (self.feed(feature_lv)
             # .se_block(name=prefix + '_L2_se', ratio=8)
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
             .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
             .separable_conv(1, 1, depth2(512), 1, name=prefix + '_L2_4')
             .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            for stage_id in range(5):
                prefix_prev = 'MConv_Stage%d' % (stage_id + 1)
                prefix = 'MConv_Stage%d' % (stage_id + 2)
                (self.feed(prefix_prev + '_L1_5',
                           prefix_prev + '_L2_5',
                           feature_lv)
                 .concat(3, name=prefix + '_concat')
                 # .se_block(name=prefix + '_L1_se', ratio=8)
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_1')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_2')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L1_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L1_4')
                 .separable_conv(1, 1, 38, 1, relu=False, name=prefix + '_L1_5'))

                (self.feed(prefix + '_concat')
                 # .se_block(name=prefix + '_L2_se', ratio=8)
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_1')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_2')
                 .separable_conv(3, 3, depth2(128), 1, name=prefix + '_L2_3')
                 .separable_conv(1, 1, depth2(128), 1, name=prefix + '_L2_4')
                 .separable_conv(1, 1, 19, 1, relu=False, name=prefix + '_L2_5'))

            # final result
            (self.feed('MConv_Stage6_L2_5',
                       'MConv_Stage6_L1_5')
             .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def loss_last(self):
        return self.get_output('MConv_Stage6_L1_5'), self.get_output('MConv_Stage6_L2_5')

    def restorable_variables(self, only_backbone=True):
        vs = {v.op.name: v for v in tf.global_variables() if
              ('MobilenetV2' in v.op.name or (only_backbone is False and 'Openpose' in v.op.name)) and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'quant' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        # print(set([v.op.name for v in tf.global_variables()]) - set(list(vs.keys())))
        return vs
