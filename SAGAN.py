import tensorflow as tf
import numpy as np

from ops import *
from architecture import *
from DataReader import DataReader

FLAGS = tf.app.flags.FLAGS

class SAGAN():
    def __init__(self):

        sess_config = tf.ConfigProto()
        sess_config.log_device_placement = False
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.48
        sess = tf.Session(config=sess_config)
        self.sess = tf.Session(config=sess_config)

        self.build_model()
        self.build_loss()
        
        #scalar summary
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.sig_d_loss_sum = tf.summary.scalar("sig_d_loss", self.sig_d_loss)       
        #image summary
        self.real_sum = tf.summary.image("real_img", self.images, max_outputs=5)
        self.fake_sum = tf.summary.image("fake_img", self.fake_img, max_outputs=5)
        
        #set optimizers here
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.g_loss, var_list=self.g_vars)
        self.sig_d_optim = tf.train.AdamOptimizer(self.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.sig_d_loss, var_list=self.sig_d_vars)

        
    def build_model(self):
        self.z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z')
        self.images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.img_hw, FLAGS.img_hw, 3], name='input_image')
        self.lr = tf.placeholder(tf.float32, name='lr')
                
        self.fake_img, self.g_nets = self.generator(self.z, name='generator')

        #real/fake logits will be used to train SAGAN
        #real/fake sig logit/out will be used to train the attached extra layer
        self.real_logits, self.real_sig_logits, self.real_sig_out, _ = self.discriminator(self.images, name='discriminator')
        self.fake_logits, self.fake_sig_logits, self.fake_sig_out, _ = self.discriminator(self.fake_img, name='discriminator', reuse=True)

        all_vars = tf.trainable_variables()
        self.d_vars = [var for var in all_vars if 'discriminator' in var.name and 'sig_layer' not in var.name]
        self.sig_d_vars = [var for var in all_vars if 'sig_layer' in var.name]
        self.g_vars = [var for var in all_vars if 'generator' in var.name]

    # Loss functions
    def build_loss(self):
        def calc_loss(logits, label):
            if label==1:
                y = tf.ones_like(logits)
            else:
                y = tf.zeros_like(logits)

            # use reduce sum when using window based discriminator
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
        

        # # vanilla gan loss
        # self.fake_d_loss = calc_loss(self.fake_logits, 0)
        # self.real_d_loss = calc_loss(self.real_logits, 1)
        # self.d_loss = self.fake_d_loss + self.real_d_loss
        # self.g_loss = calc_loss(self.fake_logits, 1)

        #hinge loss
        self.fake_d_loss = tf.reduce_mean(tf.nn.relu(1.0 + self.fake_logits))
        self.real_d_loss = tf.reduce_mean(tf.nn.relu(1.0 - self.real_logits))
        self.d_loss = self.fake_d_loss + self.real_d_loss
        self.g_loss = -tf.reduce_mean(self.fake_logits)

        #binary cross entropy loss
        self.sig_d_loss = -tf.reduce_mean(tf.log(self.real_sig_out) + tf.log(1 - self.fake_sig_out))
       
    def generator(self, z, name='generator'):
        nets=[]
        with tf.variable_scope(name) as scope: 
            net = tf.reshape(fc(z, 4*4*512, name="linear"), [-1, 4, 4, 512])
            net = norm(net, 'batch_norm', name='bn1')
            net = tf.nn.relu(net)
            nets.append(net)
            
            net = deconv2d(net, 256, [8,8], spec_norm=True, name="deconv_2")
            net = norm(net, 'batch_norm', name='bn2')
            net = tf.nn.relu(net)
            nets.append(net)
            
            net = deconv2d(net, 128, [16,16], spec_norm=True, name="deconv_3")
            net = norm(net, 'batch_norm', name='bn3')
            net = tf.nn.relu(net)
            nets.append(net)

            # Attention layer here
            net = self.attention(net, 128, name='attention_1')
            nets.append(net)
            
            net = deconv2d(net, 64, [32,32], spec_norm=True, name="deconv_4")
            net = norm(net, 'batch_norm', name='bn4')
            net = tf.nn.relu(net)
            nets.append(net)
            
            # Attention layer here
            net = self.attention(net, 64, name='attention_2')
            nets.append(net)
            
            net = deconv2d(net, 3, [64,64], spec_norm=True, name="deconv_5")
            net = tf.nn.tanh(net)
            nets.append(net)
            
        return net, nets
        
    
    def discriminator(self, input_img, name='discriminator', reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope: 
            net = conv2d(input_img, 64, kernel=4, stride=2, pad=1, spec_norm=True, name="conv_1")
            net = tf.nn.leaky_relu(net)
            nets.append(net)
            
            net = conv2d(net, 128, kernel=4, stride=2, pad=1, spec_norm=True, name="conv_2")
            net = tf.nn.leaky_relu(net)
            nets.append(net)
            
            net = conv2d(net, 256, kernel=4, stride=2, pad=1, spec_norm=True, name="conv_3")
            net = tf.nn.leaky_relu(net)
            nets.append(net)
            
            net = self.attention(net, 256, name='attention_1')
            nets.append(net)
            
            net = conv2d(net, 512, kernel=4, stride=2, pad=1, spec_norm=True, name="conv_4")
            net = tf.nn.leaky_relu(net)
            nets.append(net)
            
            net = self.attention(net, 512, name='attention_2')
            nets.append(net)
           
            with tf.variable_scope('sig_layer', reuse=reuse) as scope:
                sig_layer = fc(tf.reshape(net, [FLAGS.batch_size, -1]), 512, name='fc_layer1')
                sig_layer = fc(sig_layer, 256, name='fc_layer2')
                sig_logit = fc(sig_layer, 1, name='fc_layer3')
                sig_out = tf.nn.sigmoid(sig_logit)

            net = conv2d(net, 1, kernel=4, stride=1, name="conv_5")
            nets.append(net)
            
        return net, sig_logit, sig_out, nets
    
    def attention(self, x, channel, name='attention'):
        nets = []
        with tf.variable_scope(name) as scope:
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            f = conv2d(x, channel//8, kernel=1, stride=1, spec_norm=True, name="conv_f")
            g = conv2d(x, channel//8, kernel=1, stride=1, spec_norm=True, name="conv_g")
            h = conv2d(x, channel, kernel=1, stride=1, spec_norm=True, name="conv_h")
            
            proj_query = tf.reshape(f, [tf.shape(f)[0], -1, tf.shape(f)[-1]])
            proj_key = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
            proj_value = tf.reshape(h, [tf.shape(h)[0], -1, tf.shape(h)[-1]])
            
            s = tf.matmul(proj_key, proj_query, transpose_b=True)
            atten_map = tf.nn.softmax(s, axis=-1)

            o = tf.matmul(atten_map, proj_value)
            o = tf.reshape(o, tf.shape(x))

            x = gamma * o + x

            return x

