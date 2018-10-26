import tensorflow as tf

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def l2_regularize(x) :
    loss = tf.reduce_mean(tf.square(x))
    return loss

def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

NO_OPS = 'NO_OPS'
def spectral_norm(W, u=None, num_iters=1, update_collection=tf.GraphKeys.UPDATE_OPS, with_sigma=False):
    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(cond=lambda i, _1, _2: i < num_iters,
                                        body=power_iteration,
                                        loop_vars=(tf.constant(0, dtype=tf.int32),
                                                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
                                        )
    
    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                      '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))

    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


def norm(input, norm_type="batch_norm", name="batch_norm"):
    if norm_type == "batch_norm":
        with tf.variable_scope(name) as scope:
            input = tf.identity(input)
            channels = input.get_shape()[3]

            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

            mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)

            normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)

            return normalized_batch
    elif norm_type == "instance_norm":
        with tf.variable_scope(name) as scope:
            return tf.contrib.layers.instance_norm(input,
                                                   epsilon=1e-05,
                                                   center=True, scale=True,
                                                   scope=scope)

    else:
        print ("Invalid Norm Type!!!")


def fc(input, output_size, sn=False, name="linear"):
    shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [shape[1], output_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        if sn:
            w = spectral_norm(w)
            
        return tf.matmul(input, w) + bias


def conv2d(input, out_filter, kernel=3, stride=2, padding="VALID", pad=0, pad_type='constant', spec_norm=False, name="conv2d"):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:

        input = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=pad_type)

        w = tf.get_variable("w", [kernel, kernel, input_shape[-1], out_filter], initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_filter], initializer=tf.constant_initializer(0.0))

        if spec_norm:
            w = spectral_norm(w)
        conv = tf.nn.conv2d(input, w, 
                            strides=[1, stride, stride, 1],
                            padding=padding
                            )
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        return conv

def deconv2d(input, out_filter, out_shape, kernel=3, stride=2, padding="SAME", pad=0, pad_type='constant', spec_norm=False, name="deconv2d"):
    input_shape = input.get_shape().as_list()

    #pad = kernel - 1 - pad
    #input = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=pad_type)

    out_shape = [input_shape[0], out_shape[0], out_shape[1], out_filter]
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [kernel, kernel, out_shape[-1], input_shape[-1]], initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))

        if spec_norm:
            w = spectral_norm(w)
        
        deconv = tf.nn.conv2d_transpose(input, w, 
                                        output_shape=out_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        return deconv


def resnet(input_tensor, channels, kernel=3, name='resnet'):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            net = conv2d(input_tensor, channels, kernel=3, stride=[1,1], pad=1, pad_type='reflect', name='conv_1')
            net = norm(net, norm_type="instance_norm", name="norm_1")
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope('res2'):
            net = conv2d(net, channels, kernel=3, stride=[1,1], pad=1, pad_type='reflect', name='conv_2')
            net = norm(net, norm_type="instance_norm", name="norm_2")

        return net + input_tensor


def mis_resnet(x_init, z, channels, sn=False, name='mis_resblock') :
    with tf.variable_scope(name):
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1]) # expand

        with tf.variable_scope('res_1') :
            x = conv2d(x_init, channels, kernel=3, stride=[1,1], pad=1, pad_type="reflect", sn=sn, name="conv_1")
            # x = norm(x, norm_type="instance_norm", name="norm_1")

            x = tf.concat([x, z], axis=-1)
            x = conv2d(x, channels * 2, kernel=1, stride=[1,1], sn=sn, name="conv_2")
            x = tf.nn.relu(x)

            x = conv2d(x, channels, kernel=1, stride=[1,1], sn=sn, name="conv_3")
            x = tf.nn.relu(x)

        with tf.variable_scope('res_2') :
            x = conv2d(x, channels, kernel=3, stride=[1,1], pad=1, pad_type='reflect', sn=sn, name='conv_1')
            # x = norm(x, norm_type="instance_norm", name="norm_2")

            x = tf.concat([x, z], axis=-1)
            x = conv2d(x, channels * 2, kernel=1, stride=[1,1], sn=sn, name='conv_2')
            x = tf.nn.relu(x)

            x = conv2d(x, channels, kernel=1, stride=[1,1], sn=sn, name='conv_3')
            x = tf.nn.relu(x)

        return x + x_init
