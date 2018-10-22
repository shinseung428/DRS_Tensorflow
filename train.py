import os
import tensorflow as tf
from config import *
from SAGAN import *
from ops import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS

def train(model):
    
    saver = tf.train.Saver()
    if FLAGS.continue_train:
        last_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
        saver.restore(model.sess, last_ckpt)
        epoch = int(last_ckpt.split('_')[-1].split('.')[0])
        print ('Loaded model from %s'%last_ckpt)
    else:
        model.sess.run(tf.global_variables_initializer())
        epoch = 0
        
    #summary init
    all_summary = tf.summary.merge([model.g_loss_sum,
                                    model.real_sum,
                                    model.d_loss_sum,
                                    model.fake_sum,
                                    model.sig_d_loss_sum
                                    ])
    writer = tf.summary.FileWriter(FLAGS.log_path, model.sess.graph)

    
    
    datareader = DataReader()
    images = datareader.prepare_data()
    batch_ids = len(images)//FLAGS.batch_size

    print ('Start Training...')
    global_step = 0
    # for _ in range(FLAGS.epochs+FLAGS.extra_epochs):
    while epoch < FLAGS.epochs+FLAGS.extra_epochs:
        step = 0
        for idx in range(batch_ids):
            img_batch = images[idx*FLAGS.batch_size:idx*FLAGS.batch_size+FLAGS.batch_size]
            z = np.random.uniform(-1, 1, (FLAGS.batch_size, FLAGS.z_dim))
            if len(img_batch) != FLAGS.batch_size:
                continue
 
            #half of the extra epochs will be used to further train the model using smaller learning rate
            learning_rate = 1e-7 if epoch > FLAGS.epochs else FLAGS.learning_rate
            f_dict = {model.z : z,
                      model.images : img_batch,
                      model.lr : learning_rate,
                      }

            # last few epochs used to train the discriminator(sigmoid layer)
            summary, d_loss, _ = model.sess.run([all_summary, model.d_loss, model.d_optim],
                                                feed_dict = f_dict
                                                )
            writer.add_summary(summary, global_step)

            # update generator
            for _ in range(2):
                summary, g_loss, _ = model.sess.run([all_summary, model.g_loss, model.g_optim],
                                                     feed_dict = f_dict
                                                    )
                writer.add_summary(summary, global_step)

            print ("Epoch: [%d] Step: [%d] G_Loss: %.4f D_Loss: %.4f"%(epoch, step, g_loss, d_loss))
            step += 1
            global_step += 1
        
        ckpt_path = os.path.join(FLAGS.ckpt_path, 'model_%d.ckpt'%epoch)
        save_path = saver.save(model.sess, ckpt_path)
        print("Model saved in path : %s"%save_path)
        sample_imgs = model.sess.run(model.fake_img, feed_dict={model.z : z})
        img_tile(epoch, sample_imgs)
        epoch += 1

    print ('Train sigmoid layer...')
    processed_samples = 0
    while processed_samples < 100000:
        idx = np.random.randint(0, batch_ids)
        img_batch = images[idx*FLAGS.batch_size:idx*FLAGS.batch_size+FLAGS.batch_size]
        z = np.random.uniform(-1, 1, (FLAGS.batch_size, FLAGS.z_dim))

        #half of the extra epochs will be used to further train the model using smaller learning rate
        learning_rate = FLAGS.learning_rate
        f_dict = {model.z : z,
                  model.images : img_batch,
                  model.lr : learning_rate,
                  }
        summary, sig_d_loss, _ = model.sess.run([all_summary, model.sig_d_loss, model.sig_d_optim],
                                                feed_dict = f_dict
                                               )
        writer.add_summary(summary, global_step)
        print ("Samples #: %d D_Loss: %.4f"%(processed_samples, sig_d_loss))
        processed_samples += FLAGS.batch_size

    ckpt_path = os.path.join(FLAGS.ckpt_path, 'model_%d.ckpt'%epoch)
    save_path = saver.save(model.sess, ckpt_path)
        
    print ("Done.")


def main():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    #create graph, images, and checkpoints folder if they don't exist
    if not os.path.exists(FLAGS.ckpt_path):
        os.makedirs(FLAGS.ckpt_path)
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    if not os.path.exists(FLAGS.res_path):
        os.makedirs(FLAGS.res_path)
    with tf.Graph().as_default():
        print ("Init. model...")
        model = SAGAN()
        train(model)
        
main()

