import os
import tensorflow as tf
from config import *
from SAGAN import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS
def sigmoid(F):
    return 1/(1 + np.exp(-F))
def sample(model):
    # datareader = DataReader()
    # images = datareader.prepare_data()
    # batch_ids = len(images)//FLAGS.batch_size
    
    saver = tf.train.Saver()
    last_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    saver.restore(model.sess, last_ckpt)
    print ('Loaded model from %s'%last_ckpt)

    print ('Start BurnIn...')
    max_M = 0.0
    max_logit = 0.0
    processed_samples = 0
    while processed_samples < FLAGS.burnin_samples:
        z = np.random.uniform(-1, 1, (FLAGS.batch_size, FLAGS.z_dim))
        # idx = np.random.randint(0, batch_ids)
        # img_batch = images[idx*FLAGS.batch_size:idx*FLAGS.batch_size+FLAGS.batch_size]
        # fake_imgs, fake_sig_out, fake_logits, real_sig_out = model.sess.run([model.fake_img,
        #                                                                      model.fake_sig_out,
        #                                                                      model.fake_logits,
        #                                                                      model.real_sig_out],
        #                                                                      feed_dict = {model.z : z,
        #                                                                                   model.images : img_batch
        #                                                                                  }
        #                                                                     )
        # logits = model.sess.run(tf.reshape(fake_logits, [-1]))
        # batch_ratio = real_sig_out/fake_sig_out
        # max_idx = np.argmax(batch_ratio)
        # max_ratio = batch_ratio[max_idx]

        
        fake_imgs, discrim_logits = model.sess.run([model.fake_img, model.fake_logits],
                                                    feed_dict = {model.z : z}
                                                   )
        logits = model.sess.run(tf.reshape(discrim_logits, [-1]))
        batch_ratio = np.exp(logits)
        max_idx = np.argmax(batch_ratio)
        max_ratio = batch_ratio[max_idx]
        
        
        if max_ratio > max_M:
            max_M = max_ratio
            max_logit = logits[max_idx]
        processed_samples += FLAGS.batch_size
   

    accepted_samples = []
    while len(accepted_samples) < FLAGS.total_samples:
        z = np.random.uniform(-1, 1, (FLAGS.batch_size, FLAGS.z_dim))

        fake_imgs, discrim_logits = model.sess.run([model.fake_img, model.fake_logits],
                                                    feed_dict = {model.z : z}
                                                   )
        logits = model.sess.run(tf.reshape(discrim_logits, [-1]))

        # go through generated images
        # reject the ones that don't meet the condition
        for idx, logit in enumerate(logits):
            M = np.exp(logit)
            #update max_M if larger M is found
            if M > max_M:
                max_M = M
                max_logit = logit
                
            #calculate F and pass it into sigmoid
            # set gamma dynamically (80th percentile of F)
            F = logit - max_logit - np.log(1 - np.exp(logit - max_logit - FLAGS.epsilon))
            F_hat = F - F*FLAGS.gamma_percentile
            p = sigmoid(F)
            prob = np.random.uniform(0, 1)
            if 0.8 < p:
                save_img = (fake_imgs[idx] + 1)*127.5
                save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite("./sampled_results/result_%02d.jpg"% len(accepted_samples), save_img)
                print ("Sampled : %d/%d"%(len(accepted_samples), FLAGS.total_samples))
                print ("maxM : %.4f prob : %.4f p : %.4f F : %.4f F_hat : %.4f"%(max_M, prob, p, F, F_hat))
                accepted_samples.append(fake_imgs[idx])
    img_tile(0, np.array(accepted_samples))
    print ("Done.")


def main():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        print ("Init. model...")
        model = SAGAN()
        sample(model)
        
main()

