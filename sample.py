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
        fake_imgs, discrim_logits = model.sess.run([model.fake_img, model.fake_sig_logits],
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
        print("Processing BurnIn...%d/%d"%(processed_samples, FLAGS.burnin_samples))
        
     
    print ("Start Sampling...")
    accepted_samples = []
    counter = 0
    rejected_counter = 0
    while counter < FLAGS.total_samples:
        z = np.random.uniform(-1, 1, (FLAGS.batch_size, FLAGS.z_dim))
        fake_imgs, discrim_logits = model.sess.run([model.fake_img, model.fake_sig_logits],
                                                    feed_dict = {model.z : z}
                                                   )
        logits = model.sess.run(tf.reshape(discrim_logits, [-1]))

        batch_ratio = np.exp(logits)
        max_idx = np.argmax(batch_ratio)
        max_ratio = batch_ratio[max_idx]        
        #update max_M if larger M is found
        if max_ratio > max_M:
            max_M = max_ratio
            max_logit = logits[max_idx]

        #calculate F_hat and pass it into sigmoid
        # set gamma dynamically (80th percentile of F)
        Fs = logits - max_logit - np.log(1 - np.exp(logits - max_logit - FLAGS.epsilon))
        gamma = np.percentile(Fs, FLAGS.gamma_percentile)
        F_hat = Fs - gamma
        acceptance_prob = sigmoid(F_hat)
        
        for idx, sample in enumerate(fake_imgs):
            probability = np.random.uniform(0, 1)
            print ('prob : %.4f p : %.4f F_hat : %.4f F : %.4f'%(probability, acceptance_prob[idx], F_hat[idx], Fs[idx]))
            if probability <= acceptance_prob[idx]:
                save_img = (sample + 1)*127.5
                save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite("./sampled_results/result_%d_%.4f.jpg"%(counter, acceptance_prob[idx]), save_img)
                print ("Sampled : %d/%d"%(counter, FLAGS.total_samples))
                accepted_samples.append(sample)
                counter += 1

            if len(accepted_samples) == 64:
                img_tile(np.random.randint(0,999), np.array(accepted_samples))
                accepted_samples = []
            
    print ("Done.")


def main():
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        print ("Init. model...")
        model = SAGAN()
        sample(model)
        
main()

