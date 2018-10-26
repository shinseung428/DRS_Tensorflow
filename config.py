import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

#### Train dataset
tf.app.flags.DEFINE_string('data', 'celeb',
                           """Name of the training data""")
tf.app.flags.DEFINE_string('filename_pattern', '*.jpg',
                           """File pattern for input data""")

tf.app.flags.DEFINE_string('ckpt_path', 'models',
                           """Checkpoints path""")
tf.app.flags.DEFINE_string('log_path', 'logs',
                           """Log path""")
tf.app.flags.DEFINE_string('res_path', 'results',
                           """Resultls path""")


tf.app.flags.DEFINE_boolean('continue_train', False,
                            """flag to continue training""")
tf.app.flags.DEFINE_integer('epochs', 50,
                            """training number of epochs""")
tf.app.flags.DEFINE_integer('extra_epochs', 5,
                            """additional training number of epochs""")

#### Input pipeline
tf.app.flags.DEFINE_integer('z_dim', 128,
                            """dimension of input z vector""")
tf.app.flags.DEFINE_integer('img_hw', 64,
                            """Innput image height/width""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Train batch size""")


tf.app.flags.DEFINE_string('optim', 'adam',
                        'Optimizer')
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                            """learning rate""")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.95,
                            """decay rate of learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.0,
                            """beta1 value for adam""")
tf.app.flags.DEFINE_float('beta2', 0.9,
                            """beta2 value for adam""")

tf.app.flags.DEFINE_integer('burnin_samples',50000,
                            """Total number of samples to use""")
tf.app.flags.DEFINE_integer('total_samples', 640,
                            """Total number of samples to gather""")
tf.app.flags.DEFINE_float('gamma_percentile', 0.80,
                          """percentile of gamma used in sampling""")
tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          """epsilon used in sampling""")
