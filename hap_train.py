"""Training the hap model."""

import os
import time
from datetime import datetime

import hap_model
import tensorflow as tf
# for timeline monitor
from tensorflow.python.client import timeline

# Disable some CPU warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/hap_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

def train():
    """Train hap_model for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        examples, labels = hap_model.inputs()

        logits = hap_model.inference(examples)

        loss = hap_model.loss(logits, labels)

        train_op = hap_model.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                if self._step == -1:
                    print("time waiting for the first time\n")
                    time.sleep(5)

                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))


        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
          while not mon_sess.should_stop():
              # # timeline monitor
              # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
              # run_metadata = tf.RunMetadata()

              # mon_sess.run(train_op,
              #              options=run_options,
              #              run_metadata=run_metadata)

              mon_sess.run(train_op)


              # # Create the Timeline object, and write it to a json
              # tl = timeline.Timeline(run_metadata.step_stats)
              # ctf = tl.generate_chrome_trace_format()
              # with open('timeline_V2.json', 'w') as f:
              #     # print("write timeline\n")
              #     f.write(ctf)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()