# """Evaluation for haptic data with tcpip connection with LabView"""

import socket
import struct
from datetime import datetime

import numpy as np

import hap_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/hap_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('predict_dir', '/tmp/hap_predict',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def connect_to_labview(ip='localhost', port=6340):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)  # IP & Port should in line with LabView server
    try:
        client.connect(server_address)
    except:
        print('Cannot connect to LabView in', server_address)
        return 0

    return  client


def get_data_from_labview(client):
    # Receive data size
    size_raw_data = client.recv(4)  # Receive Size data(int 4bytes)
    size = struct.unpack('!i', size_raw_data)[0]  # unpack from bytes to int, '!' for big-endian coding
    # print('size_raw_data:', size_raw_data, '\nData size', size)

    # Get Data
    str_raw_data = client.recv(size)
    # print('Data size:', size, 'Data value:', str_raw_data)
    data_fmt = '!2i' + str(int(size / 8) - 1) + 'd'
    float_data = struct.unpack(data_fmt, str_raw_data)
    # print(float_data)

    # Change Format to (points,1,channels)
    n_points = float_data[0]
    n_channels = float_data[1]
    float_data_array = np.array(float_data[2:2 + n_points], dtype=np.float32).reshape(1, n_points, 1, n_channels)
    # float_data_array = float_data_array.reshape(n_points, 1, n_channels)
    return float_data_array


def predict(logits):
    #todo: prediction of class
    return logits



def eval(saver, summary_writer, op_todo, dict_fn, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint restored at %s" % datetime.now())
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/hap_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

            while True:
                predictions = sess.run([op_todo],feed_dict=dict_fn())
                print(predict(predictions))
            #
            # summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op))
            #
            # summary.value.add(tag='Precision', simple_value=precision)
            # summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():

    with tf.Graph().as_default() as g:
        examples = tf.placeholder(tf.float32, shape=(1, 100, 1, 1))

        logits = hap_model.inference(examples)

        variable_averages = tf.train.ExponentialMovingAverage(
            hap_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.predict_dir, g)

        # Create connection with LabView
        client = connect_to_labview()

        while True:
            # Building dictionary for data feeding
            dict_fn = lambda : {examples:get_data_from_labview(client)}

            # Evaluation
            eval(saver, summary_writer, logits, dict_fn, summary_op)
            if FLAGS.run_once:
                break
            # time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.predict_dir):
        tf.gfile.DeleteRecursively(FLAGS.predict_dir)
    tf.gfile.MakeDirs(FLAGS.predict_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
