import os
import tensorflow as tf

# Global constants describing the Hap data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100

# Dimension of a single example
# SAMP_FREQ = 5000
# MILISEC_PER_WINDOW = 100
NUM_REC_PER_WINDOW = 100  # SAMP_FREQ * MILI_SEC_PER_WINDOW / 1000
WIDTH_MULTIPLIER = 1
CHANNELS = 1


def read_hapdata(filename_queue):

    class HAPExample(object):
        pass
    result = HAPExample()

    result.length = NUM_REC_PER_WINDOW
    result.height = NUM_REC_PER_WINDOW
    result.width = WIDTH_MULTIPLIER
    result.depth = CHANNELS

    # Read an example
    reader = tf.TextLineReader()
    result.key, value = reader.read(filename_queue, name='example_string_input')

    record_defaults = [[1]] + [[1.0] for _ in range(result.length * result.depth)]

    reader_output = tf.decode_csv(value, record_defaults=record_defaults)

    result.example = tf.reshape(tf.stack(reader_output[1:]),
        [result.height, result.width, result.depth])

    result.label = tf.reshape(reader_output[0], [-1])

    return result


def _generate_example_and_label_batch(example, label,
                                      min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of examples and labels.

  Args:
    example: 3-D Tensor of [NUM_REC_PER_WINDOW, 1, CHANNELS] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    examples: Examples. 4D tensor of [batch_size, NUM_REC_PER_WINDOW, 1, CHANNELS] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # num_preprocess_threads = 16
    if shuffle:
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label],
            batch_size=batch_size,
            # num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        example_batch, label_batch = tf.train.batch(
            [example, label],
            batch_size=batch_size,
            # num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])

    # Summary
    audio = tf.reshape(example_batch, [batch_size,-1])
    tf.summary.image('records', audio)
    return example_batch, label_batch


def some_process(foo):
    return foo


def train_inputs(batch_size, data_dir=""):
    """Construct inputs for training.
    Args:
        data_dir: Path to data directory.
        batch_size: Number of pieces per batch.

    Returns:
        example_batch: examples for one epoch of training.
        label_batch: labels corresponding to the examples.
    """
    # Create filename list and corresponding labels
    filenames = [os.path.join(data_dir, 'train_data.csv')]
    # Check if there are enough files compare to the number of classes
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_hapdata(filename_queue)

    batch_list = [read_input.example, read_input.label]

    #  generating the examples for one epoch
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    # min_queue_examples=32
    print('Filling queue with %d haptic examples before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # # *Test
    # print("batch_threads=", batch_threads)
    # print("batch_size=",batch_size,"capacity=", 16 * batch_size,"min_after_dequeue=", batch_size)

    example_batch, label_batch = tf.train.shuffle_batch(batch_list,
                                                         batch_size=batch_size,
                                                         capacity=min_queue_examples + 3  * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         name='train_inputs_generator',
                                                         num_threads=16)

    # TODO:Summary
    tf.summary.histogram(name='example_batch',
                         values=example_batch)

    return example_batch, tf.reshape(label_batch, [batch_size])


def eval_input(data_dir, batch_size):
    """Construct inputs for evaluation.
        Args:
            data_dir: Path to data directory.
            batch_size: Number of pieces per batch.

        Returns:
            example_batch: examples for one epoch of training.
            label_batch: labels corresponding to the examples.
        """
    # Create filename list and corresponding labels
    filenames = [os.path.join(data_dir, 'eval_data.csv')]
    # Check if there are enough files compare to the number of classes
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_hapdata(filename_queue)

    batch_list = [read_input.example, read_input.label]

    #  generating the examples for one epoch
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)
    # min_queue_examples=32
    print('Filling queue with %d haptic examples before starting to evaluation. '
          'This will take a few minutes.' % min_queue_examples)

    example_batch, label_batch = tf.train.shuffle_batch(batch_list,
                                                        batch_size=batch_size,
                                                        capacity=min_queue_examples + 3 * batch_size,
                                                        min_after_dequeue=min_queue_examples,
                                                        name='train_inputs_generator',
                                                        num_threads=16)

    return example_batch, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':

    out_put = train_inputs(10)

    # create session
    sess = tf.Session()
    # initial session
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # create queue handler
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    print(sess.run(out_put))