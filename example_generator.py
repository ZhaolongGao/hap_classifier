import csv
import os
from random import randint

NUM_CLASSES = 3

def read_csv(src_file):
    """Read source file line by line. Return number of lines and multi-dimensional data list."""
    class EXAMPLERecord(object):
        pass
    records = EXAMPLERecord()

    records.src_temp = []
    records.total_lines = 0

    with open(src_file, newline='') as src:
        src_reader = csv.reader(src, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        for row in src_reader:
            records.src_temp.append(row)
            records.total_lines += 1

    return records

def example_generator(dist_file, window_size, num_examples):
    """generate files containing examples"""
    filename_list = [os.path.join('data', 'type%d.csv' % i)
                     for i in range(NUM_CLASSES)]
    records = [read_csv(file) for file in filename_list]

    with open(dist_file, 'w', newline='') as dist:
        dist_writer = csv.writer(dist, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(num_examples):
            # decide which class
            class_to_read = randint(0, NUM_CLASSES - 1)

            # write the class
            row_to_write = [class_to_read]

            #decide which line to start
            total_lines = records[class_to_read].total_lines
            start_line = randint(0, total_lines - window_size)

            # data to write
            data = records[class_to_read].src_temp[start_line:start_line + window_size]
            for record in data:
                for element in record:
                    row_to_write.append(element)

            dist_writer.writerow(row_to_write)



def generate_data(window_size, n_train, n_eval):

    example_generator('train_data.csv', window_size=window_size, num_examples=n_train)
    example_generator('eval_data.csv', window_size=window_size, num_examples=n_eval)


if __name__ == '__main__':
    generate_data(500, 50000, 10000)