import csv
import numpy as np

logs_path = '../resources/logs/conv_channels/'
filename = 'conv_channels_2'
floating_point = False

if __name__ == '__main__':
    path = logs_path + filename + '.csv'
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        reader = filter(lambda x: x[2] == ('fp32' if floating_point else 'int8'), reader)
        inference_time = np.array([[0] * 65] * 20, dtype=np.float32)
        # setup_time = np.array([[0] * 50] * 20, dtype=np.float32)
        # weights_mem = np.array([[0] * 50] * 20, dtype=np.float32)
        # buffer_mem = np.array([[0] * 50] * 20, dtype=np.float32)
        print(np.shape(inference_time))
        for line in reader:
            print("Convs: {}\t\tBatch: {}\t\tTime:{}".format(line[3], line[4], line[6]))
            inference_time[int(line[3]) - 1, int(line[4]) - 1] = line[6]
            # setup_time[int(line[3]) - 1, int(line[5]) - 1] = line[7]
            # weights_mem[int(line[3]) - 1, int(line[5]) - 1] = line[8]
            # buffer_mem[int(line[3]) - 1, int(line[5]) - 1] = line[9]

        for a in inference_time:
            print(a)

    files = ['inf_time.csv', 'setup_time.csv', 'w_mem.csv', 'b_mem.csv']
    path = logs_path
    with open(path + ('fp32_' if floating_point else 'int8_') + files[0], 'w') as inf:  # , open(path + files[1], 'w') as setup, \
            # open(path + files[2], 'w') as weight, open(path + files[3], 'w') as buffer:
        inference_writer = csv.writer(inf)
        # setup_writer = csv.writer(setup)
        # weights_writer = csv.writer(weight)
        # buffer_writer = csv.writer(buffer)

        inference_writer.writerows(inference_time)
        # setup_writer.writerows(setup_time)
        # weights_writer.writerows(weights_mem)
        # buffer_writer.writerows(buffer_mem)

