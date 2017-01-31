
from ._util import *
from _model_constants import *

import tensorflow as tf
import os
import glob


def make_queue(filenames):
    return tf.train.string_input_producer(filenames)


class IsingFileRecordBase:
    """
    Mostly private interface for an IFR class.

    An IFR expects a filename queue of binary files output from a MC simulation with the following structure
    Header:
        * 00-03 L Ising model length, int32
        * 04-07 K Number of configurations in the file, int32
        * 08-0B t Index of the temperature of the configurations (T_ARR_1), int32
        * 0C-0F T the temperature simulated at (should be T_ARR_1[t]), float32
    Body:
        * 10 - ... A list of K configurations of LxL integers (int32), each of which is +1.0 or -1.0

    This class takes care of reading and setting up the header info.
    Derived classes read the body and organize it as needed
    """
    def __init__(self, queue, t_c, dim, name):
        if (dim != 2) and (dim != 3):
            raise ValueError("dim must be 2 or 3")
        self._dim = dim
        self._tc_tensor = tf.constant(t_c, dtype=tf.float32)
        self._q = queue

        self._initialize_record(name)
        self._set_batch_shapes()
        self._assign_labels()
        self._read_states()

    def _initialize_record(self, name):
        self._reader = tf.WholeFileReader(name=name)
        self.key, value = self._reader.read(self._q)
        self._in_bytes = tf.decode_raw(value, tf.uint8)

        self.l = tf_int32_from_bytes(self._in_bytes, 0)
        self.k = tf_int32_from_bytes(self._in_bytes, 4)
        self.t_indx = tf_int32_from_bytes(self._in_bytes, 8)
        self._t = tf_float_from_bytes(self._in_bytes, 12)

    def _assign_labels(self):
        self.indx = tf.fill([self._batch_len], self.t_indx)
        self.temps = tf.fill([self._batch_len], self._t)
        self.label = \
            tf.fill(
                [self._batch_len],
                tf.select(
                    tf.less(self._t, self._tc_tensor),
                    tf.constant(1, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32)))

    def _set_batch_shapes(self):
        print("Warning: method _set_batch_shapes() should have been overridden.")
        self._batch_shape = None
        self._batch_len = None

    def expected_tensor_shape(self, l):
        return None

    def _modify_state(self):
        pass

    def _read_states(self):
        self.states = \
            tf.bitcast(
                tf.reshape(
                    tf.slice(self._in_bytes, [16], [-1]),
                    self._batch_shape),
                tf.float32)
        self._modify_state()
        self.file_batch = [self.states, self.label, self.indx, self.temps]


class IsingFileRecord(IsingFileRecordBase):
    """
    When initialized IsingFileRecord.file_batch is a tuple
    [states, label, index, T] of tensors being input from each file.


    """
    def __init__(self, queue, t_c, dim=2, name="IsingReader"):
        """

        :param queue: a tf string_input_producer of filenames
        """
        IsingFileRecordBase.__init__(self, queue, t_c, dim, name)

        #self._initialize_record(name)
        #self._set_batch_shapes()
        #self._assign_labels(self.k)
        #self._read_states()

    def _set_batch_shapes(self):
        self._batch_len = self.k
        if self._dim == 2:
            self._batch_shape = [self.k, self.l, self.l, 1, 4]
        else:
            self._batch_shape = [self.k, self.l, self.l, self.l, 1, 4]

    def expected_tensor_shape(self, l):
        if self._dim == 2:
            return [[l, l, 1], [], [], []]
        else:
            return [[l, l, l, 1], [], [], []]
    # def _initialize_record(self, name):
    #     self._in_bytes = tf.WholeFileReader(name=name)
    #     self.key, value = self._in_bytes.read(self._q)
    #     record_bytes = tf.decode_raw(value, tf.uint8)
    #
    #     self.l = tf_int32_from_bytes(record_bytes, 0)
    #     self.k = tf_int32_from_bytes(record_bytes, 4)
    #     t_indx = tf_int32_from_bytes(record_bytes, 8)
    #     self._t = tf_float_from_bytes(record_bytes, 12)
    #     self.indx = tf.fill([self.k], t_indx)
    #     self.temps = tf.fill([self.k], self._t)


class GroupingIsingFileRecord(IsingFileRecordBase):
    def __init__(self, queue, partition_size, t_c,
                  dim=2, name="IsingReader", transpose_chan=True):
        self._part_sz = partition_size
        self._transposed = transpose_chan
        self._exm_shape = None
        # IsingFileRecordBase.__init__(self, queue, t_c, dim, name)
        self._dim = dim
        self._tc_tensor = tf.constant(t_c, dtype=tf.float32)
        self._q = queue
        # read header: l, k
        self._initialize_record(name)


        # override the shape that the 1d input will be read as
        #self.k  # will change in mod state
        if self._dim == 2:
            self._batch_shape = \
                [self.k,  # self._part_sz,
                 self.l, self.l, 1, 4]
        else:
            self._batch_shape = \
                [self.k,  # self._part_sz,
                 self.l, self.l, self.l, 1, 4]

        self.states = \
            tf.bitcast(
                tf.reshape(
                    tf.slice(self._in_bytes, [16], [-1]),
                    self._batch_shape),
                tf.float32)
        #transform from -1/1 to 0/1 representation
        #self.states = tf.minimum(self.states, 0)
        # shuffle the states
        self.states = tf.random_shuffle(self.states)
        #self._snap = tf.mul(self.states[0], self.states[1])
        #if self._dim == 3:
         #   self._snap = tf.reduce_mean(self._snap, axis=-2)
        #self._snap = tf.expand_dims(self._snap, axis=0)
        #tf.summary.image('glass', self._snap, max_outputs=20)
        # determine how many can be grouped together
        self._batch_len = tf.floor_div(self.k, self._part_sz)
        take_len = tf.mul(self._batch_len, self._part_sz)
        # excise as many states as possible
        # then reshape and transpose if needed
        if self._dim == 2:
            self.states = tf.slice(
                self.states, [0, 0, 0, 0], [take_len, -1, -1, -1])
            self.states = tf.reshape(self.states,
                [self._batch_len, self._part_sz, self.l, self.l, 1])
            if self._transposed:
                self.states = tf.transpose(
                    self.states, [0, 4, 2, 3, 1])
        else:
            self.states = tf.slice(
                self.states, [0, 0, 0, 0, 0], [take_len, -1, -1, -1, -1])
            self.states = tf.reshape(self.states,
                [self._batch_len, self._part_sz, self.l, self.l, self.l, 1])
            if self._transposed:
                self.states = tf.transpose(
                    self.states, [0, 5, 2, 3, 4, 1])
        #finally squeeze into final dims
        self._assign_labels()
        if self._transposed:
            self.states = tf.squeeze(self.states, axis=[1])
        else:
            self.states = tf.squeeze(self.states, axis=[-1])
        #self.label = tf.select(
         #           tf.less(self._t, self._tc_tensor),
          #          tf.constant(1, dtype=tf.int32),
           #         tf.constant(0, dtype=tf.int32))
        self.file_batch = [
            self.states, self.label, self.indx, self.temps]

    def _modify_state(self):
        # shuffle all states
        self.states = tf.random_shuffle(self.states)
        # slice off whatever won't fit in the the reshape
        self.num_slices = tf.floor_div(self.k, self._part_sz)

        if self._dim == 3:
            self.states = tf.slice(
                self.states, [0, 0, 0, 0, 0], [])
            self.states = tf.transpose(
                self.states, [0, 5, 2, 3, 4, 1])

    def expected_tensor_shape(self, l):
        # the tensor state a shuffle queue should expect
        if self._dim == 2:
            if self._transposed:
                self._exm_shape = [l, l, self._part_sz]
            else:
                self._exm_shape = [self._part_sz, l, l]
        else:
            if self._transposed:
                self._exm_shape = [l, l, l, self._part_sz]
            else:
                self._exm_shape = [self._part_sz, l, l, l]
        return [self._exm_shape, [], [], []]


class IsingBatch:
    def __init__(self, ifr: IsingFileRecordBase, ising_l,
                 batch_size, min_examples, enqueue_many=True):
        self._min_examples = min_examples
        self._capacity = self._min_examples + 50*batch_size
        self.state, self.label, self.index, self.temp = \
            tf.train.shuffle_batch(
                ifr.file_batch, batch_size, self._capacity, self._min_examples,
                enqueue_many=enqueue_many,
                shapes=ifr.expected_tensor_shape(ising_l), name="IsingBatch")


def ising_2d_input(data_root, batch_size, min_examples_enqueued, ising_l):
    filenames = glob.glob(
        os.path.join(data_root, "**", "*.bin"), recursive=True)
    filename_queue = make_queue(filenames)
    ifr = IsingFileRecord(filename_queue, ISING_2D_T_C, 2)
    ising_batch = IsingBatch(ifr, ising_l,
                             batch_size, min_examples_enqueued)

    return ising_batch


def sg_3d_input(data_root, batch_size, min_examples_enqueued, ising_l):
    filenames = glob.glob(
        os.path.join(data_root, "**", "*.bin"), recursive=True)
    filename_queue = make_queue(filenames)
    ifr = IsingFileRecord(filename_queue, SGEA_3D_T_C, 3)
    ising_batch = IsingBatch(ifr, ising_l,
                             batch_size, min_examples_enqueued)

    return ising_batch


def sg_3d_multi_input(data_root, num_channels, batch_size, min_examples_enqueued, ising_l):
    filenames = glob.glob(
        os.path.join(data_root, "**", "*.bin"), recursive=True)
    with tf.name_scope('ExampleReader'):
        filename_queue = make_queue(filenames)
        ifr = GroupingIsingFileRecord(filename_queue, num_channels, SGEA_3D_T_C, 3,
                                      transpose_chan=False)
        ising_batch = IsingBatch(ifr, ising_l,
                                 batch_size, min_examples_enqueued, enqueue_many=True)

    return ising_batch