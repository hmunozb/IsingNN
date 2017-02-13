
from ._util import *
from _model_constants import *

import tensorflow as tf
import os
import glob


def make_queue(filenames):
    return tf.train.string_input_producer(filenames)


def _tc_class_tensor(t, tc, a, b):
    """
    Tensor expression:
        if t < tc return a, else b
    :param t:
    :param tc:
    :param a:
    :param b:
    :return:
    """
    return tf.select(
        tf.less(t, tc, name='CMP_LT'+str(tc)),
        tf.constant(a, dtype=tf.int32),
        tf.constant(b, dtype=tf.int32),
        name='SELECT_LT'+str(tc))


class IsingFileConfig:
    def __init__(self, ising_config: dict):
        self.L = ising_config['L']
        self.dim = ising_config['dim']
        self.replicas_per_instance = ising_config['reps_per_inst']
        self.instances_per_example = ising_config['insts_per_exmp']
        self.batch_size = ising_config['batch_size']
        self.num_threads = ising_config.get('threads', 1)


class IsingFileReader:
    """
    An IFR expects a filename queue of binary files output from a MC simulation with the following structure
    Header:
        * 00-03 L Ising model length, int32
        * 04-07 K Number of configurations in the file, int32
        * 08-0B t Index of the temperature of the configurations (T_ARR_1), int32
        * 0C-0F T the temperature simulated at (should be T_ARR_1[t]), float32
    Body:
        * 10 - ... A list of K configurations of LxL integers (float32), each of which is +1.0 or -1.0
    """
    def __init__(self, queue,
                 isfc: IsingFileConfig,
                 name="IsingFileReader"):
        self._q = queue
        self._name = name

        self._reader = tf.WholeFileReader(name=self._name)
        self.key, value = self._reader.read(self._q)
        self._in_bytes = tf.decode_raw(value, tf.uint8)

        self.l = tf_int32_from_bytes(self._in_bytes, 0)
        self.k = tf_int32_from_bytes(self._in_bytes, 4)
        self.t_indx = tf_int32_from_bytes(self._in_bytes, 8)
        self.t = tf_float_from_bytes(self._in_bytes, 12)

        self.states = \
            tf.bitcast(
                tf.reshape(
                    tf.slice(self._in_bytes, [16], [-1]),
                    [self.k, -1, 4]),
                tf.float32)
        # shuffle the states
        self.states = tf.random_shuffle(self.states)
        c = isfc.replicas_per_instance
        self.states = self.states[0:c, ...]

        self.example_tensor = [
            self.states,  self.t_indx, self.t]
        self.example_shapes = [
            [c, isfc.L ** isfc.dim], [], []]

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


class IsFRBatch:
    """
    takes the input from an ifr file as a single example, and outputs a shuffled
    batch
    """
    def __init__(self, ifr: IsingFileReader, ifc: IsingFileConfig):
        ipe = ifc.instances_per_example
        self._min_examples = 5*ipe
        self._capacity = self._min_examples + 10 * ipe
        # Note: The shape of an example is [C, N]
        self.state_example_shape = ifr.example_shapes[0]
        # Enqueue the read contents of a single file as a single example
        self.state_batch, self.index_batch, self.temp_batch = \
            tf.train.shuffle_batch(
                ifr.example_tensor,
                batch_size=ipe,  capacity=self._capacity,
                min_after_dequeue=self._min_examples,
                enqueue_many=False,
                shapes=ifr.example_shapes, name="IfrBatch")


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


class ExampleSubClassQueue:
    def __init__(self, filename_pattern, label_lambda, class_queue_index,
                 isfc: IsingFileConfig, subname="SubClassQ"):
        with tf.name_scope(subname):
            # Gather list of matching filenames
            self.filenames = glob.glob(filename_pattern)
            # create a queue of filenames (string tensors)
            self.filename_queue = make_queue(self.filenames)
            # Reader for each file
            # tensor shape read is [R, N]
            self.isfr = IsingFileReader(self.filename_queue, isfc)
            rpi = isfc.replicas_per_instance
            ipe = isfc.instances_per_example
            # Enqueues each read tensor as a single example
            # Shuffles and dequeues a batch tensor of shape [I, R, N]
            # which is treated again a single example
            self.isfbatch = IsFRBatch(self.isfr, isfc)
            capacity = 10*isfc.batch_size
            min_ex = 5*isfc.batch_size
            #self.example_shape = [ipe]+self.isfbatch.state_example_shape
            #enqueue a batch (sample) of instances as a single example
            # Enqueues the example with all appropriate labels
            self.sub_queue = tf.train.shuffle_batch(
                [self.isfbatch.state_batch,
                 self.isfbatch.index_batch[0],
                 self.isfbatch.temp_batch[0],
                 label_lambda(self.isfbatch.temp_batch[0]),
                 tf.constant(class_queue_index, dtype=tf.int32)],
                isfc.batch_size, capacity, min_ex,
                enqueue_many=False
                #shapes=[self.example_shape, [], [], [], []]
            )


class ExampleClassQueue:
    def __init__(self, file_patterns, label_lambda,
                 class_index, isfc: IsingFileConfig, name='ClassQ'):
        with tf.name_scope(name):
            self.subclasses = []
            self.subqueues = []
            for f in file_patterns:
                c = ExampleSubClassQueue(
                    f, label_lambda, class_index, isfc)
                self.subclasses.append(c)
                self.subqueues.append(c.sub_queue)
            self.b_size = isfc.batch_size
            self.queue = tf.train.shuffle_batch_join(
                self.subqueues, self.b_size,
                capacity=10*self.b_size, min_after_dequeue=5*self.b_size,
                enqueue_many=True)


class TheExampleQueue:
    def __init__(self, file_pattern_lists, label_lambdas, isfc: IsingFileConfig):
        self._num_classes = len(file_pattern_lists)
        assert len(file_pattern_lists) == len(label_lambdas)

        self.capacity = isfc.batch_size * 5 * self._num_classes
        self.min_enqueued = isfc.batch_size * 1 * self._num_classes

        self._queue_classes = []
        self._subqueues = []
        for i in range(self._num_classes):
            qc = ExampleClassQueue(
                file_pattern_lists[i], label_lambdas[i], i, isfc)
            self._queue_classes.append(qc)
            self._subqueues.append(qc.queue)
        self.state, self.indices, self.temps,\
            self.label, self.q_index = \
            tf.train.shuffle_batch_join(
                self._subqueues, isfc.batch_size,
                self.capacity, self.min_enqueued, enqueue_many=True
        )


def three_phase_input(data_roots, num_temps_array,
                      tc_triplets, isfc: IsingFileConfig ):
    num_classes = len(num_temps_array)
    assert num_classes == len(data_roots)
    assert num_classes == len(tc_triplets)

    filenames_lists = []
    lambdas = []
    for i in range(num_classes):
        class_filenames = []
        for t in range(num_temps_array[i]):
            class_filenames.append(
                os.path.join(
                    data_roots[i], '*.t{0}.*.bin'.format(t)))
        filenames_lists.append(class_filenames)

        trpl = tc_triplets[i]
        the_lambda = lambda x, v=trpl: _tc_class_tensor(x, v[0], v[1], v[2])
        lambdas.append(the_lambda)
    q = TheExampleQueue(filenames_lists, lambdas, isfc)
    return q


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
                                      transpose_chan=True)
        ising_batch = IsingBatch(ifr, ising_l,
                                 batch_size, min_examples_enqueued, enqueue_many=True)

    return ising_batch


def sg_3d_tbatch_input(data_root, t_range, num_c, batch_size,
                       min_examples_enqueued, ising_l):
    filenames = []
    for t in t_range:
        filenames.append(glob.glob(
            os.path.join(data_root, "**", "*.t%d.*.bin")).__format__(t))
    with tf.name_scope('ExampleReaders'):
        queues = []
        for fns in filenames:
            queues.append(make_queue(fns))
