import numpy as np
import tensorflow as tf

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f.tensor_train_base import TensorTrainBase
from t3f import shapes, utils, decompositions, initializers
import math
from Quantize import fw,fa,fBits,fbn_G,fbn_B,fbn_mean,fbn_var,fbn_x


def _validate_input_parameters(is_tensor, shape, **params):
  """Internal function for validating input parameters

  Args:
    is_tensor: bool, determines whether we attempt to construct a TT-tensor or
      a TT-matrix (needed for the correct shape checks).
    shape: array, the desired shape of the generated TT object
    params: optional, possible values:
      batch_size: int, for constructing batches
      tt_rank: array or int, desired TT-ranks
  """

  if is_tensor:
    if len(shape.shape) != 1:
      raise ValueError('shape should be 1d array, got %a' % shape)
    if np.any(shape < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)
  else:
    if len(shape.shape) != 2:
      raise ValueError('shape should be 2d array, got %a' % shape)
    if shape[0].size != shape[1].size:
      raise ValueError('shape[0] should have the same length as shape[1], but'
                       'got %d and %d' % (shape[0].size, shape[1].size))
    if np.any(shape.flatten() < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape.flatten()):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)

  if 'batch_size' in params:
    batch_size = params['batch_size']
    if not isinstance(batch_size, (int, np.integer)):
      raise ValueError('`batch_size` should be integer, got %f' % batch_size)
    if batch_size < 1:
      raise ValueError('Batch size should be positive, got %d' % batch_size)
  if 'tt_rank' in params:
    tt_rank = params['tt_rank']
    if tt_rank.size == 1:
      if not isinstance(tt_rank[()], np.integer):
        raise ValueError('`tt_rank` should be integer, got %f' % tt_rank[()])
    if tt_rank.size > 1:
      if not all(isinstance(tt_r, np.integer) for tt_r in tt_rank):
        raise ValueError('all elements in `tt_rank` should be integers, got'
                         ' %a' % tt_rank)
    if np.any(tt_rank < 1):
      raise ValueError('`tt_rank` should be positive, got %a' % tt_rank)

    if is_tensor:
      if tt_rank.size != 1 and tt_rank.size != (shape.size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape.size + 1, tt_rank.size))
    else:
      if tt_rank.size != 1 and tt_rank.size != (shape[0].size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape[0].size + 1, tt_rank.size))

def matrix_with_random_cores(shape, tt_rank=2, mean=0., stddev=1.,
                             dtype=tf.float32,
                             name='t3f_matrix_with_random_cores'):
  """Generate a TT-matrix of given shape with N(mean, stddev^2) cores.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        matrix_with_random_cores([[2, 2, 2], None])
      and
        matrix_with_random_cores([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    mean: a number, the mean of the normal distribution used for
      initializing TT-cores.
    stddev: a number, the standard deviation of the normal distribution used
      for initializing TT-cores.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.concatenate([[1], tt_rank, [1]])

  tt_rank = tt_rank.astype(int)
  tt_cores = [None] * num_dims
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (tt_rank[i], shape[0][i], shape[1][i],
                         tt_rank[i + 1])
      tt_cores[i] = tf.random_normal(curr_core_shape, mean=mean, stddev=stddev,
                                     dtype=dtype)
      # Quantization
      if i==0 or i==num_dims-1: 
        tt_cores[i] = fBits(tt_cores[i], 8)
      else:
        tt_cores[i] = fBits(tt_cores[i], 8)
      #print('!!!!tt_cores!! after:', tt_cores[i])

    return TensorTrain(tt_cores, shape, tt_rank)

def random_matrix(shape, tt_rank=2, mean=0., stddev=1.,
                  dtype=tf.float32, name='t3f_random_matrix'):
  """Generate a random TT-matrix of the given shape with given mean and stddev.

  Entries of the generated matrix (in the full format) will be iid and satisfy
  E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
  in fact not Gaussian.

  In the current implementation only mean 0 is supported. To get
  a random_matrix with specified mean but tt_rank greater by 1 you can call
  x = t3f.random_matrix(shape, tt_rank, stddev=stddev)
  x = mean * t3f.ones_like(x) + x

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        random_matrix([[2, 2, 2], None])
      and
        random_matrix([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    mean: a number, the desired mean for the distribution of entries.
    stddev: a number, the desired standard deviation for the distribution of
      entries.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)

  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.concatenate([[1], tt_rank, [1]])

  tt_rank = tt_rank.astype(int)
  var = np.prod(tt_rank)

  # Empirically entries of a TT tensor with cores initialized from N(0, 1)
  # will have variances np.prod(tt_rank) and mean 0.
  # We scale each TT-core to obtain the desired stddev

  cr_exponent = -1.0 / (2 * num_dims)
  var = np.prod(tt_rank ** cr_exponent)
  core_stddev = stddev ** (1.0 / num_dims) * var
  with tf.name_scope(name):
    tt = matrix_with_random_cores(shape, tt_rank=tt_rank, stddev=core_stddev,
                                  dtype=dtype)

  if np.abs(mean) < 1e-8:
    return tt
  else:
    raise NotImplementedError('non-zero mean is not supported yet')


def glorot_initializer(shape, tt_rank=2, dtype=tf.float32,
                       name='t3f_glorot_initializer'):
  """Constructs a random TT matrix with entrywise variance 2.0 / (n_in + n_out)

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        glorot_initializer([[2, 2, 2], None])
      and
        glorot_initializer([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """

  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)
  n_in = np.prod(shape[0])
  n_out = np.prod(shape[1])
  lamb = 2.0 / (n_in + n_out)

  with tf.name_scope(name):
    return random_matrix(shape, tt_rank=tt_rank, stddev=np.sqrt(lamb),
                         dtype=dtype)
def truncated_initializer(shape, tt_rank=2, dtype=tf.float32,
                       name='t3f_glorot_initializer'):
  """Constructs a random TT matrix with entrywise variance 2.0 / (n_in + n_out)

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        glorot_initializer([[2, 2, 2], None])
      and
        glorot_initializer([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """

  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)
  n_in = np.prod(shape[0])
  n_out = np.prod(shape[1])
  lamb = 1.0/math.sqrt(float(n_in))

  with tf.name_scope(name):
    return random_matrix(shape, tt_rank=tt_rank, stddev=np.sqrt(lamb),
                         dtype=dtype)

def get_variable(name,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 validate_shape=True):
  """Returns TensorTrain object with tf.Variables as the TT-cores.

  Args:
    name: The name of the new or existing TensorTrain variable.
      Used to name the TT-cores.
    dtype: Type of the new or existing TensorTrain variable TT-cores (defaults
      to DT_FLOAT).
    initializer: TensorTrain or TensorTrainBatch, initializer for the variable
      if one is created.
    regularizer: A (TensorTrain -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If True also add the variable to the graph collection
      GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
    collections:  List of graph collections keys to add the Variables
      (underlying TT-cores). Defaults to [GraphKeys.GLOBAL_VARIABLES]
      (see tf.Variable).
    caching_device: Optional device string or function describing where
      the Variable should be cached for reading. Defaults to the Variable's
      device. If not None, caches on another device. Typical use is to cache
      on the device where the Ops using the Variable reside, to deduplicate
      copying through Switch and other conditional statements.
    validate_shape: If False, allows the variable to be initialized with a value
      of unknown shape. If True, the default, the shape of initial_value must be
      known.

  Returns:
    The created or existing `TensorTrain` object with tf.Variables TT-cores.

  Raises:
    `ValueError`: when creating a new variable and shape is not declared, when
      violating reuse during variable creation, or when initializer dtype and
      dtype don't match. Reuse is set inside variable_scope.
  """
  # TODO: support validate shape: check that the tensor dimensions are correct,
  # but ignore the ranks.
  # TODO: add validate ranks flag.

  reuse = tf.get_variable_scope().reuse
  if not reuse and initializer is None:
    raise ValueError('Scope reuse is False and initializer is not provided.')

  variable_cores = []

  if reuse and not utils.in_eager_mode():
    # Find an existing variable in the collection.
    path = tf.get_variable_scope().name
    if path != '' and path[-1] != '/':
      path += '/'
    path += name

    found_v = None
    for v in tf.get_collection('TensorTrainVariables'):
      if v.name == path:
        found_v = v
        break
    if found_v is None:
      raise ValueError('ValueError: Variable %s does not exist, or was not '
                       'created with t3f.get_tt_variable(). Did you mean to '
                       'set reuse=None in VarScope?' % name)
    with tf.variable_scope(name):
      # Try to get the first core through tf.get_variable to check that we don't
      # violate reuse: it will raise a ValueError otherwise.
      tf.get_variable('core_0', dtype=dtype)
    return found_v
  else:
    # Create new variable.
    with tf.variable_scope(name):
      num_dims = initializer.ndims()
      for i in range(num_dims):
        curr_core_var = tf.get_variable('core_%d' % i,
                                        initializer=initializer.tt_cores[i],
                                        dtype=dtype, trainable=trainable,
                                        collections=collections,
                                        caching_device=caching_device)
        variable_cores.append(curr_core_var)
    if isinstance(initializer, TensorTrain):
      v = TensorTrain(variable_cores, initializer.get_raw_shape(),
                      initializer.get_tt_ranks(),
                      convert_to_tensors=False)
    else:
      v = TensorTrainBatch(variable_cores, initializer.get_raw_shape(),
                           initializer.get_tt_ranks(), initializer.batch_size,
                           convert_to_tensors=False)

    # Add the create TensorTrain object into a collection so that we can
    # retrieve it in the future by get_tt_variable('name').
    tf.add_to_collection('TensorTrainVariables', v)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with tf.name_scope(name + "/Regularizer/"):
        loss = regularizer(v)
      if loss is not None:
        tf.logging.vlog(1, "Applied regularizer to %s and added the result %s "
                        "to REGULARIZATION_LOSSES.", v.name, loss.name)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v

def renormalize_tt_cores(tt, epsilon=1e-8, name='t3f_renormalize_tt_cores'):
    """Renormalizes TT-cores to make them of the same Frobenius norm.

    Doesn't change the tensor represented by `tt` object, but renormalizes the
    TT-cores to make further computations more stable.

    Args:
      tt: `TensorTrain` or `TensorTrainBatch` object
      epsilon: parameter for numerical stability of sqrt
      name: string, name of the Op.

    Returns:
      `TensorTrain` or `TensorTrainBatch` which represents the same
      tensor as tt, but with all cores having equal norm. In the batch
      case applies to each TT in `TensorTrainBatch`.
    """
    # TODO: bad way to check if batch or not.
    with tf.name_scope(name):
      epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
      if isinstance(tt, TensorTrain):
        new_cores = []
        running_log_norm = 0
        core_norms = []
        for core in tt.tt_cores:
          cur_core_norm = tf.sqrt(tf.maximum(tf.reduce_sum(core ** 2), epsilon))
          core_norms.append(cur_core_norm)
          running_log_norm += tf.log(cur_core_norm)

        running_log_norm = running_log_norm / tt.ndims()
        fact = tf.exp(running_log_norm)
        for i, core in enumerate(tt.tt_cores):
          new_cores.append(core * fact / core_norms[i])
        return TensorTrain(new_cores)
      else:
        sz = (tt.batch_size,) + (len(tt.tt_cores[0].shape) - 1) * (1,)
        running_core_log_norms = tf.zeros(sz, dtype=tt.dtype)
        ax = np.arange(len(tt.tt_cores[0].shape))[1:]
        fact_list = []
        for core in tt.tt_cores:
          cur_core_norm_sq = tf.reduce_sum(core**2, axis=ax, keepdims=True)
          cur_core_norm = tf.sqrt(tf.maximum(epsilon, cur_core_norm_sq))
          fact_list.append(cur_core_norm)
          running_core_log_norms += tf.math.log(cur_core_norm)

        new_cores = []
        exp_fact = tf.exp(running_core_log_norms / tt.ndims())
        for i, core in enumerate(tt.tt_cores):
          new_cores.append(tf.multiply(core, exp_fact / fact_list[i]))

        return TensorTrainBatch(new_cores)


def full(tt, name='t3f_full'):
  """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object.
    name: string, name of the Op.

  Returns:
    tf.Tensor.
  """
  with tf.name_scope(name):
    if isinstance(tt, TensorTrainBatch):
      # Batch of Tensor Trains.
      print('--------going to full tt batch--------')
      return _full_tt_batch(tt)
    else:
      # TensorTrain object (not batch).
      print('--------going to full tt--------')
      return _full_tt(tt)


def _full_tt(tt):
  """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrain` object.

  Returns:
    tf.Tensor.
  """
  num_dims = tt.ndims()
  ranks = shapes.lazy_tt_ranks(tt)
  shape = shapes.lazy_shape(tt)
  raw_shape = shapes.lazy_raw_shape(tt)
  
  quan_core_list = []
  for i in range(num_dims):
    print('FP core:', tt.tt_cores[i])
    if i == 0 or i == num_dims-1:
      quan_core_list.append(fw(tt.tt_cores[i]))
    else:
      quan_core_list.append(fw(tt.tt_cores[i]))
    print('8bit core:', quan_core_list[i])
  res = quan_core_list[0]
  # Quan first core
  for i in range(1, num_dims):
    res = tf.reshape(res, (-1, ranks[i]))
    curr_core = tf.reshape(quan_core_list[i], (ranks[i], -1))
    res = tf.matmul(res, curr_core)
    print('core multi FP: ', res)
    # Quan mult cores
    res = fw(res)
    print('core multi 8bit: ', res)
  if tt.is_tt_matrix():
    intermediate_shape = []
    for i in range(num_dims):
      intermediate_shape.append(raw_shape[0][i])
      intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, intermediate_shape)
    transpose = []
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, shape)
  else:
    return tf.reshape(res, shape)


def _full_tt_batch(tt):
  """Converts a TensorTrainBatch into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrainBatch` object.

  Returns:
    tf.Tensor.
  """
  num_dims = tt.ndims()
  ranks = shapes.lazy_tt_ranks(tt)
  shape = shapes.lazy_shape(tt)
  raw_shape = shapes.lazy_raw_shape(tt)

  res = tt.tt_cores[0]
  batch_size = shapes.lazy_batch_size(tt)
  for i in range(1, num_dims):
    res = tf.reshape(res, (batch_size, -1, ranks[i]))
    curr_core = tf.reshape(tt.tt_cores[i], (batch_size, ranks[i], -1))
    res = tf.einsum('oqb,obw->oqw', res, curr_core)
    res = fw(res)
    print("full_res: ", res)
  if tt.is_tt_matrix():
    intermediate_shape = [batch_size]
    for i in range(num_dims):
      intermediate_shape.append(raw_shape[0][i])
      intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, intermediate_shape)
    transpose = [0]
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i + 1)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i + 1)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, shape)
  else:
    return tf.reshape(res, shape)
