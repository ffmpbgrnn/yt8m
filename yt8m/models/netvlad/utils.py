import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest

def transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t

def infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype
