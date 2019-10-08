import tensorflow as tf
import numpy as np
from module.Embedding import Embedding
from module.Encoder import Encoder
# from tensorflow.python.ops import control_flow_util
# control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

# num_vocab = 3
# embedding_size = 5
# batch_size = 2
# seq = 3
# lstm_units = 10
# num_layers = 2
# bidirectional = True

# 嵌入层测试
weights = [np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], dtype=np.float64)]
embedding = Embedding(3, 5, weights)  # (num_vocab, embedding_size)

word_id = tf.convert_to_tensor([[1, 2, 0], [1, 0, 0]], dtype=tf.int64)
word_embed = embedding(word_id)  # [batch, seq, embedding_size]

# print(word_embed)  # [2, 3, 5]

# 编码器测试
encoder = Encoder('LSTM', 10, 2, True)

output, states = encoder(word_embed)

# print('output:', output)
# print('states:', states)

# TensorArray测试
# ta = tf.TensorArray(size=0, dtype=tf.int64, dynamic_size=True)
# ta = ta.unstack(word_id)
# print(ta)





