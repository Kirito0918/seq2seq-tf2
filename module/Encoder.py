import tensorflow as tf
import tensorflow.keras as keras

class Encoder(keras.layers.Layer):

    def __init__(self, rnn_type,  # rnn类型
                 input_size,
                 output_size,
                 num_layers,  # rnn层数
                 bidirectional=False):
        super(Encoder, self).__init__()
        assert rnn_type in ['GRU', 'LSTM']
        if bidirectional:
            assert output_size % 2 == 0

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        units = int(output_size / self.num_directions)

        if rnn_type == 'GRU':
            rnnCell = [getattr(keras.layers, 'GRUCell')(units) for _ in range(num_layers)]
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(units) for _ in range(num_layers)]

        self.rnn = keras.layers.RNN(rnnCell, input_shape=(None, None, input_size),
                                    return_sequences=True, return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        if bidirectional:
            self.rnn = keras.layers.Bidirectional(self.rnn, merge_mode='concat')

        self.bidirectional = bidirectional


    def __call__(self, input):  # [batch, timesteps, input_dim]

        outputs = self.rnn(input)

        output = outputs[0]
        states = outputs[1:]

        print(outputs)
        print(len(outputs))
        print(len(states))

        # if self.bidirectional:
        #     for idx, layer_state in enumerate(states):
        #         print(layer_state)
        #         if self.rnn_type == 'LSTM':
        #             state_hf, state_cf, state_hb, state_cb = layer_state
        #             state_h = tf.concat([state_hf, state_hb], 1)
        #             state_c = tf.concat([state_cf, state_cb], 1)
        #             states[idx] = [state_h, state_c]
        #
        #         else:  # 'GRU'
        #             state_hf, state_cf = layer_state
        #             state_h = tf.concat([state_hf, state_hb], 1)
        #             states[idx] = [state_h]

        # output: [batch, seq, dim*dircetions]
        # states: [num_layers] * (state): tensor(batch, dim*dircetions)
        return output, states







