import tensorflow as tf
import tensorflow.keras as keras
from Embedding import Embedding
from Encoder import Encoder
from Decoder import Decoder


class Seq2seq(keras.Model):

    def __init__(self, config, embed=None):
        super(Seq2seq, self).__init__()

        self.config = config

        self.embedding = Embedding(config.num_vocab, config.embedding_size, embed)

        self.encoder = Encoder(config.ende_rnn_type,
                               config.embedding_size,
                               config.ende_output_size,
                               config.ende_num_layers,
                               config.encoder_bidirectional)

        self.decoder = Decoder(config.ende_rnn_type,
                               config.embedding_size,
                               config.ende_output_size,
                               config.ende_num_layers)

        self.projector = keras.layers.Dense(config.num_vocab)


    def __call__(self, input, inference=False):

        if not inference:  # 训练

            posts = input['posts']  # [batch, len_encoder]
            responses = input['responses']  # [batch, len_decoder+1]

            decoder_len = tf.shape(responses)[1] - 1  # [batch, len_decoder]

            encoder_input = self.embedding(posts)  # [batch, len_encoder, embedding_size]
            decoder_input = self.embedding(responses)[:, :-1, :]  # [batch, len_decoder, embedding_size]

            # encoder_states: [num_layers] * tensor(batch, output_size)
            _, encoder_states = self.encoder(encoder_input)

            ta = tf.TensorArray(size=0, dtype=tf.int64, dynamic_size=True)

            decoder_input = ta.unstack(tf.transpose(decoder_input, [1, 0, 2]))  # decoder_len * [batch, embedding_size]

            outputs = []

            for timestep in range(decoder_len.numpy()):

                if timestep == 0:

                    states = encoder_states

                # output: [batch, 1, dim]
                output, states = self.decoder(tf.expand_dims(decoder_input.read(timestep), 1), states)

                outputs.append(tf.transpose(output, [1, 0, 2]))  # [1, batch, dim]

            outputs = tf.concat(outputs, 0)  # [len_decoder, batch, dim]
            outputs = tf.transpose(outputs, [1, 0, 2])  # [batch, len_decoder, dim]

            outputs_prob = self.projector(outputs)  # [batch, len_decoder, num_vocab]

            return outputs_prob






































