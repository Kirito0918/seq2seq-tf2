import tensorflow as tf
import tensorflow.keras as keras
from Embedding import Embedding
from Encoder import Encoder
from Decoder import Decoder


class Seq2seq(keras.Model):

    def __init__(self, config, embeds=None):
        super(Seq2seq, self).__init__()

        self.config = config

        self.embedding = Embedding(config.num_vocab, config.embedding_size, weights=embeds)

        self.encoder = Encoder(config.ende_rnn_type,
                               config.embedding_size,
                               config.ende_output_size,
                               config.ende_num_layers,
                               config.encoder_bidirectional)

        self.decoder = Decoder(config.ende_rnn_type,
                               config.embedding_size,
                               config.ende_output_size,
                               config.ende_num_layers)

        self.projector = keras.layers.Dense(config.num_vocab, input_shape=(None, None, config.ende_output_size))

        self.softmax = keras.layers.Softmax(-1)


    def __call__(self, input, inference=False, max_len=60):

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

            for timestep in range(decoder_len):

                if timestep == 0:

                    states = encoder_states

                # output: [batch, 1, dim]
                output, states = self.decoder(tf.expand_dims(decoder_input.read(timestep), 1), states)

                outputs.append(output)

            outputs = tf.concat(outputs, 1)  # [batch, len_decoder, dim]

            outputs_prob = self.projector(outputs)  # [batch, len_decoder, num_vocab]
            outputs_prob = self.softmax(outputs_prob)

            return outputs_prob

        else:  # 测试

            posts = input['posts']  # [batch, len_encoder]

            batch_size = tf.shape(posts)[0]

            encoder_input = self.embedding(posts)  # [batch, len_encoder, embedding_size]

            # encoder_states: [num_layers] * tensor(batch, output_size)
            _, encoder_states = self.encoder(encoder_input)

            outputs = []
            done = tf.cast(tf.zeros([batch_size]), dtype=tf.bool)
            first_input = self.embedding(tf.convert_to_tensor([1] * self.config.start_id, dtype=tf.int32))  # [batch, embedding_size]

            for timestep in range(max_len):

                if timestep == 0:
                    states = encoder_states
                    decoder_input = first_input  # [batch, embedding_size]

                # output: [batch, 1, dim]
                output, states = self.decoder(tf.expand_dims(decoder_input, 1), states)

                outputs.append(output)  # [batch, 1, dim]

                output_prob = self.projector(output)  # [batch, 1, num_vocab]
                output_prob = self.softmax(output_prob)  # [batch, 1, num_vocab]
                next_input_id = tf.reshape(tf.argmax(output_prob, 2), [-1])  # [batch]

                _done = next_input_id == self.config.end_id
                done = done | _done
                if tf.reduce_sum(tf.cast(done, dtype=tf.int32)) == batch_size:
                    break
                else:
                    decoder_input = self.embedding(next_input_id)  # [batch, embedding_size]

            outputs = tf.concat(outputs, 1)  # [batch, len_decoder, dim]

            outputs_prob = self.projector(outputs)  # [batch, len_decoder, num_vocab]
            outputs_prob = self.softmax(outputs_prob)

            return outputs_prob
