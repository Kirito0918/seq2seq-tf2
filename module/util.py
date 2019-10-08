import tensorflow as tf

def prepare_states(rnn_type,
                   bidirectional,
                   states):
    assert rnn_type in ['GRU', 'LSTM']

    new_states = []

    if bidirectional:
        if rnn_type == 'LSTM':

            for (state_hf, state_cf, state_hb, state_cb) in states:  # [batch, dim]

                state_h = tf.concat([state_hf, state_hb], 1)
                state_c = tf.concat([state_cf, state_cb], 1)

                state = (state_h, state_c)

                new_states.append(state)

        else:  # gru

            for (state_hf, state_hb) in states:

                state_h = tf.concat([state_hf, state_hb], 1)

                new_states.append(state_h)

    else:

        new_states = states

    return new_states






