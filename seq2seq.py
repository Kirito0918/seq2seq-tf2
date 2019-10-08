from module.config import Config
from module.model import Seq2seq
import tensorflow.keras as keras

config = Config()

def main():

    model = Seq2seq(config)
    model.summary()
    keras.utils.plot_model(model, 'mnist_model.png')
    keras.utils.plot_model(model, 'model_info.png', show_shapes=True)


if __name__ == '__main__':
    main()