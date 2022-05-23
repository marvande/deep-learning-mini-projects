"""test

    This script allows the user to evaluate (train and test) deep learning models
    and to save the plot of those evaluation in .png after the evaluation
    is done.

        * evaluate - call the evaluate function passed as paremeters for each model
          in models
        * main - the main function of the script
"""

# evaluation methods
import accuracy_loss_eval
import accuracy_ciphers_weighted_loss_eval

# models
from basic_siamese_cnn_aux_model import Conv_basic
from deep_siamese_cnn_model import SiameseConvNet2
from siamese_rnn_model import ImageRNN
from siamese_rnn_2_model import ImageRNN_2
from siamese_conv_relu_drop2d_shared_model import SiameseConvNet
from siamese_conv_relu_model import SiameseConvNoSharingNet
from siamese_linear_relu_model import SiameseNoSharingNet
from siamese_linear_relu_shared_model import SiameseNet
from linear_relu_model import BaselineNet


def evaluate(models, evaluation_func, epochs):
    for model in models:
        evaluation_func.evaluate(model, epochs)
        print("\n")


def main():
    epochs = 100

    # ------------------------------------------------------------------
    # models evaluated with accuracy and loss
    # ------------------------------------------------------------------
    accuracy_loss_models = [BaselineNet(),
                            SiameseNoSharingNet(),
                            SiameseNet(),
                            SiameseConvNoSharingNet(),
                            SiameseConvNet()
                            ]

    evaluate(accuracy_loss_models,
             accuracy_loss_eval,
             epochs)

    # ------------------------------------------------------------------
    # models evaluated with accuracy, ciphers accuracy and weighted loss
    # ------------------------------------------------------------------
    rnn_params = 64, 64, 14, 50, 10
    accuracy_ciphers_weighted_loss_eval_models = [Conv_basic(),
                                                  SiameseConvNet2(),
                                                  ImageRNN(*rnn_params),
                                                  ImageRNN_2(*rnn_params)
                                                  ]

    evaluate(accuracy_ciphers_weighted_loss_eval_models,
             accuracy_ciphers_weighted_loss_eval,
             epochs)

    print("Script ended.")


if __name__ == "__main__":
    main()
