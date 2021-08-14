# ColourPredictor
Train a neural network on an RGB value and output the best overlaying colour for text: [while, black].

A program that sends an RGB [[0],[0],[0]] value into a feedforward network which outputs
an arrar([float, float]) to suggest [white or black] as overlay colour for text.

Requirements:

    Feedforward_Network.py -> https://github.com/P-O-K/Feedforward-Neural-Network

Main.py

    Opens a pygame window with 2 rectangles, fitted with some sample text.

    Click the rectangle with the easiest to read text, to train
    the network on that BACKGROUND/TEXT colour combination

    The small rectangle under text represents the networks prediction

ColourPredictor.py

    Feeds RGB colour through the network predictor function and returns
    an array[ float, float ] representing font colours [ WHITE, BLACK ]
    
Constants.py

    File of constant variables to be accessible by both: Main.py & ColourPredictor.py

WeightsAndBiases_3_4_2_.npy

    Quick pre-trained set of Weights & Biases for network shape[ 3, 4, 2 ]
