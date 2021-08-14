
from Constants import NETWORK_SHAPE, NETWORK_ACTIVATION_METHOD, WEIGHT_FILE_NAME
import Feedforward_Network as FF_Net
from numpy import array
from os import listdir



def getFontColour( colour:array ) -> tuple( ( str, tuple ) ):

	if WEIGHT_FILE_NAME in listdir( ):
		FNN = FF_Net.FeedforwardNetwork( NETWORK_SHAPE )
		FNN.ACTIVATION_METHOD = NETWORK_ACTIVATION_METHOD
		FNN.loadWeights( WEIGHT_FILE_NAME )
		
		result = FNN.predictor( colour /255 )
		if result[ 0 ] > result[ 1 ]: return 'White', ( 255, 255, 255 )
		if result[ 1 ] > result[ 0 ]: return 'Black', (   0,   0,   0 )



if __name__ == '__main__':
	print( 'Font Overlay Colour:', getFontColour( array( [ [83], [32], [173] ] ) ) )
	print( 'Font Overlay Colour:', getFontColour( array( [ [109], [183], [240] ] ) ) )