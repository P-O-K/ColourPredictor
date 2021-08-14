
from numpy.random import randint
from numpy import array

import Feedforward_Network as FFN
import pygame; pygame.init( )
import Constants as const
from os import listdir


class InitSetup( object ):

	pygame.display.set_caption( 'ColourPicker' )
	MAIN_WINDOW = pygame.display.set_mode( ( const.DISPLAY_WIDTH, const.DISPLAY_HEIGHT ) )
	FONT_SET = pygame.font.SysFont( const.FONT_TYPE, const.FONT_SIZE )

	CLOCK = pygame.time.Clock( )

	LEFT_RECT  = pygame.Rect(                              5,   5, ( const.DISPLAY_WIDTH //2 ) -5,  const.DISPLAY_HEIGHT -10 )
	RIGHT_RECT = pygame.Rect( ( const.DISPLAY_WIDTH //2 ) +5,   5, ( const.DISPLAY_WIDTH //2 ) -10, const.DISPLAY_HEIGHT -10 )
	PRED_RECT  = pygame.Rect( 0, 0, 80, 20 )

	DECISION_NETWORK = FFN.FeedforwardNetwork( const.NETWORK_SHAPE )
	DECISION_NETWORK.ACTIVATION_METHOD = const.NETWORK_ACTIVATION_METHOD
	DECISION_NETWORK.MAX_LEARNING_RATE = const.NETWORK_MAX_LEARNING_RATE

	CURRENT_COLOUR :array
	PRED_RECT_COLOUR = ( 0, 0, 0 )



class ColorPicker( InitSetup ):


	def __init__( self ):
		if const.WEIGHT_FILE_NAME in listdir( ):
			if self.DECISION_NETWORK.loadWeights( const.WEIGHT_FILE_NAME ):
				print( f'File Loaded Successfully! -> "{const.WEIGHT_FILE_NAME}"' )
		else:
			print( f'File "{const.WEIGHT_FILE_NAME}" Does\'nt exist, using untrained network instead!' )
			const.WEIGHT_FILE_NAME = 'unknownWeights.npy'

		self.CURRENT_COLOUR = self.getNewColor( )
		self.testNetwork( )
		self.runModule( )



	@staticmethod
	def getNewColor( ) -> array:
		return array( [ [ randint( 0, 255 ) ] for _ in range( 3 ) ] )



	def trainNetwork( self, colour:array, label:array ) -> None:
		self.DECISION_NETWORK.runInstance( colour/255, label )



	def testNetwork( self ) -> None:
		result = self.DECISION_NETWORK.predictor( self.CURRENT_COLOUR/255 )
		rct, col = pygame.Rect( const.DISPLAY_WIDTH//2-5, self.LEFT_RECT.bottom, 10, 10 ), ( 0, 0, 0 )
		
		if result[ 0 ] > result[ 1 ]: rct, col = self.LEFT_RECT, const.LEFT_TEXT_COLOUR
		elif result[ 0 ] < result[ 1 ]: rct, col = self.RIGHT_RECT, const.RIGHT_TEXT_COLOUR
		
		self.PRED_RECT.centerx, self.PRED_RECT.centery = rct.centerx, rct.bottom -50
		self.PRED_RECT_COLOUR = col



	def setTextObject( self ) -> None:
		leftTextObject = self.FONT_SET.render( const.SAMPLE_TEXT, True, const.LEFT_TEXT_COLOUR )
		rightTextObject = self.FONT_SET.render( const.SAMPLE_TEXT, True, const.RIGHT_TEXT_COLOUR )

		fontSizeX = self.FONT_SET.size( const.SAMPLE_TEXT )[ 0 ] /2
		fontSizeY = self.FONT_SET.size( const.SAMPLE_TEXT )[ 1 ] /2
		leftTextLocation = ( self.LEFT_RECT.centerx -fontSizeX, self.LEFT_RECT.centery -fontSizeY )
		rightTextLocation = ( self.RIGHT_RECT.centerx -fontSizeX, self.RIGHT_RECT.centery -fontSizeY )

		self.MAIN_WINDOW.blit( leftTextObject,  leftTextLocation )
		self.MAIN_WINDOW.blit( rightTextObject, rightTextLocation )



	def renderWindow( self ) -> None:
		self.MAIN_WINDOW.fill( const.BACKGROUND_COLOUR )
 
		pygame.draw.rect( self.MAIN_WINDOW, self.CURRENT_COLOUR, self.LEFT_RECT  )
		pygame.draw.rect( self.MAIN_WINDOW, self.CURRENT_COLOUR, self.RIGHT_RECT )
		pygame.draw.rect( self.MAIN_WINDOW, self.PRED_RECT_COLOUR, self.PRED_RECT )

		self.setTextObject( )

		pygame.display.update( )



	def eventHandle( self ) -> bool:
		for event in pygame.event.get( ):
			if event.type == pygame.QUIT:
				return True

			if event.type == pygame.MOUSEBUTTONDOWN:
				if self.LEFT_RECT.collidepoint( pygame.mouse.get_pos( ) ):
					self.trainNetwork( colour=self.CURRENT_COLOUR, label=array([ [1],[-1] ]) )
					self.CURRENT_COLOUR = self.getNewColor( )
					self.testNetwork( )

				if self.RIGHT_RECT.collidepoint( pygame.mouse.get_pos( ) ):
					self.trainNetwork( colour=self.CURRENT_COLOUR, label=array([ [-1],[1] ]) )
					self.CURRENT_COLOUR = self.getNewColor( )
					self.testNetwork( )

		return False



	def runModule( self ) -> None:
		exitStatus = False
		while not exitStatus:

			exitStatus = self.eventHandle( )
			self.renderWindow( )
			self.CLOCK.tick( const.FRAMERATE )

		self.DECISION_NETWORK.saveWeights( const.WEIGHT_FILE_NAME )
		pygame.quit( )



if __name__ == '__main__':
	CP = ColorPicker(  )
	raise SystemExit