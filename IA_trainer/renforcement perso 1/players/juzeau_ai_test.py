import os
import math
import numpy as np
import random as rd
# import pandas as pd

from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input, Softmax, Lambda
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau


#from JuniaDotsnBoxes import OWNER_NONE, BOARDSIZE
OWNER_NONE = 0

BOARDSIZE = 8

class Agent:

	# ----------------------------------------
	# 			Initialize Agent
	# ----------------------------------------

	def __init__(self, train, moves_list):
		# id who will be displayed
		self.name = "ai_test"

		# private attributes for load and save the neural network with keras
		self.model_name = "qlearner"
		self.model = None

		# game info
		self.moves_ref = moves_list
		
		if os.path.isfile(f"players/models/{self.model_name}.h5"):
			self.model = load_model(f"players/models/{self.model_name}.h5")
			print(self.model.summary())
			#exit()
		else:
			print("modèle introuvable")
			exit()


	# ----------------------------------------
	# 			Common internal methods
	# ----------------------------------------

	def normalize_nn_input(self, lineboard):
		return np.array([int(x != OWNER_NONE) for x in lineboard]).reshape(1,-1)


	# need keras rewrite
	def get_probs(self, normalized_lineboard):
		return self.model.predict(normalized_lineboard)[0]


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def play(self, lineboard, moves_remaining):
		# get model predictions
		probs = self.get_probs(self.normalize_nn_input(lineboard))

		# discard already done moves
		probs[[index for index in range(len(probs)) if lineboard[index] != OWNER_NONE]] = -1
		
		# return tuple of the move which have the highest probability
		return self.moves_ref[np.argmax(probs)]




	def save(self):
		pass

	def clear(self):
		pass

	def play_result(self, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass


	def train(self, lineboard, ai_score: int, other_score: int):
		pass
	
	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass
	