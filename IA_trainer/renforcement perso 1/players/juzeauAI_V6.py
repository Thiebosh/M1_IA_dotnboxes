import os
import numpy as np
from datetime import date
import keras
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input, Softmax, Lambda
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau

import random as rd
from scipy.special import softmax


#from JuniaDotsnBoxes import OWNER_NONE, BOARDSIZE
OWNER_NONE = 0

BOARDSIZE = 8

class Agent:

	# ----------------------------------------
	# 			Initialize Agent
	# ----------------------------------------

	def __init__(self, load_trained: bool, moves_list):
		# id who will be displayed
		#self.name = "Thibaut Juzeau"
		self.name = __file__[__file__.rfind('\\')+1:-3] #retire path et extension

		# private attributes for load and save the neural network with keras
		self.train_model = True
		self.load_trained_model = load_trained
		self.model_name = self.name #date.today().strftime("%Y-%m-%d")
		self.model = None

		# game info
		self.moves_ref = moves_list

		# private attributes for building the neural network
		self.nn_learningRate = 0.06 #min e^(-3) = 0.05
		self.nn_gamma = 0.025 #coup aléatoire pour ne pas rester bloqué dans une strat
		
		# private attributes for training the neural network
		self.last_ai_score = 0 #initial score = 0
		self.last_other_score = 0  #initial score = 0

		self.reward_weight = 1.0
		self.lineboard_by_turn = []
		self.chosen_move_by_turn = []
		self.qvalue_move_with_reward_by_turn = []
		self.qvalues_by_turn = []

		# choose correct option
		if self.load_trained_model:
			self.load()
		else :
			self.build()
	
	
	def build(self):
		"""
		Creates a neural network
		"""
		if self.model != None:
			return

		input_positions = Input(shape=(len(self.moves_ref)*2,), name='inputs')
		x = Dense(len(self.moves_ref)*6, kernel_initializer='glorot_uniform', activation='relu')(input_positions)
		x = Dense(len(self.moves_ref)*4, kernel_initializer='glorot_uniform', activation='relu')(x)
		x = Dense(len(self.moves_ref)*2, kernel_initializer='glorot_uniform', activation='relu')(x)
		q_values = Dense(len(self.moves_ref), kernel_initializer='glorot_uniform', activation='relu', name='q_values')(x)
		
		
		self.model = Model(inputs = input_positions, outputs = q_values)
		self.model.compile(optimizer=Adam(learning_rate=self.nn_learningRate), loss='categorical_crossentropy')

		#self.save()



	def save(self):
		"""
		save good model
		"""
		if self.train_model:
			self.model.save(f"players/models/{self.model_name}.h5")


	def load(self):
		"""
		load existing model
		"""
		if os.path.isfile(f"players/models/{self.model_name}.h5"):
			self.model = load_model(f"players/models/{self.model_name}.h5")
		else:
			print("\n\nwarning : model not found\n\n")
			self.build()

	# ----------------------------------------
	# 			Common internal methods
	# ----------------------------------------

	def normalize_nn_input(self, lineboard: np.ndarray) -> np.ndarray:
		"""
		Converts lineboard to an input feature vector for the Neural Network.
		The input feature vector is a bit array of size (3*lineboard.size) :
		- the first lineboard.size bits are set to 1 on positions with our pieces,
		- the second lineboard.size bits are set to 1 on positions with our opponents pieces,
		- the final lineboard.size bits are set to 1 on empty positions.
		:param lineboard: 
		:return: a bit array which describe 'our positions', 'ennemy positions' and 'free positions'
		"""
		return np.array([(lineboard != OWNER_NONE).astype(int), (lineboard == OWNER_NONE).astype(int)]).reshape(1,-1)


	# need keras rewrite
	def get_probs(self, lineboard_array: np.ndarray) -> ([float], [float]):
		"""
		Computes the Q values and corresponding nn_probabilities for all moves (including illegal ones).
		:param input_vector: The feature vector analyzed by the Neural Network.
		:return: A tuple of nn_probabilities and q values of all actions (including illegal ones).
		"""
		q_values = self.model.predict(self.normalize_nn_input(lineboard_array))
		probs = softmax(q_values)
		return probs[0], q_values[0]
		# probabilities = Softmax()(q_values)
		# return probabilities, q_values


	def reward(self, ai_score):
		#update last move coeff with score diff
		diff_score = (ai_score - self.last_ai_score)

		# if abs(diff_score) == 3:
		# 	pass
		# elif abs(diff_score) == 7:
		# 	pass

		#diff_score = 1 + ((ai_score - self.last_ai_score) - (other_score - self.last_other_score)) / 10

		diff_score = (1 + diff_score / 40)

		# if diff_score < 0:
		# 	diff_score = (1 + diff_score / 10) / self.reward_weight
		# elif diff_score > 0:
		# 	diff_score = (1 + diff_score / 10) * self.reward_weight
		

		if diff_score != 0:
			if self.qvalue_move_with_reward_by_turn[-1] == 0.0:
				self.qvalue_move_with_reward_by_turn[-1] = 0.001
			self.qvalue_move_with_reward_by_turn[-1] *= diff_score
			# print(f"poid final   : {self.qvalue_move_with_reward_by_turn[-1]}")
		
		self.last_ai_score = ai_score


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def play(self, lineboard, moves_remaining):
		return self.play_competitive(lineboard) if not self.train_model else self.play_train(lineboard, moves_remaining)


	def play_competitive(self, lineboard) -> (int, int): # must be ultra optimized
		"""
		:param board: [(id_point, coord x, coord y, id_connected_points)]
		:param lineboard: [owner_segment_0, owner_segment_1, ...], owner = nobody (0), player1 (1) or player2 (2)
		:param moves_remaining: [(id_point_0, id_point_1), (id_point_1, id_point_2), ...]
		:return: a tuple of two id_points who represent a line
		"""
		
		# get model predictions
		probs, qvalues = self.get_probs(np.array(lineboard))

		# discard already done moves
		probs[[index for index in range(len(qvalues)) if lineboard[index] != OWNER_NONE]] = -1
		
		# return tuple of the move which have the highest probability
		return self.moves_ref[np.argmax(probs)]


	def clear(self):
		self.lineboard_by_turn = []
		self.chosen_move_by_turn = []
		self.qvalue_move_with_reward_by_turn = []
		self.qvalues_by_turn = []


	def play_train(self, lineboard, moves_remaining) -> (int, int): # an heavier version of play
		"""
		...
		"""	
		lineboard_array = np.array(lineboard)
		
		# record all game positions to feed them into the NN for training with the corresponding updated Q values
		self.lineboard_by_turn.append(lineboard_array.copy())

		# get model predictions
		probs, qvalues = self.get_probs(lineboard_array)

		if self.nn_gamma > rd.random(): #de temps en temps, fait move aléatoire pour ne pas se figer
			move = self.moves_ref.index(rd.choice(moves_remaining))
		else:
			# discard already done moves
			probs[[index for index in range(len(qvalues)) if lineboard[index] != OWNER_NONE]] = -1
			
			# Our next move is the one with the highest probability after removing all illegal ones.
			move = np.argmax(probs)  # indice du meilleur mouvement à tracer

		self.chosen_move_by_turn.append(move)
		self.qvalues_by_turn.append(qvalues)
		self.qvalue_move_with_reward_by_turn.append(qvalues[move])

		# We return the selected move
		return self.moves_ref[move]

	def play_result(self, ai_score, other_score, nb_moves_chained):
		if not self.train_model:
			return
		self.reward(ai_score)

	def other_play(self, lineboard_before, ai_choice, score_AI, score_other, nb_moves_chained):
		pass

	def train(self, lineboard, ai_score: int, other_score: int):
		"""
		This method is called once the game is over. If 'self.train_model' is True, we execute a training run for
		the Neural Network.
		:param lineboard: The result of the game that just finished.
		"""
		if not self.train_model:
			return
		
		# reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.2,
        #                     min_lr=0.001, mode='auto', verbose=1)

		# # Compute the final reward based on the game outcome
		# reward = self.reward_weight * ai_score / (ai_score + other_score) #peut l'appliquer à chaque poid...?

		# If we are in training mode we run the optimizer.
		if self.train_model:
			# We calculate our new estimate of what the true Q values are and feed that into the network as learning target

			targets = []
			for i in range(len(self.chosen_move_by_turn)):
				target = np.copy(self.qvalues_by_turn[i])
				target[self.chosen_move_by_turn[i]] = self.qvalue_move_with_reward_by_turn[i] #* reward ?
				targets.append(target.reshape(1,-1))

			for board, target in zip(self.lineboard_by_turn, targets):
				self.model.fit(self.normalize_nn_input(board), target, verbose=0) #verbose=1, callbacks=[reduce_lr])
				# self.model.fit({"input_positions":board[0]}, {'q_values':target, 'probas':softmax(target)})
