import os
import numpy as np
from datetime import date
import keras
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input, Softmax, Lambda
#from keras.activations import softmax
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.losses import mean_squared_error, MeanSquaredError, MSE

import random as rd
from scipy.special import softmax


#from JuniaDotsnBoxes import OWNER_NONE, OWNER_AI1, OWNER_AI2, BOARDSIZE
OWNER_NONE = 0
OWNER_AI1 = 1
OWNER_AI2 = 2

BOARDSIZE = 8

class Agent:

	# ----------------------------------------
	# 			Initialize Agent
	# ----------------------------------------

	def __init__(self, load_trained: bool, ai_sign: int, moves_list):
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
		self.ai_sign = ai_sign
		self.other_sign = OWNER_AI2 if ai_sign == OWNER_AI1 else OWNER_AI1

		# private attributes for building the neural network
		self.nn_learningRate = 0.08 #min e^(-3) = 0.05
        # self.nn_input_positions = None
        # self.nn_target_input = None
        # self.nn_q_values = None
        # self.nn_probabilities = None
        # self.nn_train_step = None
		
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

		input_positions = Input(shape=(len(self.moves_ref)*3,), name='inputs')
		x = Dense(len(self.moves_ref)*9, kernel_initializer='glorot_uniform', activation='relu')(input_positions)
		x = Dense(len(self.moves_ref)*5, kernel_initializer='glorot_uniform', activation='relu')(x)
		x = Dense(len(self.moves_ref)*2, kernel_initializer='glorot_uniform', activation='relu')(x)
		q_values = Dense(len(self.moves_ref), kernel_initializer='glorot_uniform', activation='relu', name='q_values')(x)
		# probabilities = Softmax(name='probas')(q_values)
		# self.model = Model(inputs = input_positions, outputs = [q_values, probabilities])
		# self.model.compile(optimizer=SGD(learning_rate=0.1), loss={'q_values': 'mean_squared_error', 'probas': None})
		
		self.model = Model(inputs = input_positions, outputs = q_values)
		self.model.compile(optimizer=Adam(learning_rate=self.nn_learningRate), loss='binary_crossentropy')

		#self.save()

		# input_positions = Input(shape=(len(self.moves_ref),), name='inputs')
		# target_inputs = Input(shape=(BOARDSIZE,), name='targets')

		# x = Dense(len(self.moves_ref)*9, activation='relu')(input_positions)

		# q_values = Dense(BOARDSIZE, name='q_values')(x)
		# probabilities = Softmax(name='probabilities')(q_values)
		# mse = MeanSquaredError()

		# self.model = Model(inputs = [input_positions, target_inputs], outputs = [q_values, probabilities, target_inputs])
		# self.model.compile(optimizer=SGD(learning_rate=0.1), loss=[None, mse])
		
		# return probabilities




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
		return np.array([(lineboard == self.ai_sign).astype(int),
						(lineboard == self.other_sign).astype(int),
						(lineboard == OWNER_NONE).astype(int)])


	# need keras rewrite
	def get_probs(self, input_vector: np.ndarray) -> ([float], [float]):
		"""
		Computes the Q values and corresponding nn_probabilities for all moves (including illegal ones).
		:param input_vector: The feature vector analyzed by the Neural Network.
		:return: A tuple of nn_probabilities and q values of all actions (including illegal ones).
		"""
		#probs, qvalues = tf.Session().run([self.nn_probabilities, self.nn_q_values], feed_dict={self.nn_input_positions: [input_vector]}) #TensorFlow 1.x
		#return probs[0], qvalues[0] #why index 0 ?
		q_values = self.model.predict(input_vector.reshape(1,-1))
		probs = softmax(q_values)
		return probs[0], q_values[0]
		# probabilities = Softmax()(q_values)
		# return probabilities, q_values
		#random probs
		# print(list(q_values))
		# print([rd.random() for i in range(int(input_vector.size/3))], [rd.random() for i in range(int(input_vector.size/3))])
		# print(len([rd.random() for i in range(int(input_vector.size/3))]), len([rd.random() for i in range(int(input_vector.size/3))]))


	def reward(self, ai_score, other_score):
		#update last move coeff with score diff
		diff_score = (ai_score - self.last_ai_score) - (other_score - self.last_other_score)

		# if abs(diff_score) == 3:
		# 	pass
		# elif abs(diff_score) == 7:
		# 	pass

		#diff_score = 1 + ((ai_score - self.last_ai_score) - (other_score - self.last_other_score)) / 10

		diff_score = (1 + diff_score / 10)

		# if diff_score < 0:
		# 	diff_score = (1 + diff_score / 10) / self.reward_weight
		# elif diff_score > 0:
		# 	diff_score = (1 + diff_score / 10) * self.reward_weight
		
		# print(f"\n\nlast ai score : {self.last_ai_score}, last other score : {self.last_other_score}")
		# print(f"ai score : {ai_score}, other score : {other_score}, diff score : {tmp}")
		# print(f"reward : {diff_score}")
		# print(f"poid initial : {self.qvalue_move_with_reward_by_turn[-1]}")

		if diff_score != 0:
			if self.qvalue_move_with_reward_by_turn[-1] == 0.0:
				self.qvalue_move_with_reward_by_turn[-1] = 0.001
			self.qvalue_move_with_reward_by_turn[-1] *= diff_score
			# print(f"poid final   : {self.qvalue_move_with_reward_by_turn[-1]}")
		
		self.last_ai_score = ai_score
		self.last_other_score = other_score


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def play(self, lineboard, moves_remaining):
		return self.play_competitive(lineboard) if not self.train_model else self.play_train(lineboard)


	def play_competitive(self, lineboard) -> (int, int): # must be ultra optimized
		"""
		:param board: [(id_point, coord x, coord y, id_connected_points)]
		:param lineboard: [owner_segment_0, owner_segment_1, ...], owner = nobody (0), player1 (1) or player2 (2)
		:param moves_remaining: [(id_point_0, id_point_1), (id_point_1, id_point_2), ...]
		:return: a tuple of two id_points who represent a line
		"""

		# prepare board
		nn_input = self.normalize_nn_input(np.array(lineboard))
		
		# get model predictions
		probs, qvalues = self.get_probs(nn_input)
		
		nn_input = nn_input.reshape(-1)
		# discard already done moves
		for index in range(len(qvalues)):#moyen de faire plus efficace? 112 valeurs à traiter...
			if not nn_input[len(lineboard) * 2 + index]: # ligne pas encore tracée (3e tableau)
				probs[index] = -1

		# return tuple of the move which have the highest probability
		return self.moves_ref[np.argmax(probs)]


	def clear(self):
		self.lineboard_by_turn = []
		self.chosen_move_by_turn = []
		self.qvalue_move_with_reward_by_turn = []
		self.qvalues_by_turn = []


	def play_train(self, lineboard) -> (int, int): # an heavier version of play
		"""
		...
		"""	
		lineboard_array = np.array(lineboard)
		
		# record all game positions to feed them into the NN for training with the corresponding updated Q values
		self.lineboard_by_turn.append(lineboard_array.copy())

		nn_input = self.normalize_nn_input(lineboard_array)
		probs, qvalues = self.get_probs(nn_input)

		nn_input = nn_input.reshape(-1)
		# discard already done moves
		for index in range(len(qvalues)):#moyen de faire plus efficace? 112 valeurs à traiter...
			if not nn_input[len(lineboard) * 2 + index]: # ligne pas encore tracée (3e tableau)
				probs[index] = -1

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
		self.reward(ai_score, other_score)

	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained):
		if not self.train_model:
			return
			
		#normalise correctement
		other_move = self.moves_ref.index(other_move)

 		#inverse point de vue
		self.ai_sign, self.other_sign = self.other_sign, self.ai_sign
		self.last_ai_score, self.last_other_score = self.last_other_score, self.last_ai_score


		lineboard_array = np.array(lineboard)
		
		# record all game positions to feed them into the NN for training with the corresponding updated Q values
		self.lineboard_by_turn.append(lineboard_array.copy())

		nn_input = self.normalize_nn_input(lineboard_array)
		probs, qvalues = self.get_probs(nn_input)
		

		# record the action selected as well as the Q values of the current state for later use when adjusting NN weights.
		self.chosen_move_by_turn.append(other_move)
		self.qvalues_by_turn.append(qvalues)
		self.qvalue_move_with_reward_by_turn.append(qvalues[other_move])
		

		#pondère intérêt du coup étudié selon score obtenu
		self.reward(ai_score, other_score)

		self.ai_sign, self.other_sign = self.other_sign, self.ai_sign #retablit point de vue
		self.last_ai_score, self.last_other_score = self.last_other_score, self.last_ai_score


	def train(self, lineboard, ai_score: int, other_score: int):
		"""
		This method is called once the game is over. If 'self.train_model' is True, we execute a training run for
		the Neural Network.
		:param lineboard: The result of the game that just finished.
		"""
		if not self.train_model:
			return
		
		reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.2,
                            min_lr=0.001, mode='auto', verbose=1)

		# Compute the final reward based on the game outcome
		reward = self.reward_weight * ai_score / (ai_score + other_score) #peut l'appliquer à chaque poid...?

		# If we are in training mode we run the optimizer.
		if self.train_model:
			# We calculate our new estimate of what the true Q values are and feed that into the network as learning target

			targets = []
			for i in range(len(self.chosen_move_by_turn)):
				target = np.copy(self.qvalues_by_turn[i])
				target[self.chosen_move_by_turn[i]] = self.qvalue_move_with_reward_by_turn[i] #* reward ?
				targets.append(target)

			# We convert the input states we have recorded to feature vectors to feed into the training.
			nn_input = [(self.normalize_nn_input(x)).reshape(1,-1) for x in self.lineboard_by_turn]

			for board, target in zip(nn_input, targets):
				self.model.fit(board[0].reshape(1,-1), target.reshape(1,-1), verbose=0) #verbose=1, callbacks=[reduce_lr])
				# self.model.fit({"input_positions":board[0]}, {'q_values':target, 'probas':softmax(target)})
			# print(len(nn_input))
			# print(len(targets))
			# print(targets)
			# We run the training step with the recorded inputs and new Q value targets.
			# for target in targets:
			# 	
			# tf.Session().run([self.nn_train_step], feed_dict={self.nn_input_positions: nn_input, self.nn_target_input: targets})
