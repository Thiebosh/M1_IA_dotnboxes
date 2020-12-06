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

	def __init__(self, train: bool, moves_list):
		# id who will be displayed
		#self.name = "Thibaut Juzeau"
		self.name = __file__[__file__.rfind('\\')+1:-3] #retire path et extension

		# private attributes for load and save the neural network with keras
		self.train_model = train
		self.model_name = self.name #date.today().strftime("%Y-%m-%d")
		self.model = None

		# game info
		self.moves_ref = moves_list

		# private attributes for building the neural network
		self.nn_learningRate = 0.06 #min e^(-3) = 0.05
		self.nn_gamma = 0.01 #coup aléatoire pour ne pas rester bloqué dans une strat
		
		# private attributes for training the neural network
		self.last_ai_score_for_ai = 0
		self.last_other_score_for_ai = 0
		self.last_other_score_for_other = 0
		self.last_ai_score_for_other = 0

		self.ai_lineboard_by_turn = []
		self.ai_chosen_move_by_turn = []
		self.ai_qvalue_move_with_reward_by_turn = []
		self.ai_qvalues_by_turn = []
		
		self.other_lineboard_by_turn = []
		self.other_chosen_move_by_turn = []
		self.other_qvalue_move_with_reward_by_turn = []
		self.other_qvalues_by_turn = []


		if os.path.isfile(f"players/models/{self.model_name}.h5"):
			self.model = load_model(f"players/models/{self.model_name}.h5")
		else:
			print("\n\nwarning : model not found\n\n")
			
			input_positions = Input(shape=(len(self.moves_ref),), name='inputs')
			x = Dense(len(self.moves_ref)*4, kernel_initializer='glorot_uniform', activation='relu')(input_positions)
			x = Dense(len(self.moves_ref)*6, kernel_initializer='glorot_uniform', activation='relu')(x)
			x = Dense(len(self.moves_ref)*4, kernel_initializer='glorot_uniform', activation='relu')(x)
			x = Dense(len(self.moves_ref)*2, kernel_initializer='glorot_uniform', activation='relu')(x)
			output_prob = Dense(len(self.moves_ref), kernel_initializer='glorot_uniform', activation='softmax', name='q_values')(x)
			
			
			self.model = Model(inputs = input_positions, outputs = output_prob)
			self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.nn_learningRate))


		#self.save()


	def save(self):
		"""
		save good model
		"""
		if self.train_model:
			self.model.save(f"players/models/{self.model_name}.h5")


	# ----------------------------------------
	# 			Common internal methods
	# ----------------------------------------

	def normalize_nn_input(self, lineboard: np.ndarray) -> np.ndarray:
		return np.array([int(x != OWNER_NONE) for x in lineboard]).reshape(1,-1)


	# need keras rewrite
	def get_probs(self, normalized_lineboard) -> ([float], [float]):
		return self.model.predict(normalized_lineboard)[0]


	def reward(self, ai_score, last_ai_score, other_score, last_other_score, nb_moves_chained):
		
		#update last move coeff with score diff
		diff_score = (ai_score - last_ai_score) - (other_score - last_other_score)
		# self.data_score_delta.append(diff_score)

		if abs(diff_score) < 20: #log(20 + 1) / 3 = 1.01 donc risque de négatif au delà de 20  
			reward = 1 + math.copysign(math.log(abs(diff_score)+1) / 3, diff_score) #applique signe de diff_score à expression
		elif diff_score > 0:
			reward = 2
		else:
			reward = 0
		# self.data_move_reward.append(reward)

		return reward


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def play(self, lineboard, moves_remaining):
		return self.play_competitive(lineboard) if not self.train_model else self.play_train(lineboard, moves_remaining)


	def play_competitive(self, lineboard) -> (int, int): # must be ultra optimized
		# get model predictions
		probs = self.get_probs(self.normalize_nn_input(lineboard))

		# discard already done moves
		probs[[index for index in range(len(probs)) if lineboard[index] != OWNER_NONE]] = -1
		
		# return tuple of the move which have the highest probability
		return self.moves_ref[np.argmax(probs)]


	def clear(self):
		self.ai_lineboard_by_turn = []
		self.ai_chosen_move_by_turn = []
		self.ai_qvalue_move_with_reward_by_turn = []
		self.ai_qvalues_by_turn = []

		self.other_lineboard_by_turn = []
		self.other_chosen_move_by_turn = []
		self.other_qvalue_move_with_reward_by_turn = []
		self.other_qvalues_by_turn = []


	def play_train(self, lineboard, moves_remaining) -> (int, int): # an heavier version of play
		normalized_lineboard = self.normalize_nn_input(lineboard)
		
		# record all game positions to feed them into the NN for training with the corresponding updated Q values
		self.ai_lineboard_by_turn.append(normalized_lineboard)

		# get model predictions
		probs = self.get_probs(normalized_lineboard)

		if self.nn_gamma > rd.random(): #de temps en temps, fait move aléatoire pour ne pas se figer
			move = self.moves_ref.index(rd.choice(moves_remaining))
		else:
			# discard already done moves
			probs[[index for index in range(len(probs)) if lineboard[index] != OWNER_NONE]] = -1
			
			# Our next move is the one with the highest probability after removing all illegal ones.
			move = np.argmax(probs)  # indice du meilleur mouvement à tracer

		self.ai_chosen_move_by_turn.append(move)
		self.ai_qvalues_by_turn.append(probs)
		self.ai_qvalue_move_with_reward_by_turn.append(probs[move])

		# We return the selected move
		return self.moves_ref[move]

	def play_result(self, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		if not self.train_model:
			return

		if nb_moves_chained > 0: #récompense coup qui a permis de rejouer
			self.ai_qvalue_move_with_reward_by_turn[-2] *= (math.log(1 + nb_moves_chained / 20) + 1) #1,...

		reward = self.reward(ai_score, self.last_ai_score_for_ai, other_score, self.last_other_score_for_ai, nb_moves_chained)

		# self.data_move_qvalue_before.append(self.ai_qvalue_move_with_reward_by_turn[-1])
		self.ai_qvalue_move_with_reward_by_turn[-1] *= reward
		# self.data_move_qvalue_after.append(self.ai_qvalue_move_with_reward_by_turn[-1])

		self.last_ai_score_for_ai = ai_score
		self.last_other_score_for_ai = other_score

	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained, remain):
		if not self.train_model:
			return
			
		#normalise correctement
		other_move = self.moves_ref.index(other_move)
		
		# record all game positions to feed them into the NN for training with the corresponding updated Q values
		normalized_lineboard = self.normalize_nn_input(lineboard)

		self.other_lineboard_by_turn.append(normalized_lineboard)

		probs = self.get_probs(normalized_lineboard)
		

		# record the action selected as well as the Q values of the current state for later use when adjusting NN weights.
		self.other_chosen_move_by_turn.append(other_move)
		self.other_qvalues_by_turn.append(probs)
		self.other_qvalue_move_with_reward_by_turn.append(probs[other_move])
		
		
		if nb_moves_chained > 0: #récompense coup qui a permis de rejouer
			self.other_qvalue_move_with_reward_by_turn[-2] *= (math.log(1 + nb_moves_chained / 20) + 1) #1,...

		#pondère intérêt du coup étudié selon score obtenu
		reward = self.reward(other_score, self.last_other_score_for_other, ai_score, self.last_ai_score_for_other, nb_moves_chained)

		# self.data_move_qvalue_before.append(self.other_qvalue_move_with_reward_by_turn[-1])
		self.other_qvalue_move_with_reward_by_turn[-1] *= reward
		# self.data_move_qvalue_after.append(self.other_qvalue_move_with_reward_by_turn[-1])

		self.last_other_score_for_other = other_score
		self.last_ai_score_for_other = ai_score



	def train(self, lineboard, ai_score, other_score):
		if not self.train_model:
			return
		
		# reduce_lr = ReduceLROnPlateau(monitor='lr', factor=0.2,
        #                     min_lr=0.001, mode='auto', verbose=1)

		# Compute the final reward based on the game outcome
		ai_global_reward = 2 * ai_score / (ai_score + other_score) #entre 0 et 2. Match nul == 1
		other_global_reward = 2 * other_score / (ai_score + other_score) #entre 0 et 2. Match nul == 1

		# If we are in training mode we run the optimizer.
		if self.train_model:
			# We calculate our new estimate of what the true Q values are and feed that into the network as learning target
			targets = []
			for i in range(len(self.ai_chosen_move_by_turn)):
				target = np.copy(self.ai_qvalues_by_turn[i])
				target[self.ai_chosen_move_by_turn[i]] = self.ai_qvalue_move_with_reward_by_turn[i] * ai_global_reward #vient nuancer l'ensemble des coups
				targets.append(target.reshape(1,-1))

			for board, target in zip(self.ai_lineboard_by_turn, targets):
				self.model.fit(board, target, verbose=0)#, epochs = 1 if ai_global_reward < other_global_reward else 3)

			targets = []
			for i in range(len(self.other_chosen_move_by_turn)):
				target = np.copy(self.other_qvalues_by_turn[i])
				target[self.other_chosen_move_by_turn[i]] = self.other_qvalue_move_with_reward_by_turn[i] * other_global_reward #vient nuancer l'ensemble des coups
				targets.append(target.reshape(1,-1))

			for board, target in zip(self.other_lineboard_by_turn, targets):
				self.model.fit(board, target, verbose=0)#, epochs = 1 if ai_global_reward > other_global_reward else 3)
