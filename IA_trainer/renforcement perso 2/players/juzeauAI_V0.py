import os
import math
import numpy as np
import random as rd

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
		#self.name = "Thibaut Juzeau"
		self.name = __file__[__file__.rfind('\\')+1:-3] #retire path et extension

		# private attributes for load and save the neural network with keras
		self.train_model = train
		self.model_name = self.name #date.today().strftime("%Y-%m-%d")
		self.model = None

		# game info
		self.moves_ref = moves_list

		# private attributes for building the neural network
		self.nn_learningRate = 0.001 #min e^(-3) = 0.05
		
		self.ai_lineboard_by_turn = []
		self.ai_to_predict_by_turn = []


		
		if os.path.isfile(f"players/models/{self.model_name}.h5"):
			self.model = load_model(f"players/models/{self.model_name}.h5")
		else:
			print("\n\nwarning : model not found\n\n")
			
			input_positions = Input(shape=(len(self.moves_ref),), name='inputs')
			x = Dense(len(self.moves_ref)*4, activation='relu')(input_positions)
			x = Dense(len(self.moves_ref)*10, activation='relu')(x)
			x = Dense(len(self.moves_ref)*20, activation='relu')(x)
			x = Dense(len(self.moves_ref)*6, activation='relu')(x)
			x = Dense(len(self.moves_ref)*2, activation='relu')(x)
			output_prob = Dense(len(self.moves_ref), activation='softmax', name='q_values')(x)
			
			
			self.model = Model(inputs = input_positions, outputs = output_prob)
			self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.nn_learningRate))


		#self.save()

	def save(self):
		if self.train_model:
			self.model.save(f"players/models/{self.model_name}.h5")


	# ----------------------------------------
	# 			Common internal methods
	# ----------------------------------------

	def normalize_nn_input(self, lineboard: np.ndarray) -> np.ndarray:
		return np.array([int(x == OWNER_NONE) for x in lineboard]).reshape(1,-1)


	# need keras rewrite
	def get_probs(self, normalized_lineboard) -> ([float], [float]):
		return self.model.predict(normalized_lineboard)[0]


	def reward(self, normalized_lineboard, predictions, moves_remaining):
		good = 0.85
		mean = 0.5
		bad = 0.15
		not_free = 0.05

		for id_tick in range(len(predictions)):
			if normalized_lineboard[id_tick] == 0:
				predictions[id_tick] = not_free
				continue

			neighbours, is_out = self.free_ticks_in_neighbours_boxes(id_tick, moves_remaining)

			box0_ticks = len(neighbours[0])
			box1_ticks = len(neighbours[1])

			if is_out[0]:
				if box1_ticks == 0:
					predictions[id_tick] = good
				elif box1_ticks == 1:
					predictions[id_tick] = bad
				else:
					predictions[id_tick] = mean

			elif is_out[1]:
				if box0_ticks == 0:
					predictions[id_tick] = good
				elif box0_ticks == 1:
					predictions[id_tick] = bad
				else:
					predictions[id_tick] = mean

			else:
				if box0_ticks == 0 or box1_ticks == 0:
					predictions[id_tick] = good
				elif box0_ticks == 1 or box1_ticks == 1:
					predictions[id_tick] = bad
				else:
					predictions[id_tick] = mean

				
				# if box0_ticks == 1 or box1_ticks == 1:
				# 	predictions[id_tick] = bad
				# elif box0_ticks == 0 or box1_ticks == 0: #éliminé les cas 1 précédemment
				# 	predictions[id_tick] = good
				# else:
				# 	predictions[id_tick] = mean

		return predictions


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def clear(self):
		self.ai_lineboard_by_turn = []
		self.ai_to_predict_by_turn = []

	def play(self, lineboard, moves_remaining):
		normalized_lineboard = self.normalize_nn_input(lineboard)

		# get model predictions
		probs = self.get_probs(normalized_lineboard)

		self.ai_lineboard_by_turn.append(normalized_lineboard)
		self.ai_to_predict_by_turn.append(self.reward(normalized_lineboard[0], probs.copy(), moves_remaining).reshape(1,-1))
		# print(probs)
		# print(self.ai_to_predict_by_turn[-1])
		# input("\n\n")

		# if self.nn_gamma > rd.random(): #de temps en temps, fait move aléatoire pour ne pas se figer
		# 	move = self.moves_ref.index(rd.choice(moves_remaining))
		# else:
			# discard already done moves
		probs[[index for index in range(len(probs)) if lineboard[index] != OWNER_NONE]] = -1
		
		# Our next move is the one with the highest probability after removing all illegal ones.
		move = np.argmax(probs)  # indice du meilleur mouvement à tracer


		# We return the selected move
		return self.moves_ref[move]

	def play_result(self, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass

	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass


	def train(self, lineboard, ai_score: int, other_score: int):
		if not self.train_model:
			return
			
		for board, target in zip(self.ai_lineboard_by_turn, self.ai_to_predict_by_turn):
			# print(board)
			# print(target)
			self.model.fit(board, target, verbose=0)


	def free_ticks_in_neighbours_boxes(self, id_tick, moves_remaining):
		free_tick_in_box1 = []
		free_tick_in_box2 = []
		box_out = [False, False]
		
		if self.moves_ref[id_tick][0] + 1 == self.moves_ref[id_tick][1]: #trait est ligne

			if self.ligne_haut(id_tick) == -1: #-1 si extérieur
				box_out[0] = True
			else:
				for tmp in [self.ligne_haut(id_tick), self.colonne_haut_gauche(id_tick), self.colonne_haut_droite(id_tick)]:
					if self.moves_ref[tmp] in moves_remaining : # libre
						free_tick_in_box1.append(tmp)
			
			if self.ligne_bas(id_tick) == -1: #-1 si extérieur
				box_out[1] = True
			else:
				for tmp in [self.ligne_bas(id_tick), self.colonne_bas_gauche(id_tick), self.colonne_bas_droite(id_tick)]:
					if self.moves_ref[tmp] in moves_remaining : # libre
						free_tick_in_box2.append(tmp)

		else: #trait est colonne
			
			if self.colonne_gauche(id_tick) == -1: #-1 si extérieur
				box_out[0] = True
			else:
				for tmp in [self.colonne_gauche(id_tick), self.ligne_haut_gauche(id_tick), self.ligne_bas_gauche(id_tick)]:
					if self.moves_ref[tmp] in moves_remaining : # libre
						free_tick_in_box1.append(tmp)
			
			if self.colonne_droite(id_tick) == -1: #-1 si extérieur
				box_out[1] = True
			else:
				for tmp in [self.colonne_droite(id_tick), self.ligne_haut_droite(id_tick), self.ligne_bas_droite(id_tick)]:
					if self.moves_ref[tmp] in moves_remaining : # libre
						free_tick_in_box2.append(tmp)

		return (free_tick_in_box1, free_tick_in_box2), box_out



	def ligne_haut(self, id_ligne):
		return id_ligne - 15 if id_ligne >= 15 else -1
	
	def ligne_bas(self, id_ligne):
		return id_ligne + 15 if id_ligne <= 96 else -1


	def colonne_gauche(self, id_colonne):
		return id_colonne - 1 if self.moves_ref[id_colonne][0] % 8 != 0 else -1
	
	def colonne_droite(self, id_colonne):
		return id_colonne + 1 if self.moves_ref[id_colonne][0] % 8 != 7 else -1


	def ligne_haut_gauche(self, id_colonne):
		id2 = self.moves_ref[id_colonne][0]
		id1 = id2-1 if id2 % 8 != 0 else -1
		return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

	def ligne_bas_gauche(self, id_colonne):
		id2 = self.moves_ref[id_colonne][1]
		id1 = id2-1 if id2 % 8 != 0 else -1 #pb here
		return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

	def ligne_haut_droite(self, id_colonne):
		id1 = self.moves_ref[id_colonne][0]
		id2 = id1+1 if id1 % 8 != 7 else -1
		return self.moves_ref.index((id1, id2)) if id2 != -1 else -1

	def ligne_bas_droite(self, id_colonne):
		id1 = self.moves_ref[id_colonne][1]
		id2 = id1+1 if id1 % 8 != 7 else -1
		return self.moves_ref.index((id1, id2)) if id2 != -1 else -1


	def colonne_haut_gauche(self, id_ligne):
		id2 = self.moves_ref[id_ligne][0]
		id1 = id2-8 if id2 > 7 else -1
		return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

	def colonne_haut_droite(self, id_ligne):
		id2 = self.moves_ref[id_ligne][1]
		id1 = id2-8 if id2 > 7 else -1
		return self.moves_ref.index((id1, id2)) if id1 != -1 else -1

	def colonne_bas_gauche(self, id_ligne):
		id1 = self.moves_ref[id_ligne][0]
		id2 = id1+8 if id1 < 56 else -1
		return self.moves_ref.index((id1, id2)) if id2 != -1 else -1

	def colonne_bas_droite(self, id_ligne):
		id1 = self.moves_ref[id_ligne][1]
		id2 = id1+8 if id1 < 56 else -1
		return self.moves_ref.index((id1, id2)) if id2 != -1 else -1


