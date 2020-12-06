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
		self.nn_learningRate = 0.0005 #min e^(-3) = 0.05
		
		self.ai_lineboard_by_turn = []
		self.ai_to_predict_by_turn = []

		self.high_score_boxes = {
			7 : (48, 55, 56, 63),
			3 : (16, 23, 24, 31,
				20, 27, 28, 35,
				76, 83, 84, 91,
				80, 87, 88, 95)
		}

		
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

	def normalize_nn_input(self, lineboard, moves_remaining):
		input_lineboard = []

		for id_tick in range(len(lineboard)):
			if lineboard[id_tick] != 0:
				input_lineboard.append(0)
			else:
				input_lineboard.append(self.evaluate_tick_value(id_tick, moves_remaining))

		return np.array(input_lineboard).reshape(1,-1)

		#return np.array([int(x == OWNER_NONE) for x in lineboard]).reshape(1,-1)


	# need keras rewrite
	def get_probs(self, normalized_lineboard) -> ([float], [float]):
		return self.model.predict(normalized_lineboard)[0]


	def target_lineboard(self, normalized_lineboard):
		#correspond à ce que la couche de sortie aurait dû prédire
		#softmax donc entre 0 et 1
		min_val = abs(min(normalized_lineboard))
		max_val = max(normalized_lineboard)
		interval = min_val+max_val
		
		return np.array([(min_val + value) / interval for value in normalized_lineboard]).reshape(1,-1)

	def print_lineboard(self, lineboard):
		counter = 0
		for i in range(7):
			print(" ",lineboard[counter : counter + 7])
			counter += 7
			print(lineboard[counter : counter + 8])
			counter += 8
		print(" ",lineboard[counter : counter + 6])


	# ----------------------------------------
	# 			Main methods
	# ----------------------------------------

	def clear(self):
		self.ai_lineboard_by_turn = []
		self.ai_to_predict_by_turn = []

	def play(self, lineboard, moves_remaining):
		normalized_lineboard = self.normalize_nn_input(lineboard, moves_remaining)

		# get model predictions
		probs = self.get_probs(normalized_lineboard)

		self.ai_lineboard_by_turn.append(normalized_lineboard)
		self.ai_to_predict_by_turn.append(self.target_lineboard(normalized_lineboard[0]))
		
		# self.print_lineboard(self.ai_lineboard_by_turn[-1][0])
		# self.print_lineboard(self.ai_to_predict_by_turn[-1][0])
		# input("\n\n")

		# if self.nn_gamma > rd.random(): #de temps en temps, fait move aléatoire pour ne pas se figer
		# 	move = self.moves_ref.index(rd.choice(moves_remaining))
		# else:
			# discard already done moves
		# move_before_correction = np.argmax(probs)
		# probs_before = probs.copy()

		probs[[index for index in range(len(probs)) if lineboard[index] != OWNER_NONE]] = -1
		
		# Our next move is the one with the highest probability after removing all illegal ones.
		move = np.argmax(probs)  # indice du meilleur mouvement à tracer

		# print(f"{move_before_correction} => {move}")
		# if move_before_correction != move:
		# 	print(probs_before)
		# 	input()

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



	def evaluate_tick_value(self, id_tick, moves_remaining):
		#calculer nombre de trait libres dans chacune des deux boites auxquelles il appartient
		free_tick_in_boxes, box_out = self.free_ticks_in_neighbours_boxes(id_tick, moves_remaining)

		value_tick = 0
		close_box = False
		value_next_play = 0
		
		for i, id_free in enumerate(free_tick_in_boxes): #0, 1, 2 ou 3 traits
			if box_out[i]:
				continue

			nb_free = len(id_free)

			if nb_free == 0: #ferme la boite
				close_box = True

				if id_tick in self.high_score_boxes[7]:
					value_tick += 14.0
				elif id_tick in self.high_score_boxes[3]:
					value_tick += 6.0
				else:
					value_tick += 2.0

			elif nb_free == 1: #laisse cuvette (si rejoue, top)
				
				if id_tick in self.high_score_boxes[7] and id_free[0] in self.high_score_boxes[7]:
					value_next_play = 6.9
				elif id_tick in self.high_score_boxes[3] and id_free[0] in self.high_score_boxes[3]:
					value_next_play = 2.9
				else:
					value_next_play = 0.9

				value_tick -= value_next_play

			else: #laisse angle ou entame boite
				if id_tick in self.high_score_boxes[7] and id_free[0] in self.high_score_boxes[7]:
					value_tick += 0.7
				elif id_tick in self.high_score_boxes[3] and id_free[0] in self.high_score_boxes[3]:
					value_tick += 0.3
				else:
					value_tick += 0.1
					
		if close_box and value_next_play != 0:
			value_tick += 2 * value_next_play #pressentir chaine
			
		return value_tick



	def free_ticks_in_neighbours_boxes(self, id_tick, moves_remaining):
		free_tick_in_box1 = []
		free_tick_in_box2 = []
		box_out = [False, False]

		# (0,1,2,3,4,5,6), 
		# (7,22,37,52,67,82,97), 
		# (14,29,44,59,74,89,104), 
		# (105,106,107,108,109,110,111)

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


