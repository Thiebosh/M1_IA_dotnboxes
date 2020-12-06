import os
import time
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

OWNER_NONE = 0
OWNED = 1

BOARDSIZE = 8

name = "Thibaut Juzeau"

class Agent:

	# ----------------------------------------
	# 			Initialize Agent
	# ----------------------------------------

	def __init__(self):
		#global name

		self.model_name = name

		self.high_score_boxes = {
			7 : (48, 55, 56, 63),
			3 : (16, 23, 24, 31,
				20, 27, 28, 35,
				76, 83, 84, 91,
				80, 87, 88, 95)
		}
		self.nn_learningRate = 0.0005 #min e^(-3) = 0.05
		

		self.moves_ref = [] #referentiel d associations <id ligne - (id point 1, id point 2)> : [(id_pt1, id_pt2), ...]
		for lineIndex in range (BOARDSIZE-1):
			begin = BOARDSIZE*lineIndex
			end = BOARDSIZE*(lineIndex+1)
			self.moves_ref.extend((n,n+1) for n in range(begin, end-1))
			self.moves_ref.extend((n,n+BOARDSIZE) for n in range(begin, end))
		self.moves_ref.extend((n,n+1) for n in range(BOARDSIZE*(BOARDSIZE-1), BOARDSIZE*BOARDSIZE-1))
		
		#genere le plateau du point de vue des traits
		self.lineboard_init = [OWNER_NONE for i in range(len(self.moves_ref))]#etat du jeu du point de vue des traits : [libre, occupé, ...]
		self.lineboard = self.lineboard_init.copy()

		#initialise les traits restants
		self.moves_remaining_before_init = [self.moves_ref[i] for i in range(len(self.moves_ref))] # id(available_cells) a t-1
		self.moves_remaining_before = self.moves_remaining_before_init.copy()

		if os.path.isfile(f"players/{self.model_name}.h5"):
			#self.model = load_model(f"players/{self.model_name}.h5")
			self.create_model()
			self.model.load_weights(f"players/{self.model_name}.h5")

		else:
			print(f"\n\nwarning : model players/{self.model_name}.h5 not found\n\n")
			exit()

	
	def create_model(self):
		input_positions = Input(shape=(len(self.moves_ref),), name='inputs')
		x = Dense(len(self.moves_ref)*5, activation='relu')(input_positions)
		output_prob = Dense(len(self.moves_ref), activation='softmax', name='q_values')(x)
		
		self.model = Model(inputs = input_positions, outputs = output_prob)
		self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.nn_learningRate))


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

	def play(self, board, available_cells, player):
	#def play(self, lineboard, moves_remaining):
		if len(self.moves_remaining_before) < len(available_cells):
			print("fin de partie")
			#exit()
			self.lineboard = self.lineboard_init.copy()
			self.moves_remaining_before = self.moves_remaining_before_init.copy()

		#update lineboard and moves_remaining_before
		if len(self.moves_remaining_before) >= len(available_cells):
			played = [self.moves_ref.index(tuple_line) for tuple_line in self.moves_remaining_before if tuple_line not in available_cells]
			
			for index in played:
				self.lineboard[index] = OWNED #apply opponent moves
			
			self.moves_remaining_before = available_cells

		normalized_lineboard = self.normalize_nn_input(self.lineboard, available_cells)

		# get model predictions
		probs = self.get_probs(normalized_lineboard)
		probs[[index for index in range(len(probs)) if self.lineboard[index] != OWNER_NONE]] = -1
		move = np.argmax(probs)  # indice du meilleur mouvement à tracer

		if max(normalized_lineboard[0]) > normalized_lineboard[0][move]:
			move = np.argmax(normalized_lineboard[0])

		self.lineboard[move] = OWNED

		move = self.moves_ref[move]

		try:
			self.moves_remaining_before.remove(move)
		except:
			print("\n")
			print(move)
			print("\n")
			input("pause")

		# We return the selected move
		return move


	# ----------------------------------------
	# 			Common internal methods
	# ----------------------------------------

	def normalize_nn_input(self, lineboard, moves_remaining):
		return np.array([-100 if lineboard[x] != 0 else self.evaluate_tick_value(x, moves_remaining) for x in range(len(lineboard))]).reshape(1,-1)


	def get_probs(self, normalized_lineboard):
		return self.model.predict(normalized_lineboard)[0]


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

			elif nb_free == 1: #laisse cuvette
				
				#si ne ferme pas de boite, value_next_play ne sert pas : écrase
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

		if value_next_play != 0: #si ce trait forme une cuvette parmi un ou ses deux voisins
			
			if close_box: #je rejoue donc je peux la fermer : chaine pour moi
				value_tick = self.recursive_close_boxes(id_tick, free_tick_in_boxes, moves_remaining.copy()) #etudier chaine
			
			else: #je ne rejoue pas : chaine pour adversaire
				#simule mon coup pour voir ce que peut faire l'adversaire après mon coup
				next_moves_remaining1 = moves_remaining.copy()
				next_moves_remaining1.remove(self.moves_ref[id_tick])
				next_moves_remaining2 = next_moves_remaining1.copy()

				# #vérifie si chaine de côté 1
				first_side = 0
				if len(free_tick_in_boxes[0]) == 1:
					#donne possibilité de 0 et plateau mis a jour
					id_tick = free_tick_in_boxes[0][0]
					free_ticks_by_neighbours = self.free_ticks_in_neighbours_boxes(id_tick, next_moves_remaining1)[0]
					first_side = self.recursive_closable_boxes(id_tick, free_ticks_by_neighbours, next_moves_remaining1)

				# #vérifie si chaine de côté 0
				second_side = 0
				if len(free_tick_in_boxes[1]) == 1:
					#donne possibilité de 1 et plateau mis a jour
					id_tick = free_tick_in_boxes[1][0]
					free_ticks_by_neighbours = self.free_ticks_in_neighbours_boxes(id_tick, next_moves_remaining2)[0]
					second_side = self.recursive_closable_boxes(id_tick, free_ticks_by_neighbours, next_moves_remaining2)

				value_tick = first_side + second_side
			
		return value_tick


	def recursive_closable_boxes(self, id_tick, free_ticks_by_neighbours, moves_remaining): #argument côté : 0 ou 1 au début puis 2 par la suite?
		#sait qu'on laisse une cuvette, vérifie la longueur du côté demandé
		if id_tick in self.high_score_boxes[7]:
			box_value = -6.9
		elif id_tick in self.high_score_boxes[3]:
			box_value = -2.9
		else:
			box_value = -0.9	

		moves_remaining.remove(self.moves_ref[id_tick])

		if len(free_ticks_by_neighbours[0]) + len(free_ticks_by_neighbours[1]) == 1:
			new_tick = free_ticks_by_neighbours[0 if len(free_ticks_by_neighbours[0]) == 1 else 1][0]
			new_free_ticks_by_neighbours = self.free_ticks_in_neighbours_boxes(new_tick, moves_remaining)[0]
			
			return box_value + self.recursive_closable_boxes(new_tick, new_free_ticks_by_neighbours, moves_remaining)

		else:
			return box_value


	def recursive_close_boxes(self, id_tick, free_ticks_by_neighbours, moves_remaining):
		if id_tick in self.high_score_boxes[7]:
			box_value = 14
		elif id_tick in self.high_score_boxes[3]:
			box_value = 6
		else:
			box_value = 2

		moves_remaining.remove(self.moves_ref[id_tick])

		if len(free_ticks_by_neighbours[0]) + len(free_ticks_by_neighbours[1]) == 1:
			new_tick = free_ticks_by_neighbours[0 if len(free_ticks_by_neighbours[0]) == 1 else 1][0]
			new_free_ticks_by_neighbours = self.free_ticks_in_neighbours_boxes(new_tick, moves_remaining)[0]
			
			return box_value + self.recursive_close_boxes(new_tick, new_free_ticks_by_neighbours, moves_remaining)

		else:
			return box_value


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


thibaut_player = Agent()


def timing(f):
	def wrap(*args, **kwargs):
		time1 = time.time()
		ret = f(*args, **kwargs)
		time2 = time.time()
		print(f'function took {(time2-time1)*1000.0:.3f} ms')
		return ret
	return wrap

#@timing #~30ms / coup
def play(board, available_cells, player):
	return thibaut_player.play(board, available_cells, player)