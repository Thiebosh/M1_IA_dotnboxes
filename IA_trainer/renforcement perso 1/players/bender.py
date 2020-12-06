from random import choice

class Agent:

	def __init__(self, train, moves_list):
		self.name = "Bender" # put your name here
		


	def save(self):
		pass
	
	def play_result(self, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass

	def clear(self):
		pass

	def train(self, lineboard, ai_score: int, other_score: int):
		pass

	def play(self, lineboard, moves_remaining):
		return choice(moves_remaining) # random available cell
		
		
	def other_play(self, lineboard, other_move, ai_score, other_score, nb_moves_chained, remaining_moves_before):
		pass

	