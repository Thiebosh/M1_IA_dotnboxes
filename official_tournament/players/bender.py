from random import *

# Init your variables here 

# Put your name Here 
name = "Bender"

# Bender strategy : return random available cell

def play(board, available_cells, player):
	return choice(available_cells)