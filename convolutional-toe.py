import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

board = np.zeros((3,3), dtype=int)

PLAYER_X = 1
PLAYER_O = -1

class TicTacToeAI(torch.nn.Module):

    def __init__(self):
        super(TicTacToeAI, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = x.view(-1, 1, 3, 3)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 32 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def is_game_over(board):
    for i in range(3):
        if np.sum(board[i, :]) == 3 or np.sum(board[:, i]) == 3:
            return True, PLAYER_X
        elif np.sum(board[i, :]) == -3 or np.sum(board[:, i]) == -3:
            return True, PLAYER_O

    if board[0, 0] + board[1, 1] + board[2, 2] == 3 or board[0, 2] + board[1, 1] + board[2,0] == 3:
        return True, PLAYER_X
    elif board[0, 0] + board[1, 1] + board[2, 2] == -3 or board[0, 2] + board[1, 1] + board[2,0] == -3:
        return True, PLAYER_O

    if np.all(board != 0):
        return True, O

    return False, None

def print_board(board):
    for row in board:
        print(" ".join(["X" if cell == PLAYER_X else "O" if cell == PLAYER_O else "-" for cell in row]))

def get_ai_move(board, ai_model):
    board_tensor = torch.tensor(board, dtype=torch.float32)
    board_flat = board_tensor.flatten()
    ai_model.eval()
    with torch.no_grad():
        prediction = ai_model(board_tensor.unsqueeze(0).unsqueeze(0))
    valid_moves = [i for i in range(9) if board_flat[i] == 0]
    probabilities = torch.softmax(prediction, dim=1).squeeze()
    ai_move_index = torch.argmax(probabilities[valid_moves])
    ai_move = valid_moves[ai_move_index]
    return ai_move // 3, ai_move % 3

def get_human_move():
    while True:
        try:
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column(0-2): "))
            if 0 <= row <= 2 and 0 <= col <= 2 and board[row, col] == 0:
                return row, col
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Please enter integers between 0 and 2.")

def play_game():
    ai_model = TicTacToeAI()

    print("Welcome to Alpin's Tic Tac Toe AI! You are X and the AI is O.")
    print_board(board)

    while True:
        row, col = get_human_move()
        board[row, col] = PLAYER_X
        print_board(board)
        game_over, winner = is_game_over(board)
        if game_over:
            if winner == PLAYER_X:
                print("Congratulations! You beat the AI!")
            elif winner == PLAYER_O:
                print("Boo! The AI won!")
            else:
                print("It's a draw! You have the same intelligence as this stupid AI!")
            break

        row, col = get_ai_move(board, ai_model)
        board[row, col] = PLAYER_O
        print("AI's move: ")
        print_board(board)
        game_over, winner = is_game_over(board)
        if game_over:
            if winner == PLAYER_X:
                print("Congratulations! You beat the AI!")
            elif winner == PLAYER_O:
                print("Boo! The AI won!")
            else:
                print("It's a draw! You have the same intelligence as this stupid AI!")
            break

if __name__ == "__main__":
    play_game()
