import numpy as np
import torch

board = np.zeros((3, 3), dtype=int)

PLAYER_X = 1
PLAYER_O = -1

class TicTacToeAI(torch.nn.Module):
    def __init__(self):
        super(TicTacToeAI, self).__init__()
        self.fc1 = torch.nn.Linear(9, 18)
        self.fc2 = torch.nn.Linear(18, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def is_game_over(board):
    for i in range(3):
        if np.sum(board[i, :]) == 3 or np.sum(board[:, i]) == 3:
            return True, PLAYER_X
        elif np.sum(board[i, :]) == -3 or np.sum(board[:, i]) == -3:
            return True, PLAYER_O

    if board[0, 0] + board[1, 1] + board[2, 2] == 3 or board[0, 2] + board[1, 1] + board[2, 0] == 3:
        return True, PLAYER_X
    elif board[0, 0] + board[1, 1] + board[2, 2] == -3 or board[0, 2] + board[1, 1] + board[2, 0] == -3:
        return True, PLAYER_O

    if np.all(board != 0):
        return True, 0

    return False, None

def print_board(board):
    for row in board:
        print(" ".join(["X" if cell == PLAYER_X else "O" if cell == PLAYER_O else "-" for cell in row]))

def get_ai_move(board, ai_model):
    board_flat = board.flatten()
    board_flat = torch.tensor(board_flat, dtype=torch.float32)
    ai_model.eval()
    with torch.no_grad():
        prediction = ai_model(board_flat)
    valid_moves = [i for i in range(9) if board_flat[i] == 0]
    ai_move = max(valid_moves, key=lambda i: prediction[i].item())
    return ai_move // 3, ai_move % 3

def get_human_move():
    while True:
        try:
            row = int(input("Enter the row (0-2): "))
            col = int(input("Enter the column (0-2): "))
            if 0 <= row <= 2 and 0 <= col <= 2 and board[row, col] == 0:
                return row, col
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Invalid input. Please enter integers.")

def play_game():
    ai_model = TicTacToeAI()

    print("Welcome to Tic Tac Toe! You are X and the AI is O.")
    print_board(board)

    while True:
        row, col = get_human_move()
        board[row, col] = PLAYER_X
        print_board(board)
        game_over, winner = is_game_over(board)
        if game_over:
            if winner == PLAYER_X:
                print("Congratulations! You win!")
            elif winner == PLAYER_O:
                print("AI wins. Better luck next time!")
            else:
                print("It's a draw!")
            break

        row, col = get_ai_move(board, ai_model)
        board[row, col] = PLAYER_O
        print("AI's Move:")
        print_board(board)
        game_over, winner = is_game_over(board)
        if game_over:
            if winner == PLAYER_X:
                print("Congratulations! You win!")
            elif winner == PLAYER_O:
                print("AI wins. Better luck next time!")
            else:
                print("It's a draw!")
            break

if __name__ == "__main__":
    play_game()
