import subprocess
import sys
import os
import argparse

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all necessary packages are installed
install_package('torch')
install_package('torchvision')
install_package('chess') # This will install python-chess
install_package('Pillow') 

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import chess
from PIL import Image

# c. The device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# d. reverse_piece_map dictionary
# This maps the numerical output of the model (0-12) to chess.Piece objects.
# 0: Empty, 1-6: White P, N, B, R, Q, K, 7-12: Black p, n, b, r, q, k
reverse_piece_map = {
    0: None, # Represents empty square
    1: chess.Piece(chess.PAWN, chess.WHITE),
    2: chess.Piece(chess.KNIGHT, chess.WHITE),
    3: chess.Piece(chess.BISHOP, chess.WHITE),
    4: chess.Piece(chess.ROOK, chess.WHITE),
    5: chess.Piece(chess.QUEEN, chess.WHITE),
    6: chess.Piece(chess.KING, chess.WHITE),
    7: chess.Piece(chess.PAWN, chess.BLACK),
    8: chess.Piece(chess.KNIGHT, chess.BLACK),
    9: chess.Piece(chess.BISHOP, chess.BLACK),
    10: chess.Piece(chess.ROOK, chess.BLACK),
    11: chess.Piece(chess.QUEEN, chess.BLACK),
    12: chess.Piece(chess.KING, chess.BLACK),
}

# e. piece_predictions_to_fen function
def piece_predictions_to_fen(piece_predictions_tensor, current_turn=chess.WHITE):
    # piece_predictions_tensor shape: (batch_size, 8*8*13), we expect batch_size=1
    num_classes_per_square = 13
    # Reshape predictions to (8, 8, 13) for easier square-wise processing
    reshaped_predictions = piece_predictions_tensor.view(8, 8, num_classes_per_square)

    board = chess.Board(None) # Create an empty board

    for i in range(8): # Rows (0 to 7, corresponds to rank 8 to 1)
        for j in range(8): # Columns (0 to 7, corresponds to file a to h)
            # Get predictions for the current square
            square_predictions = reshaped_predictions[i, j, :]
            # Get the predicted piece index (0-12)
            predicted_piece_idx = torch.argmax(square_predictions).item()

            # Convert to chess.Piece object
            piece = reverse_piece_map.get(predicted_piece_idx)

            # Place the piece on the board.
            # chess.square takes file (0-7 for a-h) and rank (0-7 for 1-8).
            # Our i (row) goes from 0 (rank 8) to 7 (rank 1), so 7 - i gives the chess rank.
            chess_square = chess.square(j, 7 - i)
            if piece is not None:
                board.set_piece_at(chess_square, piece)
            # If piece is None (predicted_piece_idx == 0), the square remains empty, which is correct.

    # Set basic FEN attributes (not predicted by the image-to-piece model)
    board.turn = current_turn # Default to white to move, as per instructions
    board.set_castling_fen('-') # Clear all castling rights, as per instructions
    board.ep_square = None # No en passant square, as per instructions
    board.halfmove_clock = 0 # Default halfmove clock
    board.fullmove_number = 1 # Default fullmove number

    return board.fen()

# g. torchvision.transforms.Compose for image preprocessing
# These must be consistent with the supervised training setup
preprocess_transforms = transforms.Compose([
    transforms.Resize((256, 256)),          # Resize images to 256x256
    transforms.ToTensor(),                  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485,0.456,0.406], # Normalize with ImageNet mean
                         std=[0.229,0.224,0.225])   # Normalize with ImageNet std
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert chess board PNG images to FEN notation.')
    parser.add_argument('--input_dir', type=str, default='/content/',
                        help='Directory containing PNG images of chess boards.')
    parser.add_argument('--weights_file', type=str, default='model_weights.pth',
                        help='Path to the trained model weights file (e.g., model_weights.pth).')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional: File to write FENs to. If not specified, FENs are printed to console.')
    args = parser.parse_args()

    print(f"Running PNG to FEN converter on device: {device}")

    # Instantiate a standard ResNet18 model with default weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final classification layer to match the one that was trained
    num_ftrs = model.fc.in_features
    num_classes_per_square = 13
    output_dim = 8 * 8 * num_classes_per_square # 64 squares * 13 classes = 832
    hidden_dim = 256 # Consistent with supervised training

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    # Load the model_weights.pth file
    weights_file = args.weights_file
    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Successfully loaded model weights from {weights_file}")
    else:
        print(f"Error: Weights file '{weights_file}' not found. Please ensure it's in the same directory as this script, or provide the full path.")
        sys.exit(1) # Exit if weights are not found

    # Set the model to evaluation mode
    model = model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Scan the input directory for .png files
    input_dir = args.input_dir
    all_files_in_input_dir = os.listdir(input_dir)
    # Filter for PNG files and sort them for consistent output
    png_files = sorted([f for f in all_files_in_input_dir if f.lower().endswith('.png')])

    if not png_files:
        print(f"No PNG files found in {input_dir}. Please ensure chess board PNG images are present.")
        sys.exit(0)

    output_lines = []
    print(f"Found {len(png_files)} PNG files in {input_dir}. Processing...")
    for png_file in png_files:
        file_path = os.path.join(input_dir, png_file)
        try:
            # 1. Load the image using PIL
            img = Image.open(file_path).convert('RGB') # Ensure image has 3 channels

            # 2. Apply the defined transforms
            input_tensor = preprocess_transforms(img)
            input_batch = input_tensor.unsqueeze(0) # Add a batch dimension (B, C, H, W)
            input_batch = input_batch.to(device) # Move tensor to the correct device

            # 3. Pass the transformed image through the model to get piece predictions
            with torch.no_grad(): # Disable gradient calculation for inference
                output = model(input_batch)

            # 4. Call piece_predictions_to_fen to convert these predictions to a FEN string
            predicted_fen = piece_predictions_to_fen(output.cpu())

            # 5. Store or print the result
            result_line = f"Filename: {png_file}, Predicted FEN: {predicted_fen}"
            output_lines.append(result_line)

        except Exception as e:
            error_line = f"Error processing {png_file}: {e}"
            output_lines.append(error_line)
            print(error_line) # Print errors to console even if writing to file

    if args.output_file:
        with open(args.output_file, 'w') as f:
            for line in output_lines:
                f.write(line + '
')
        print(f"Generated FENs saved to '{args.output_file}'")
    else:
        for line in output_lines:
            print(line)
