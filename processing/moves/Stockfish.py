import chess
import chess.pgn
import chess.engine
import concurrent.futures
import statistics
import datetime
import os
import psutil

def analyze_game(game, engine_path, depth):
    """Analyze a single chess game and return centipawn loss for both players."""
    
    # Initialize engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        board = game.board()
        moves = list(game.mainline_moves())
        white_cpl = []
        black_cpl = []
        
        # Get starting position evaluation
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        last_score = info["score"].white()
        
        # Analyze each move
        for i, move in enumerate(moves):
            board.push(move)
            
            # Get position evaluation after move
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            current_score = info["score"].white()
            
            # Calculate centipawn loss
            if last_score.is_mate():
                # Handle mate scores
                mate_score = last_score.mate() * 10000
                if current_score.is_mate():
                    current_mate = current_score.mate() * 10000
                    loss = abs(mate_score - current_mate)
                else:
                    loss = abs(mate_score - current_score.score())
            else:
                if current_score.is_mate():
                    mate_score = current_score.mate() * 10000
                    loss = abs(last_score.score() - mate_score)
                else:
                    loss = max(0, abs(last_score.score() - current_score.score()))
            
            # Store centipawn loss for appropriate player
            if i % 2 == 0:  # White's move
                white_cpl.append(loss)
            else:  # Black's move
                black_cpl.append(loss)
            
            last_score = current_score
        
        # Calculate average centipawn loss
        white_avg_cpl = statistics.mean(white_cpl) if white_cpl else 0
        black_avg_cpl = statistics.mean(black_cpl) if black_cpl else 0
        
        return {
            "game": game.headers["Site"],
            "white_player": game.headers["White"],
            "black_player": game.headers["Black"],
            "white_avg_cpl": white_avg_cpl,
            "black_avg_cpl": black_avg_cpl,
            "white_moves_cpl": white_cpl,
            "black_moves_cpl": black_cpl
        }
        
    finally:
        engine.quit()

def process_pgn_file(pgn_path, engine_path, num_workers=7, depth=20):
    """Process multiple games from a PGN file using parallel workers."""
    
    results = []
    
    # Set process affinity to specific cores
    process = psutil.Process(os.getpid())
    
    # Get all available CPU cores and exclude the last one
    available_cores = list(range(psutil.cpu_count()))
    cores_to_use = available_cores[:-1]  # Exclude the last core
    process.cpu_affinity(cores_to_use)
    
    with open(pgn_path) as pgn_file:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Read games and submit them for analysis
            futures = []
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                futures.append(executor.submit(analyze_game, game, engine_path, depth))
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed analysis of {result['game']}")
                except Exception as e:
                    print(f"Error analyzing game: {e}")
    
    return results

def test0():
    pgn_path = "pgn/MyGamesCosmos_IV.pgn"
    engine_path = "/usr/games/stockfish"
    depth = 20
    
    results = process_pgn_file(pgn_path, engine_path, depth)
    
    # Print or save results
    for result in results:
        print(f"\nGame: {result['game']}")
        print(f"White ({result['white_player']}): {result['white_avg_cpl']:.2f} average centipawn loss")
        print(f"Black ({result['black_player']}): {result['black_avg_cpl']:.2f} average centipawn loss")

def test1():
    print(f"Start time: {datetime.datetime.now()}")
    
    pgn_path = "pgn/test.pgn"
    engine_path = "/usr/games/stockfish"
    depth = 20
    
    results = process_pgn_file(pgn_path, engine_path, depth)
    
    # Print or save results
    for result in results:
        print(f"\nGame: {result['game']}")
        print(f"White ({result['white_player']}): {result['white_avg_cpl']:.2f} average centipawn loss")
        print(f"Black ({result['black_player']}): {result['black_avg_cpl']:.2f} average centipawn loss")

    print(f"End time: {datetime.datetime.now()}")

def main():
    #test0()
    test1()

if __name__ == "__main__":
    main()