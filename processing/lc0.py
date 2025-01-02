import chess
import chess.pgn
import chess.engine
import io
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging
import asyncio
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessAnalyzer:
    def __init__(self, engine_path: str, network_path: str):
        """
        Initialize the chess analyzer with Lc0 engine
        
        Args:
            engine_path: Path to the lc0 executable
            network_path: Path to the neural network weights
        """
        self.engine_path = Path(engine_path)
        self.network_path = Path(network_path)
        self.engine = None
        
    async def start_engine(self):
        """Start the Lc0 engine with appropriate settings"""
        transport, engine = await chess.engine.popen_uci(str(self.engine_path))
        # Configure engine options after initialization
        await engine.configure({
            "WeightsFile": str(self.network_path),
            "Backend": "cudnn-auto",
            "MinibatchSize": 256
        })
        self.engine = engine
        
    async def stop_engine(self):
        """Stop the engine properly"""
        if self.engine:
            await self.engine.quit()
            
    async def analyze_position(self, board: chess.Board, depth: int = 20) -> Optional[float]:
        """
        Analyze a single position and return the evaluation
        
        Args:
            board: Chess position to analyze
            depth: Search depth
            
        Returns:
            Evaluation in centipawns
        """
        try:
            info = await self.engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].relative.score(mate_score=10000)
            return score
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            return None

    async def analyze_game(self, pgn_text: str) -> Dict:
        """
        Analyze a complete game and calculate centipawn loss for each move
        
        Args:
            pgn_text: PGN text of the game
            
        Returns:
            Dictionary containing analysis results
        """
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return None
            
        result = {
            "event": game.headers.get("Event", "Unknown"),
            "white": game.headers.get("White", "Unknown"),
            "black": game.headers.get("Black", "Unknown"),
            "date": game.headers.get("Date", "Unknown"),
            "moves": []
        }
        
        board = game.board()
        node = game
        
        while node.variations:
            move = node.variations[0].move
            prev_eval = await self.analyze_position(board)
            
            # Make the move
            board.push(move)
            next_eval = await self.analyze_position(board)
            
            if prev_eval is not None and next_eval is not None:
                # Calculate centipawn loss
                if board.turn:  # Black just moved
                    loss = max(0, -(next_eval + prev_eval))
                else:  # White just moved
                    loss = max(0, next_eval - prev_eval)
                    
                move_data = {
                    "move": move.uci(),
                    "ply": len(board.move_stack),
                    "evaluation": next_eval,
                    "centipawn_loss": loss
                }
                result["moves"].append(move_data)
            
            node = node.variations[0]
            
        return result

async def analyze_pgn_file(pgn_path: str, engine_path: str, network_path: str, output_path: str):
    """
    Analyze all games in a PGN file
    
    Args:
        pgn_path: Path to input PGN file
        engine_path: Path to lc0 executable
        network_path: Path to neural network weights
        output_path: Path to save analysis results
    """
    analyzer = ChessAnalyzer(engine_path, network_path)
    await analyzer.start_engine()
    
    results = []
    
    try:
        with open(pgn_path) as pgn_file:
            while True:
                game_text = ""
                line = pgn_file.readline()
                
                if not line:
                    break
                    
                while line and not line.strip() == "":
                    game_text += line
                    line = pgn_file.readline()
                
                if game_text:
                    result = await analyzer.analyze_game(game_text)
                    if result:
                        results.append(result)
                        logger.info(f"Analyzed game: {result['white']} vs {result['black']}")
    
    finally:
        await analyzer.stop_engine()
        
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Analysis complete. Results saved to {output_path}")

def test_0():
    # Define paths for test
    pgn_path = "pgn/test.pgn"
    engine_path = "/home/cosmos/lc0/build/release/lc0"
    network_path = "/home/cosmos/Documents/Personal/BT4-1740.pb.gz"
    output_path = "json/test_analysis.json"
    
    # Run analysis
    asyncio.run(analyze_pgn_file(pgn_path, engine_path, network_path, output_path))

def main():
    test_0()

if __name__ == "__main__":
    main()