"""Quick benchmark: random game playouts per second."""

import time
import random
from engine import QuoridorGame

def play_random_game(num_players=2) -> int:
    """Play a random game, return number of moves."""
    g = QuoridorGame(num_players)
    while not g.is_over:
        moves = g.get_legal_moves()
        # Bias toward pawn moves (faster games, more realistic)
        pawn_moves = [m for m in moves if m[0] == "move"]
        if random.random() < 0.7 and pawn_moves:
            g.make_move(random.choice(pawn_moves))
        else:
            g.make_move(random.choice(moves))
    return len(g.move_history)

# Warmup
play_random_game()

for np in [2, 4]:
    print(f"\n{'=' * 40}")
    print(f"  {np}-player random games")
    print(f"{'=' * 40}")

    n_games = 100
    t0 = time.perf_counter()
    total_moves = 0
    for _ in range(n_games):
        total_moves += play_random_game(np)
    elapsed = time.perf_counter() - t0

    avg_moves = total_moves / n_games
    games_per_sec = n_games / elapsed
    moves_per_sec = total_moves / elapsed

    print(f"  {n_games} games in {elapsed:.2f}s")
    print(f"  {games_per_sec:.1f} games/sec")
    print(f"  {moves_per_sec:.0f} moves/sec")
    print(f"  avg game length: {avg_moves:.1f} moves")
