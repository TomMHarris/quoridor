"""
Tests for the Quoridor game engine.
Run:  python -m pytest test_engine.py -v
  or: python test_engine.py
"""

import sys
from engine import QuoridorGame, MoveEncoder


# -- Helpers ---------------------------------------------------------------

def make_game(**overrides) -> QuoridorGame:
    """Quick 2-player game with optional overrides."""
    g = QuoridorGame(2)
    for k, v in overrides.items():
        setattr(g, k, v)
    return g


passed = 0
failed = 0

def check(condition: bool, name: str):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# -- Tests -----------------------------------------------------------------

def test_initial_state():
    print("\n-- Initial state --")
    g = QuoridorGame(2)
    check(g.pawns == [(8, 4), (0, 4)], "starting positions")
    check(g.current_player == 0, "player 0 starts")
    check(g.walls_remaining == [10, 10], "10 walls each")
    check(g.winner is None, "no winner yet")
    check(not g.is_over, "game not over")


def test_basic_pawn_moves():
    print("\n-- Basic pawn moves --")
    g = QuoridorGame(2)
    pawn_moves = g._get_pawn_moves(0)
    # P0 at (8,4): can go up (7,4), left (8,3), right (8,5) -- NOT down (off board)
    check((7, 4) in pawn_moves, "P0 can move up")
    check((8, 3) in pawn_moves, "P0 can move left")
    check((8, 5) in pawn_moves, "P0 can move right")
    check((9, 4) not in pawn_moves, "P0 cannot move off board")
    check(len(pawn_moves) == 3, "exactly 3 pawn moves from start")


def test_wall_blocks_movement():
    print("\n-- Wall blocks movement --")
    g = QuoridorGame(2)
    g.make_move(("move", (7, 4)))   # P0 forward
    g.make_move(("move", (1, 4)))   # P1 forward

    # Place horizontal wall above P0's new position
    g.make_move(("wall", (6, 3, "H")))  # blocks (6,3)<->(7,3) and (6,4)<->(7,4)
    # P0 is at (7,4) -- cannot move up through wall at (6,4)
    g_test = g.clone()
    g_test.current_player = 0
    moves = g_test._get_pawn_moves(0)
    check((6, 4) not in moves, "wall blocks upward movement")
    check((7, 3) in moves, "can still move left")


def test_wall_conflict_detection():
    print("\n-- Wall conflict detection --")
    g = QuoridorGame(2)
    g.make_move(("wall", (3, 3, "H")))

    # Same center -> conflict (crossing)
    check(g._wall_has_conflict(3, 3, "V"), "crossing wall conflicts")

    # Overlapping horizontal wall
    check(g._wall_has_conflict(3, 4, "H"), "overlapping H wall right")
    check(g._wall_has_conflict(3, 2, "H"), "overlapping H wall left")

    # Non-conflicting walls
    check(not g._wall_has_conflict(3, 1, "H"), "no conflict 2 spaces left")
    check(not g._wall_has_conflict(4, 3, "H"), "no conflict one row down")
    check(g._wall_has_conflict(3, 3, "H"), "same position same orient conflicts (center)")


def test_path_must_exist():
    print("\n-- Path must always exist --")
    g = QuoridorGame(2)
    g.walls_placed.add((0, 3, "H"))
    g.wall_centers.add((0, 3))
    g._add_wall_edges(0, 3, "H")
    g.walls_remaining[0] -= 1

    g.walls_placed.add((0, 5, "H"))
    g.wall_centers.add((0, 5))
    g._add_wall_edges(0, 5, "H")
    g.walls_remaining[0] -= 1

    check(g._has_path(1), "P1 still has path after partial blocking")


def test_straight_jump():
    print("\n-- Straight jump over opponent --")
    g = QuoridorGame(2)
    g.pawns = [(4, 4), (3, 4)]  # P0 below P1, adjacent
    g.current_player = 0
    moves = g._get_pawn_moves(0)
    check((2, 4) in moves, "P0 can jump over P1 to (2,4)")
    check((3, 4) not in moves, "P0 cannot move onto P1")


def test_diagonal_jump():
    print("\n-- Diagonal jump when straight blocked --")
    g = QuoridorGame(2)
    g.pawns = [(4, 4), (3, 4)]  # P0 below P1
    # Place wall behind P1 blocking straight jump
    g.walls_placed.add((2, 3, "H"))
    g.wall_centers.add((2, 3))
    g._add_wall_edges(2, 3, "H")  # blocks (2,3)<->(3,3) and (2,4)<->(3,4)

    g.current_player = 0
    moves = g._get_pawn_moves(0)
    check((2, 4) not in moves, "straight jump blocked by wall")
    check((3, 3) in moves, "diagonal jump left available")
    check((3, 5) in moves, "diagonal jump right available")


def test_diagonal_jump_wall_blocks_diagonal():
    print("\n-- Diagonal jump blocked by wall --")
    g = QuoridorGame(2)
    g.pawns = [(4, 4), (3, 4)]
    # Block straight jump behind P1
    g.walls_placed.add((2, 3, "H"))
    g.wall_centers.add((2, 3))
    g._add_wall_edges(2, 3, "H")

    # Also block the left diagonal
    g.v_blocked.add((3, 3))  # blocks (3,3)<->(3,4)

    g.current_player = 0
    moves = g._get_pawn_moves(0)
    check((3, 3) not in moves, "diagonal left blocked by vertical wall")
    check((3, 5) in moves, "diagonal right still available")


def test_win_detection():
    print("\n-- Win detection --")
    g = QuoridorGame(2)
    g.pawns = [(1, 4), (7, 4)]
    g.current_player = 0
    g.make_move(("move", (0, 4)))  # P0 reaches row 0
    check(g.winner == 0, "P0 wins reaching row 0")
    check(g.is_over, "game is over")
    check(len(g.get_legal_moves()) == 0, "no legal moves after game over")


def test_no_walls_remaining():
    print("\n-- No walls remaining --")
    g = QuoridorGame(2)
    g.walls_remaining = [0, 0]
    walls = g.get_legal_walls()
    check(len(walls) == 0, "no wall moves when 0 walls left")
    moves = g.get_legal_moves()
    check(all(m[0] == "move" for m in moves), "only pawn moves available")


def test_move_encoder_roundtrip():
    print("\n-- MoveEncoder round-trip --")
    g = QuoridorGame(2)
    pos = g.pawns[g.current_player]
    all_ok = True
    for move in g.get_legal_moves():
        idx = MoveEncoder.encode(move, pos)
        decoded = MoveEncoder.decode(idx, pos)
        if decoded != move:
            all_ok = False
            break
    check(all_ok, "all moves encode/decode correctly")

    mask = MoveEncoder.legal_mask(g)
    check(mask.sum() == len(g.get_legal_moves()), "legal mask count matches")
    check(mask.shape == (140,), "mask shape is (140,)")


def test_serialization_roundtrip():
    print("\n-- Serialization round-trip --")
    g = QuoridorGame(2)
    g.make_move(("move", (7, 4)))
    g.make_move(("wall", (3, 3, "H")))
    g.make_move(("move", (6, 4)))

    d = g.to_dict()
    g2 = QuoridorGame.from_dict(d)
    check(g2.pawns == g.pawns, "pawns match")
    check(g2.walls_placed == g.walls_placed, "walls match")
    check(g2.h_blocked == g.h_blocked, "h_blocked match")
    check(g2.v_blocked == g.v_blocked, "v_blocked match")
    check(g2.current_player == g.current_player, "current player matches")


def test_shortest_path():
    print("\n-- Shortest path --")
    g = QuoridorGame(2)
    check(g.shortest_path_length(0) == 8, "P0 starts 8 steps from goal")
    check(g.shortest_path_length(1) == 8, "P1 starts 8 steps from goal")

    path = g.shortest_path(0)
    check(path is not None, "path exists")
    check(path[0] == (8, 4), "path starts at pawn")
    check(path[-1][0] == 0, "path ends at goal row")
    check(len(path) == 9, "path has 9 cells (8 steps)")


def test_tensor_shape():
    print("\n-- Tensor shapes --")
    g2 = QuoridorGame(2)
    t2 = g2.to_tensor()
    check(t2.shape == (9, 9, 9), "2p tensor: (9, 9, 9)")
    check(t2.dtype.name == "float32", "tensor is float32")
    check(t2[0, 8, 4] == 1.0, "current player pawn marked")

    g4 = QuoridorGame(4)
    t4 = g4.to_tensor()
    check(t4.shape == (14, 9, 9), "4p tensor: (14, 9, 9)")


def test_four_player_basics():
    print("\n-- 4-player basics --")
    g = QuoridorGame(4)
    check(len(g.pawns) == 4, "4 pawns")
    check(g.walls_remaining == [5, 5, 5, 5], "5 walls each")

    # P0 at (8,4) can move up, left, right
    moves = g._get_pawn_moves(0)
    check((7, 4) in moves, "P0 can move up")
    check(len(moves) == 3, "P0 has 3 starting moves")

    # P2 at (4,0) can move up, down, right (not left -- board edge)
    moves2 = g._get_pawn_moves(2)
    check((4, 1) in moves2, "P2 can move right")
    check((3, 0) in moves2, "P2 can move up")
    check((5, 0) in moves2, "P2 can move down")
    check(len(moves2) == 3, "P2 has 3 starting moves")

    # Turn order
    g.make_move(("move", (7, 4)))
    check(g.current_player == 1, "turn passes to P1")
    g.make_move(("move", (1, 4)))
    check(g.current_player == 2, "turn passes to P2")
    g.make_move(("move", (4, 1)))
    check(g.current_player == 3, "turn passes to P3")
    g.make_move(("move", (4, 7)))
    check(g.current_player == 0, "turn wraps to P0")


def test_four_player_win():
    print("\n-- 4-player win --")
    g = QuoridorGame(4)
    g.pawns = [(1, 4), (7, 4), (4, 7), (4, 1)]
    g.current_player = 0
    g.make_move(("move", (0, 4)))
    check(g.winner == 0, "P0 wins reaching row 0 in 4p")


def test_clone_independence():
    print("\n-- Clone independence --")
    g = QuoridorGame(2)
    g.make_move(("move", (7, 4)))
    g2 = g.clone()
    g2.make_move(("move", (1, 4)))
    check(g.current_player == 1, "original unchanged after clone modified")
    check(g2.current_player == 0, "clone advanced correctly")
    check(g.pawns[1] == (0, 4), "original P1 unmoved")
    check(g2.pawns[1] == (1, 4), "clone P1 moved")


def test_jump_at_board_edge():
    print("\n-- Jump at board edge (straight blocked -> diagonal) --")
    g = QuoridorGame(2)
    g.pawns = [(0, 4), (1, 4)]  # P0 at top edge, P1 just below
    g.current_player = 1
    moves = g._get_pawn_moves(1)
    # P1 trying to jump over P0 upward -> (-1, 4) off board -> diagonal
    check((-1, 4) not in moves, "cannot jump off board")
    check((0, 3) in moves, "diagonal jump left at edge")
    check((0, 5) in moves, "diagonal jump right at edge")


# -- Run all ---------------------------------------------------------------

if __name__ == "__main__":
    test_initial_state()
    test_basic_pawn_moves()
    test_wall_blocks_movement()
    test_wall_conflict_detection()
    test_path_must_exist()
    test_straight_jump()
    test_diagonal_jump()
    test_diagonal_jump_wall_blocks_diagonal()
    test_win_detection()
    test_no_walls_remaining()
    test_move_encoder_roundtrip()
    test_serialization_roundtrip()
    test_shortest_path()
    test_tensor_shape()
    test_four_player_basics()
    test_four_player_win()
    test_clone_independence()
    test_jump_at_board_edge()

    print(f"\n{'=' * 40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    sys.exit(1 if failed else 0)
