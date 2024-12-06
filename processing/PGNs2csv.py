import chess.pgn
import pytz
import csv
import re
from datetime import datetime, timedelta

tests = 5

# Define the time zones
utc = pytz.utc
est = pytz.timezone("US/Eastern")

def simplify_termination(description):
    description = description.lower()
    if "resignation" in description:
        return "resignation"
    elif "checkmate" in description:
        return "checkmate"
    elif "abandoned" in description:
        return "abandoned"
    elif "time" in description and "insufficient material" in description:
        return "timeout vs insufficient material"
    elif "time" in description:
        return "timeout"
    elif "insufficient material" in description:
        return "insufficient material"
    elif "stalemate" in description:
        return "stalemate"
    elif "agreement" in description:
        return "agreement"
    elif "repetition" in description:
        return "repetition"
    elif "50-move" in description or "50 moves" in description:
        return "50 move rule"
    return "unknown"

def moves_splitter(moves, color, time_control):
    # Regex pattern to capture moves after the ". " (a period followed by a space) and stopping at the next space or brace
    move_pattern = r'\.\s([a-zA-Z0-9\-+#]+)\s*[{]'

    # Regex pattern to capture the times (assuming they are always in [%clk ...] format)
    time_pattern = r'\[%clk (\d+:\d+:\d+(\.\d+)?)\]'

    # Extract moves using the defined pattern
    parsed_moves = re.findall(move_pattern, moves)

    # Extract times using the defined time pattern
    parsed_times = re.findall(time_pattern, moves)
    parsed_times = [t[0] for t in re.findall(time_pattern, moves)]

    # Count moves
    if color == "white":
        my_num_moves = len(parsed_moves[::2])
        opp_num_moves = len(parsed_moves[1::2])
    else:
        my_num_moves = len(parsed_moves[1::2])
        opp_num_moves = len(parsed_moves[::2])

    # Convert time control to seconds
    time_control_seconds = int(time_control)

    # Calculate total time used
    my_times = parsed_times[::2] if color == "white" else parsed_times[1::2]
    opp_times = parsed_times[1::2] if color == "white" else parsed_times[::2]

    # Calculate time used
    my_last_time = my_times[-1] if my_times else "0:00:00"
    opp_last_time = opp_times[-1] if opp_times else "0:00:00"

    my_total_time = time_control_seconds - time_to_seconds(my_last_time)
    opp_total_time = time_control_seconds - time_to_seconds(opp_last_time)

    # Calculate average time per move
    my_avg_tpm = round(my_total_time / my_num_moves, 2) if my_num_moves > 0 else 0
    opp_avg_tpm = round(opp_total_time / opp_num_moves, 2) if opp_num_moves > 0 else 0

    return parsed_moves, parsed_times, my_num_moves, opp_num_moves, round(my_total_time, 2), round(opp_total_time, 2), my_avg_tpm, opp_avg_tpm

def time_to_seconds(time_str):
    # Split the time string into hours, minutes, seconds, and fractional seconds
    hours, minutes, seconds = time_str.split(':')
    
    # Further split the seconds into whole and fractional parts if necessary
    if '.' in seconds:
        whole_seconds, fractional_seconds = seconds.split('.')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(whole_seconds) + float(f'0.{fractional_seconds}')
    else:
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    
    return total_seconds

def analyze_castling(moves, color):
    i_castled_first = False
    i_castled_short = False
    i_castled_long = False
    opp_castled_short = False
    opp_castled_long = False
    
    opp_castled = False
    
    my_castling_move = "N/A"
    opp_castling_move = "N/A"

    is_my_move = color == "white"  # Start with white's move
    
    move_count = 1

    for move in moves:
        if move == "O-O":
            if is_my_move:
                i_castled_short = True
                my_castling_move = move_count

                if not opp_castled:
                    i_castled_first = True
            else:
                opp_castled_short = True
                opp_castling_move = move_count
                opp_castled = True
        elif move == "O-O-O":
            if is_my_move:
                i_castled_long = True
                my_castling_move = move_count

                if not opp_castled:
                    i_castled_first = True
            else:
                opp_castled_long = True
                opp_castling_move = move_count
                opp_castled = True

        is_my_move = not is_my_move  # Switch to the other player's move
        if is_my_move == (color == "white"):
            move_count += 1

    return  (i_castled_first, i_castled_short, i_castled_long, opp_castled_short, 
             opp_castled_long, my_castling_move, opp_castling_move)

def get_hour_of_day(start_datetime):
    return start_datetime.hour + 1  # Convert to 1-24 range

def get_day_of_week(start_datetime):
    # Python's weekday() returns 0=Monday, ..., 6=Sunday
    return (start_datetime.weekday() + 1) % 7 + 1  # Convert to 1=Sunday, ..., 7=Saturday

def get_game_of_day_and_week(current_time, daily_game_times, weekly_game_times):
    day_start = current_time - timedelta(hours=24)
    week_start = current_time - timedelta(days=7)
    
    game_of_day = sum(1 for t in daily_game_times if t > day_start)
    game_of_week = sum(1 for t in weekly_game_times if t > week_start)
    
    return game_of_day + 1, game_of_week + 1

# Open the PGN file
with open("pgn/MyGamesCosmosSolitarus.pgn") as pgn_file:
    # Prepare to write to MyGames.csv
    with open("csv/MyGamesCosmosSolitarus.csv", mode="w", newline="") as csv_file:
        gamefieldnames = [  "Account", "GameNumber", "Date", "StartTime", 
                            "DayOfWeek", "HourOfDay", "GameOfDay", "GameOfWeek", 
                            "TimeControl", "MyElo", "OppElo", 
                            "Color", "ECO", "Result", "Termination",
                            "ICastledFirst", "ICastledShort", "ICastledLong",
                            "OppCastledShort", "OppCastledLong", 
                            "LastResult", "2ndLastResult", 
                            "MyNumMoves", "OppNumMoves", "MyTotalTime", "OppTotalTime",
                            "MyAvgTPM", "OppAvgTPM", "MyCastlingMove", "OppCastlingMove",
                            "TimeSinceLast"]
        games_writer = csv.DictWriter(csv_file, fieldnames=gamefieldnames)
        games_writer.writeheader()

        # Prepare to write to Moves.csv
        with open("csv/MovesCosmosSolitarus.csv", mode="w", newline="") as moves_file:
            movefieldnames = ["Account", "GameNumber",  
                              "MoveNumber", "Move", "Time"]
            moves_writer = csv.DictWriter(moves_file, fieldnames=movefieldnames)
            moves_writer.writeheader()

            game_number = 1
            daily_game_times = []
            weekly_game_times = []

            # Variables to store last two results
            last_result = "N/A"
            second_last_result = "N/A"

            # Variable to store the end time of the last game
            last_game_end_time = None

            while True:
                # Read the next game from the PGN file
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if (game.headers.get("TimeControl") == "600" or game.headers.get("TimeControl") == "300"):
                    # Extract headers
                    headers = game.headers
                    utc_date = headers.get("UTCDate")
                    utc_time = headers.get("UTCTime")

                    # Convert UTC date and time to datetime object
                    utc_datetime = datetime.strptime(f"{utc_date} {utc_time}", "%Y.%m.%d %H:%M:%S")
                    start_datetime = utc.localize(utc_datetime).astimezone(est)

                    # Determine the color, result, and ELO ratings
                    white_player = headers.get("White")
                    black_player = headers.get("Black")
                    result = headers.get("Result")
                    my_elo = headers.get("WhiteElo") if white_player == "CosmosSolitarus" else headers.get("BlackElo")
                    opp_elo = headers.get("BlackElo") if white_player == "CosmosSolitarus" else headers.get("WhiteElo")
                    color = "white" if white_player == "CosmosSolitarus" else "black"
                    result_text = "draw"
                    if result == "1-0" and color == "white":
                        result_text = "won"
                    elif result == "0-1" and color == "black":
                        result_text = "won"
                    elif result == "1-0" and color == "black":
                        result_text = "lost"
                    elif result == "0-1" and color == "white":
                        result_text = "lost"

                    # Simplify termination
                    termination_description = headers.get("Termination", "")
                    termination_simplified = simplify_termination(termination_description)

                    # Extract moves and times
                    try:
                        moves, times, my_num_moves, opp_num_moves, my_total_time, opp_total_time, my_avg_tpm, opp_avg_tpm = moves_splitter(str(game[0]), color, headers.get("TimeControl"))
                    except Exception as e:
                        # Print the error message, game number, and details of the game
                        print(f"Error processing game {game_number}: {e}")
                        print("White Player:", headers.get("White"))
                        print("Black Player:", headers.get("Black"))
                        print("Game Termination:", headers.get("Termination"))
                        print("Game Result:", headers.get("Result"))
                        print("TimeControl:", headers.get("TimeControl"))
                        
                        # Skip the game and continue processing
                        continue  # Skip this game and move to the next

                    # Analyze castling
                    i_castled_first, i_castled_short, i_castled_long, opp_castled_short, opp_castled_long, my_castling_move, opp_castling_move = analyze_castling(moves, color)

                    # Calculate GameOfDay and GameOfWeek
                    game_of_day, game_of_week = get_game_of_day_and_week(start_datetime, daily_game_times, weekly_game_times)

                    # Add new game time
                    daily_game_times.append(start_datetime)
                    weekly_game_times.append(start_datetime)

                    # Remove old timestamps from lists
                    day_start = start_datetime - timedelta(hours=24)
                    week_start = start_datetime - timedelta(days=7)
                    daily_game_times = [t for t in daily_game_times if t > day_start]
                    weekly_game_times = [t for t in weekly_game_times if t > week_start]

                    # Calculate TimeSinceLast
                    time_since_last = "N/A"
                    if last_game_end_time:
                        time_since_last = (start_datetime - last_game_end_time).total_seconds()

                    # Write game details to MyGames.csv
                    games_writer.writerow({
                        "Account": "CosmosSolitarus",
                        "GameNumber": game_number,
                        "Date": start_datetime.strftime("%Y-%m-%d"),
                        "StartTime": start_datetime.strftime("%H:%M:%S"),
                        "DayOfWeek": get_day_of_week(start_datetime),
                        "HourOfDay": get_hour_of_day(start_datetime),
                        "GameOfDay": game_of_day,
                        "GameOfWeek": game_of_week,
                        "TimeControl": headers.get("TimeControl"),
                        "MyElo": my_elo,
                        "OppElo": opp_elo,
                        "Color": color,
                        "ECO": headers.get("ECO"),
                        "Result": result_text,
                        "Termination": termination_simplified,
                        "ICastledFirst": "Yes" if i_castled_first else "No",
                        "ICastledShort": "Yes" if i_castled_short else "No",
                        "ICastledLong": "Yes" if i_castled_long else "No",
                        "OppCastledShort": "Yes" if opp_castled_short else "No",
                        "OppCastledLong": "Yes" if opp_castled_long else "No",
                        "LastResult": last_result,
                        "2ndLastResult": second_last_result,
                        "MyNumMoves": my_num_moves,
                        "OppNumMoves": opp_num_moves,
                        "MyTotalTime": my_total_time,
                        "OppTotalTime": opp_total_time,
                        "MyAvgTPM": my_avg_tpm,
                        "OppAvgTPM": opp_avg_tpm,
                        "MyCastlingMove": my_castling_move,
                        "OppCastlingMove": opp_castling_move,
                        "TimeSinceLast": f"{time_since_last:.2f}" if time_since_last != "N/A" else "N/A"
                    })

                    # Add Moves.csv entries
                    move_number = 1
                    increment_flag = False

                    for i in range(0, len(moves)):
                        move = moves[i]
                        time = times[i]

                        moves_writer.writerow({
                            "Account": "CosmosSolitarus", 
                            "GameNumber": game_number, 
                            "MoveNumber": move_number, 
                            "Move": move, 
                            "Time": time
                        })
                        
                        # Increment move_number only for white's move
                        if increment_flag:
                            move_number += 1
        
                        increment_flag = not increment_flag

                    # Update result history
                    second_last_result = last_result
                    last_result = result_text
                    
                    # Update the end time of the current game
                    last_game_end_time = start_datetime + timedelta(seconds=my_total_time)  # Assuming total time is in seconds

                    game_number += 1

print("Conversion complete! The data has been saved to MyGames.csv and Moves.csv.")