import urllib.request
import json
import os

def get_pgn(username):
    """
    Fetches and processes PGN game archives for a given Chess.com username.

    Parameters:
        username (str): Chess.com username to fetch archives for.
    """
    output_file = f"pgn/MyGames{username}.pgn"  # Output file with PGN extension

    # Construct the base URL for archives
    base_url = f"https://api.chess.com/pub/player/{username}/games/archives"

    try:
        response = urllib.request.urlopen(base_url)
        data = json.loads(response.read())
        archives_list = data["archives"]
    except Exception as e:
        print(f"Failed to retrieve archives for {username}: {e}")
        return

    output_data = ""

    # Process each archive
    for i, archive_url in enumerate(archives_list):
        try:
            pgn_url = f"{archive_url}/pgn"
            input_file = f"{archive_url.split('/')[-1]}.pgn"
            urllib.request.urlretrieve(pgn_url, input_file)
            print(f"Archive {i+1}/{len(archives_list)}: {input_file}")

            with open(input_file) as fp:
                input_data = fp.read()

            # Reverse the order of games in the archive
            games = input_data.strip().split("\n\n\n")
            games.reverse()
            output_data += "\n\n\n".join(games) + "\n\n"

            os.remove(input_file)
        except Exception as e:
            print(f"Failed to process archive {archive_url}: {e}")
            continue

    # Write all data to the output file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as fp:
            fp.write(output_data)
        print(f"\nAll archives (including current month's games) appended to {output_file} in chronological order.\n")
    except Exception as e:
        print(f"Failed to write to output file: {e}")
