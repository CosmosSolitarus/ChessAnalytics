import urllib.request
import json
import os

# Variables to be changed
username = "Cosmos_IV"  # Your Chess.com username
outputFile = "MyGamesCosmos_IV.pgn"  # The output file with PGN extension

# Do NOT modify from here
baseUrl = f"https://api.chess.com/pub/player/{username}/games/archives"

try:
    response = urllib.request.urlopen(baseUrl)
    data = json.loads(response.read())
    archivesList = data["archives"]
except Exception as e:
    print(f"Failed to retrieve archives: {e}")
    exit()

outputData = ""

# Process each archive
for i, archiveUrl in enumerate(archivesList):
    try:
        pgn_url = f"{archiveUrl}/pgn"
        inputFile = f"{archiveUrl.split('/')[-1]}.pgn"
        urllib.request.urlretrieve(pgn_url, inputFile)
        print(f"Archive {i+1}/{len(archivesList)}: {inputFile}")

        with open(inputFile) as fp:
            inputData = fp.read()

        # Reverse the order of games in the archive
        games = inputData.strip().split("\n\n\n")
        games.reverse()
        outputData += "\n\n\n".join(games) + "\n\n"
        
        os.remove(inputFile)
    except Exception as e:
        print(f"Failed to process archive {archiveUrl}: {e}")
        continue

# Write all data to the output file
try:
    with open(outputFile, 'w') as fp:
        fp.write(outputData)
    print(f"\nAll archives (including current month's games) appended to {outputFile} in chronological order.\n")
except Exception as e:
    print(f"Failed to write to output file: {e}")