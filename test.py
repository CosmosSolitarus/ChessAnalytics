import re

# Your raw chess data as a string
data = "1. d4 {[%clk 0:09:59.9]} 1... d6 {[%clk 0:09:57.8]} 2. c4 {[%clk 0:09:58.8]} 2... a5 {[%clk 0:09:52.5]} 3. d5 {[%clk 0:09:57.6]} 3... c5 {[%clk 0:09:47.2]} 4. e4 {[%clk 0:09:54.8]} 4... e5 {[%clk 0:09:31.4]} 5. Nc3 {[%clk 0:09:53.6]} 5... Ne7 {[%clk 0:09:26.8]} 6. b3 {[%clk 0:09:52.4]} 6... g6 {[%clk 0:09:14]} 7. Bg5 {[%clk 0:09:50.8]} 7... h6 {[%clk 0:08:56]} 8. Bxe7 {[%clk 0:09:43.4]} 8... Bxe7 {[%clk 0:08:54]} 9. Be2 {[%clk 0:09:37.5]} 9... g5 {[%clk 0:08:51.6]} 10. g4 {[%clk 0:09:33]} 10... h5 {[%clk 0:08:48.2]} 11. h3 {[%clk 0:09:11.6]} 11... f6 {[%clk 0:08:27.9]} 12. Qd2 {[%clk 0:09:02.2]} 12... Na6 {[%clk 0:08:12.2]} 13. O-O-O {[%clk 0:08:56.8]} 13... Nb4 {[%clk 0:08:09.2]} 14. a3 {[%clk 0:08:51.2]} 14... Na6 {[%clk 0:07:59.8]} 15. Qb2 {[%clk 0:08:42.2]} 15... Bd7 {[%clk 0:07:50.5]} 16. gxh5 {[%clk 0:08:29.7]} 16... f5 {[%clk 0:07:37.2]} 17. Rd3 {[%clk 0:08:12.9]} 17... g4 {[%clk 0:07:16.3]} 18. hxg4 {[%clk 0:08:06.9]} 18... Bg5+ {[%clk 0:07:10.5]} 19. Kd1 {[%clk 0:07:57.9]} 19... fxg4 {[%clk 0:06:48.2]} 20. Rg3 {[%clk 0:07:39.8]} 20... Qf6 {[%clk 0:06:28.6]} 21. f3 {[%clk 0:07:19]} 21... Rg8 {[%clk 0:05:57.7]} 22. fxg4 {[%clk 0:07:16.5]} 22... Bf4 {[%clk 0:05:49.8]} 23. Rf3 {[%clk 0:07:10.3]} 23... Bxg4 {[%clk 0:05:27]} 24. Rf2 {[%clk 0:06:59.7]} 24... Bxe2+ {[%clk 0:05:12.7]} 25. Qxe2 {[%clk 0:06:49.2]} 25... O-O-O {[%clk 0:05:02.9]} 26. Nh3 {[%clk 0:06:41]} 26... Rdf8 {[%clk 0:04:27.2]} 27. Rhf1 {[%clk 0:06:37.7]} 27... Qh6 {[%clk 0:04:21.8]} 28. Nxf4 {[%clk 0:05:50.3]} 28... exf4 {[%clk 0:04:18.5]} 29. Rh2 {[%clk 0:05:29.8]} 29... Rg3 {[%clk 0:03:56.8]} 30. Rf3 {[%clk 0:05:10.9]} 30... Rg1+ {[%clk 0:03:52.6]} 31. Kd2 {[%clk 0:04:59.4]} 31... Ra1 {[%clk 0:03:32.4]} 32. a4 {[%clk 0:04:44.5]} 32... Ra3 {[%clk 0:03:26.8]} 33. Ke1 {[%clk 0:03:29.4]} 33... Rxb3 {[%clk 0:03:17.2]} 34. Nd1 {[%clk 0:03:19.1]} 34... Rb1 {[%clk 0:02:40.1]} 35. Qc2 {[%clk 0:03:02.8]} 35... Ra1 {[%clk 0:02:24.7]} 36. Qb2 {[%clk 0:02:59.9]} 36... Rxa4 {[%clk 0:02:17]} 37. Qb3 {[%clk 0:02:32.3]} 37... Ra1 {[%clk 0:01:58.9]} 38. Ra2 {[%clk 0:02:27.7]} 38... Rxa2 {[%clk 0:01:39.9]} 39. Qxa2 {[%clk 0:02:26.1]} 39... Qxh5 {[%clk 0:01:37.8]} 40. Qa3 {[%clk 0:02:11]} 40... Nb4 {[%clk 0:01:24.9]} 41. Rh3 {[%clk 0:02:01.9]} 41... Nc2+ {[%clk 0:01:20.4]} 0-1"

# Regex pattern to capture moves after the ". " (a period followed by a space) and stopping at the next space or brace
move_pattern = r'\.\s([a-zA-Z0-9]+)\s*[{]'

# Regex pattern to capture the times (assuming they are always in [%clk ...] format)
time_pattern = r'\[%clk (\d+:\d+:\d+\.\d+)\]'

# Extract moves using the defined pattern
moves = re.findall(move_pattern, data)

# Extract times using the defined time pattern
times = re.findall(time_pattern, data)

# Create separate lists for white and black moves based on alternating turns
white_moves = moves[::2]  # White moves are at even indices
black_moves = moves[1::2]  # Black moves are at odd indices

# Output the results
print("White Moves:", white_moves)
print("Black Moves:", black_moves)
print("Times:", times)