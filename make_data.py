from preprocessing.preprocessing import *

games_dir = 'data/games/'
all_files = [f'{games_dir}{f}' for f in os.listdir(games_dir) if os.path.isfile(os.path.join(games_dir, f))]

Game2Dataset(all_files[1])

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(Game2Dataset, all_files)