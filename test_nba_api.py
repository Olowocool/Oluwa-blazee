from nba_api.stats.endpoints import leaguegamefinder

games = leaguegamefinder.LeagueGameFinder().get_data_frames()[0]

print(games.head())
print(len(games))