from .Progression import all_charts
from .PercentagesAndEdges import create_analysis_table

def charts():
    all_charts()
    print("Progression visualization complete.")
    
    create_analysis_table()
    print("Table statistics complete.")