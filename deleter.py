from pandas import pd
from tictoc import tictoc

@tictoc
def deleter(station, variable):
    """This function deletes the undesired variables in the merged and normalized 
    databases of the defined station. This is needed before implementing any method
    due to some variables having too many gaps. Only needed to be run once, after
    preprocessing the original files, technically.
    ----------
    Arguments:
    station -- the station number as an integer.
    variable -- the name of the variable to delete as a string.

    Return:
    None"""

    station = 916
    variables = ['absorbance']

    # Read the dfs
    merged = pd.read_csv(f'data/merged_{station}.csv', sep=',', encoding='utf-8')
    normed = pd.read_csv(f'data/normed_{station}.csv', sep=',', encoding='utf-8')

    for var in variables:

        merged.drop(f'{var}_{station}', inplace=True, axis=1)
        normed.drop(f'{var}_{station}', inplace=True, axis=1)

    # Save the files
    merged.to_csv(f'data/merged_{station}.csv', sep=',', encoding='utf-8', index=False)
    normed.to_csv(f'data/normed_{station}.csv', sep=',', encoding='utf-8', index=False)