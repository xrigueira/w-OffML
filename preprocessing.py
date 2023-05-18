import os

from checkGaps import checkGaps
from joiner import joiner

# Define the data we want to study
files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]

varNames = [i[0:-4] for i in files] # Extract the names of the variables, e.g.: pH_901.txt (4 intervals already appended following temporal sequence)
stations = [901, 902, 904, 905, 906, 907, 910, 916] # Define with stations to process

# Define the time frame we want to use (a: months, b: weeks, c: days)
timeFrame = 'b'
timeStep = '15 min'

if __name__ == '__main__':

    for varName in varNames:
        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.txt', timestep=timeStep)
        print('[INFO] checkGaps() DONE')
        
    
    for station in stations:
        # Join the databases by station number
        joiner(station=station)
        print(f'[INFO] joiner() DONE | station: {station}')
