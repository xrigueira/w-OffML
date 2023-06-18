"""This file gets the percentage of gaps in each variable
of each merged file"""

import pandas as pd

from tictoc import tictoc
from matplotlib import pyplot as plt
plt.style.use('ggplot')

def labeler(varname):

    """This function is just to label the plots correctly for the research papers."""

    if varname == 'ammonium':
        label_title = r'$NH_4$'
        label_y_axis = r'$NH_4$ ' + r'$(m*g/L)$'
    elif varname == 'conductivity':
        label_title = r'Conductivity'
        label_y_axis = r'Conductivity ' r'$(\mu*S/cm)$'
    elif varname == 'nitrates':
        label_title = r'$NO_{3^-}$'
        label_y_axis = r'$NO_{3^-}$ ' +r'$(m*g/L)$'
    elif varname == 'dissolved_oxygen':
        label_title = r'$O_2$'
        label_y_axis = r'$O_2$ ' r'$(m*g/L)$'
    elif varname == 'pH':
        label_title = r'pH'
        label_y_axis = r'pH'
    elif varname == 'temperature':
        label_title = r'Temperature'
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'water_temperature':
        label_title = r'River water remperature '
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'water_flow':
        label_title = r'Flow'
        label_y_axis = r'Flow ' + r'($m^3/s$)'
    elif varname == "turbidity":
        label_title = r'Turbidity'
        label_y_axis = r'Turbidity ' + r'(NTU)'
    elif varname == "rainfall":
        label_title = r'Rainfall'
        label_y_axis = r'Rainfall ' + r'(mm)'
    elif varname == "water_level":
        label_title = r'Water level'
        label_y_axis = r'Water level ' + r'(m)'
    
    return label_title, label_y_axis

@tictoc
def multivar_plotter(station):
    """This computes a multivariate plot of each anomaly in a 
    specific database.
    ----------
    Arguments:
    None

    Return:
    None"""

    # Improve https://www.pythoncharts.com/
    
    # Read the database
    df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8', parse_dates=['date'])
    
    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    
    # Filter the data to select only rows where the label column has a value of 1
    df_index = df[df["label"] == 1]
    
    # Create a new column with the difference between consecutive dates
    date_diff = (df_index['date'] - df_index['date'].shift()).fillna(pd.Timedelta(minutes=15))

    # Create groups of consecutive dates
    date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

    # Get the starting and ending indexes of each group of consecutive dates
    grouped = df.groupby(date_group)
    consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]
    
    # Set date as the index column
    df.set_index('date', inplace=True)
    
    # Drop the label column
    df.drop(df.columns[-1], axis=1, inplace=True)
    
    # Plot each anomaly
    counter = 1
    for i in consecutive_dates_indexes:

        fig = df.iloc[int(i[0]):int(i[1]), :].plot(figsize=(10,5))
        plt.title(f"Anomaly {counter} station {station}")
        plt.xlabel('Date')
        plt.ylabel('Standarized values')
        # plt.show()
        
        # Save the image
        fig = fig.get_figure()
        fig.subplots_adjust(bottom=0.19)
        fig.savefig(f'images/anomaly_{station}_{counter}.png', dpi=300)
        
        # Close the fig for better memory management
        plt.close(fig=fig)
        
        counter += 1

def window_plotter(station, group_size, step_size):
    
    # Read the data
    data = pd.read_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8')

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data.iloc[:, 1:-1] = scaler.fit_transform(data.iloc[:, 1:-1])

    # Convert variable columns to np.ndarray
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    
    # Make groups of data
    groups = []
    for i in range(0, X.shape[0] - group_size + 1, step_size):
        group = X[i:i + group_size]
        groups.append(group)
    
    # Make groups of labels
    labels = []
    for i in range(0, y.shape[0] - group_size + 1, step_size):
        grouped_labels = y[i:i + group_size]
        label = sum(grouped_labels) / len(grouped_labels)
        labels.append(label)
    
    counter = 0
    for group, label in zip(groups, labels):
        
        if label != 0:
            
            # Convert to data frame
            df = pd.DataFrame(group, columns=data.columns[1:-1])
            
            # Plot the data frame
            fig = df.plot(figsize=(10, 5))
            plt.title(f'Window {counter} station {station} label {label}')
            # plt.show()
            
            # Save the image
            fig = fig.get_figure()
            fig.savefig(f'windows/window_{counter}_{station}.png', dpi=100)
            
            plt.close(fig=fig)
            
            counter += 1
    

if __name__ == '__main__':
    
    # multivar_plotter(station=916)
    
    window_plotter(station=901, group_size=32, step_size=1)

