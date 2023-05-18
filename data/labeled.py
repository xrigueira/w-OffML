from river import stream

from . import base

# Maybe I could leverage inheritance to make this code less wet

class labeled_901(base.FileDataset):
    """Water quality date from the 901 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""
    
    def __init__(self):
        super().__init__(
            filename="labeled_901_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_901": float,
                        "conductivity_901": float,
                        "dissolved_oxygen_901": float,
                        "pH_901": float,
                        "turbidity_901": float,
                        "water_temperature_901": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_902(base.FileDataset):
    """Water quality date from the 902 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""
    
    def __init__(self):
        super().__init__(
            filename="labeled_902_cle.csv",
            task=base.BINARY_CLF,
            n_features=16,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_902": float,
                        "conductivity_902": float,
                        "dissolved_oxygen_902": float,
                        "nitrates_902": float,
                        "pH_902": float,
                        "turbidity_902": float,
                        "water_temperature_902": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_904(base.FileDataset):
    """Water quality date from the 904 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""
    
    def __init__(self):
        super().__init__(
            filename="labeled_904_cle.csv",
            task=base.BINARY_CLF,
            n_features=16,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_904": float,
                        "conductivity_904": float,
                        "dissolved_oxygen_904": float,
                        "pH_904": float,
                        "turbidity_904": float,
                        "water_temperature_904": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_905(base.FileDataset):
    """Water quality date from the 905 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_905_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_905": float,
                        "conductivity_905": float,
                        "dissolved_oxygen_905": float,
                        "pH_905": float,
                        "turbidity_905": float,
                        "water_temperature_905": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_906(base.FileDataset):
    """Water quality date from the 906 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_906_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_906": float,
                        "conductivity_906": float,
                        "dissolved_oxygen_906": float,
                        "pH_906": float,
                        "turbidity_906": float,
                        "water_temperature_906": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_907(base.FileDataset):
    """Water quality date from the 907 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_907_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_907": float,
                        "conductivity_907": float,
                        "dissolved_oxygen_907": float,
                        "pH_907": float,
                        "turbidity_907": float,
                        "water_temperature_907": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_910(base.FileDataset):
    """Water quality date from the 910 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_910_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_910": float,
                        "conductivity_910": float,
                        "dissolved_oxygen_910": float,
                        "pH_910": float,
                        "turbidity_910": float,
                        "water_temperature_910": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )

class labeled_916(base.FileDataset):
    """Water quality date from the 916 station.
    
    The file contains water quality data from the Ebro river in Spain recorded every 15 minutes
    from Jan 1st, 1999 to Dec 31st, 2022. The goal is to implement online learning for anomaly 
    detection and prediction.
    
    References
    ----------
    [1] [SAICA EBRO](https://saica.chebro.es/)"""

    def __init__(self):
        super().__init__(
            filename="labeled_916_cle.csv",
            task=base.BINARY_CLF,
            n_features=15,
            n_samples=631106
            )
    
    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="label",
            converters={"ammonium_916": float,
                        "conductivity_916": float,
                        "dissolved_oxygen_916": float,
                        "pH_916": float,
                        "turbidity_916": float,
                        "water_level_916": float,
                        "water_temperature_916": float,
                        "label": int,
                
            },
            parse_dates={"date": "%Y-%m-%d %H:%M:%S"},
        )