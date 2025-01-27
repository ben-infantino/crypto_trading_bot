from src.data_processing.utility.indicator_registry import INDICATOR_FUNCTIONS

class IndicatorPipeline:
    """
    Applies a sequence of indicator computations to a given DataFrame.
    You can define which indicators to include and their parameter overrides.
    """
    def __init__(self, indicators_config, historical_data:bool=True):
        """
        indicators_config: list of dicts, each containing:
          {
            'name': 'ema',        # name as in the registry
            'override_params': {
              'window': 50        # optional overrides for the default params
              'close_col': 'my_close_col'
            }
          }
        """
        self.indicators_config = indicators_config
        self.historical_data = historical_data
    def run(self, df):
        """
        Apply each indicator in sequence, mutating the DataFrame.
        Returns the same DataFrame with new columns for each indicator.
        """
        for config in self.indicators_config:
            ind_name = config['name']
            # get the function + default params from the registry
            if ind_name not in INDICATOR_FUNCTIONS:
                raise ValueError(f"Indicator '{ind_name}' not found in registry.")
            
            func = INDICATOR_FUNCTIONS[ind_name]['func']
            params = INDICATOR_FUNCTIONS[ind_name]['params'].copy()
            params['historical_data'] = self.historical_data
            # override defaults if provided
            if 'override_params' in config:
                for k, v in config['override_params'].items():
                    params[k] = v

            # apply the function
            df = func(df, **params)

        return df