import warnings
warnings.filterwarnings(action="ignore")
import os
import pandas as pd
import redis
import cudf as cd
import yaml

def load_config(file_path):
    """
    ** Description: ** <em> It opens the file at the given path, reads the contents, and then parses the contents as a YAML file </em>
    
    Args:
        file_path (str): The path to the YAML file
    Returns:
        (dict): A dictionary
    """
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
    
config = load_config("fs_config.yml")

KEY_PREFIX = "recommendations_preprocessing"

class UpdateFeatureStore:
    def __init__(self, df):
        """
        ** Description: ** <em> The function takes in a dataframe and initializes the class with the dataframe, the asset and
        user feature lists, and the asset and user feature stores </em>
        
        Args:
            df (pd.DataFrame): the dataframe that contains the data
        """
        self.df = df
        self.asset_feature_list = config["asset_feature_list"]
        self.user_feature_list = config["user_feature_list"]
        self.asset_fs = redis.StrictRedis(host=os.environ["REDIS_IP"],
                                              port=os.environ["REDIS_PORT"],
                                              db=os.environ["ASSET_FS_DB"],
                                              decode_responses=True)
        self.user_fs = redis.StrictRedis(host=os.environ["REDIS_IP"],
                                             port=os.environ["REDIS_PORT"],
                                             db=os.environ["USER_FS_DB"],
                                             decode_responses=True)

    def get_key_name(self, feature_type, key):
        """
        ** Description: ** <em> It takes a feature type and a key, and returns a string that is the key name </em>
        
        Args:
            feature_type (str): The type of feature you're looking for. For example, if you're looking for
            key (str): The key to store the feature in
        
        Returns:
            (str): The key name for the feature type and key.
        """
        return "{}_{}:{}".format(KEY_PREFIX, feature_type, key)

    def update_asset_feature_store(self):
        """
        ** Description: ** <em> It takes the asset_feature_list from the dataframe, drops duplicates, sets the index to
        lomotif_id, transposes the dataframe, converts it to a pandas dataframe, converts it to a
        dictionary, and then updates the redis asset feature store with the data </em>
        """
        asset_df = self.df[self.asset_feature_list].drop_duplicates(subset = ["lomotif_id"], keep = "first").set_index('lomotif_id').T.to_pandas().to_dict('series')
        with self.asset_fs.pipeline() as pipe:
            for k, v in asset_df.items():
                values = dict(v)
                values.update({'lomotif_id': k})
                pipe.hmset(self.get_key_name('asset', k), values)
                if len(pipe) == 1000:
                    pipe.execute()
            pipe.execute()

    def update_user_feature_store(self):
        """
        ** Description: ** <em> We take the user_id column from the dataframe, drop duplicates, set the index to user_id,
        transpose the dataframe, convert it to a pandas dataframe, convert it to a dictionary, and then
        iterate through the dictionary to update the redis user feature store </em>
        """
        user_df = self.df[self.user_feature_list].drop_duplicates(subset = ["user_id"], keep = "first").set_index('user_id').T.to_pandas().to_dict('series')
        with self.user_fs.pipeline() as pipe:
            for k, v in user_df.items():
                pipe.hmset(self.get_key_name('user', k), dict(v))
                if len(pipe) == 1000:
                    pipe.execute()
            pipe.execute()
