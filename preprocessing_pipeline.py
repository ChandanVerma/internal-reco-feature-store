import warnings
warnings.filterwarnings(action='ignore')
import cudf as cd
import cupy as cp
import time
import numpy as np
# from feature_store.update_feature_store import UpdateFeatureStore
import pandas as pd
import os
from update_feature_store import UpdateFeatureStore
from datetime import datetime
import time
import yaml
from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from snowflake.sqlalchemy import URL
from logzero import logger
import boto3
import traceback
from pathlib import Path

load_dotenv("./.env")

# QUERY = """
# select * from DS_INTERNAL_RECOMMENDATION.PUBLIC.FEATURED_FEED_VIEW
# """

def load_config(file_path):
    """
    ** Description: ** <em> It opens the file at the given path, reads the contents, and parses the contents as a YAML file </em>
    Args:
        file_path (str): The path to the YAML file
    Returns:
        (dict): A dictionary with all configuration values
    """
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# config = load_config("/home/ubuntu/model_based_recommendations/feature_store/fs_config.yml")
config = load_config("fs_config.yml")

def get_latest_file_name():
    """
    ** Description: ** <em> It returns the name of the latest file in the S3 bucket </em>
    Returns:
        (str): The latest file name in the s3 bucket
    """
    s3_client = boto3.client("s3")   
    response = s3_client.list_objects_v2(Bucket=config["s3_bucket_name"], Prefix='data_science/internal_recommendation/featured_feed_raw/')
    all = response['Contents']        
    latest = max(all, key=lambda x: x['LastModified'])
    return latest["Key"]

def save_preprocessed_data(df, save_file_name):
    """
    ** Description: ** <em> It takes in a dataframe and a file name, and saves the dataframe as a parquet file in the specified 
    S3 bucket. </em>
    
    Args:
        df (cudf.DataFrame): the dataframe to be saved
        save_file_name (str): The name of the file to save the preprocessed data to
    """
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    s3_url = f"s3://lomotif-datalake-prod/data_science/preprocessing_internal_recommendations/preprocessed_data_{save_file_name}.parquet.gzip"
    df.to_parquet(s3_url, index=False)

def convert_types(df):
    """
    ** Description: ** <em> It converts the columns in the dataframe to strings </em>
    
    Args:
        df (cudf.DataFrame): the dataframe to be converted

    Returns:
        df (cudf.DataFrame): the dataframe with the asset and user features converted to strings.
    """
    df[config["asset_feature_list"]] = df[config["asset_feature_list"]].astype(str)
    df[config["user_feature_list"]] = df[config["user_feature_list"]].astype(str)
    return df

class preprocessing:
    def __init__(self, df, config):
        """
        ** Description: ** <em> The function takes in a dataframe and a configuration dictionary and returns a dataframe with
        the columns specified in the configuration dictionary </em>
        
        Args:
            df (cudf.DataFrame): The dataframe that contains the data that we want to use to train our model
            config (dict): This is a dictionary that contains the configuration for the model
        """
        self.df = df
        self.config = config
    
    def generate_target(self):
        """
        ** Description: ** <em> The function generates a target variable called "target" which is 1 if the completion rate is
        greater than or equal to 0.8 and 0 otherwise </em>
        """
        logger.info("1. Generating target")
        try:
            self.df["completion_rate"] = self.df["WATCH_TIME"]/ self.df["VIDEO_DURATION"]
            # self.df["target"] = self.df["completion_rate"].map(lambda x: 1 if x >= 0.8 else 0)
            self.df["target"] = cp.where(self.df["completion_rate"] >= 0.8, 1, 0)
            logger.info("Target generated successfully")
        except:
            logger.warn(f"Could not generate target: {traceback.format_exc()}")
    
    def split_categories(self):
        """
        ** Description: ** <em> It takes the dataframe, splits the primary and secondary categories into two columns each, and
        drops the original columns </em>
        """
        logger.info("2. Splitting categories")
        try:
            self.df[["PC_1", "PC_2"]] = self.df['PRIMARY_CATEGORIES'].str.split(',', 1, expand=True)
            self.df[["SC_1", "SC_2"]] = self.df['SECONDARY_CATEGORIES'].str.split(',', 1, expand=True)       
            self.df.drop(labels = ["PRIMARY_CATEGORIES", "SECONDARY_CATEGORIES"], axis = 1, inplace = True)
            logger.info("Splitting categories done successfully")
        except:
            logger.error(f"Could not split categories: {traceback.format_exc()}")

    def split_datetime(self):
        """
        ** Description: ** <em> It takes the datetime column, splits it into hour, minute, weekday, day, and month, and then
        adds those columns to the dataframe </em>
        """
        logger.info("3. Splitting datetime")
        try:
            self.df['EVENT_TIME'] = cd.to_datetime(self.df['EVENT_TIME'])
            self.df['ts_hour'] = self.df['EVENT_TIME'].dt.hour
            self.df['ts_minute'] = self.df['EVENT_TIME'].dt.minute
            self.df['ts_weekday'] = self.df['EVENT_TIME'].dt.weekday
            self.df['ts_day'] = self.df['EVENT_TIME'].dt.day
            self.df['ts_month'] = self.df['EVENT_TIME'].dt.month
            logger.info("Datetime generated successfully")
        except:
            logger.info(f"Could not split datetime: {traceback.format_exc()}")

    def treat_missing_data(self):
        """
        ** Description: ** <em> It takes a dataframe as input and returns a dataframe with missing values replaced by "Unknown" </em>
        """
        logger.info("4. Treat missing data")
        try:
            vars_with_na = [var for var in self.df.columns if self.df[var].isnull().sum() > 0]
            logger.info(f"Following attributes consists of missing values: {vars_with_na}")
            for col in vars_with_na:
                self.df[col].fillna("Unknown", inplace = True)
            logger.info("Done !!!")
        except:
            logger.info(f"Error occured while treating missing values: {traceback.format_exc()}")

        # self.df = self.df.to_pandas()
        # return self.df
    
    def hash_encode(self):
        """
        ** Description: ** <em> The function takes the dataframe and hashes the values of the columns PC_1, ASSET_ID and USER_ID
        using the murmur3 hashing algorithm </em>
        """
        logger.info("5. Hash encoding attributes")
        try:
            self.df["PC_1_H"] = self.df["PC_1"].hash_values(method="murmur3")
            self.df["ASSET_ID_H"] = self.df["ASSET_ID"].hash_values(method="murmur3")  
            self.df["USER_ID_H"] = self.df["USER_ID"].hash_values(method="murmur3")
            logger.info("Hash encoding done")
        except:
            logger.info(f"Could not hash encode attributes: {traceback.format_exc()}")

    def drop_columns(self):
        """
        ** Description: ** <em> It drops the columns that are not required for the analysis </em>
        """
        logger.info("6. Dropping unnecessary raw attributes")
        try:
            self.req_cols = self.config["req_columns"]
            self.df = self.df[self.req_cols]
            logger.info("Removed unnecessary raw features")
            # return self.df
        except:
            logger.warn(f"Could not remove unnecessary raw attributes you might face memory issues: {traceback.format_exc()}")

    def target_encode(self, train, col, target, smooth=20):
        """
        ** Description: ** <em> For each column in the list of columns, group by that column and calculate the mean of the
        target and the count of the column. Then, calculate the target encoding for each group </em>
        
        Args:
            train (cudf.Dataframe): the training dataframe
            col (str): the column to encode
            target (str): The target column
            smooth (int): This is the smoothing parameter. It's the number of observations we want to add to the numerator and denominator of the target encoding calculation, defaults to 20 (optional)
        Returns:
            (cudf.DataFrame): the train dataframe with the new column added.
        """
        col_name = '_'.join(col)   
        df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        mn = train[target].mean()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_' + col_name] = ((df_tmp['mean']*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
        df_tmp = df_tmp.drop('mean', axis=1)
        df_tmp = df_tmp.drop('count', axis=1)
        train = train.merge(df_tmp, on=col, how='left')
        return train

    # def count_encode(self, df, col, gpu=True):
    #     """
    #     ** Description: ** <em> It takes a dataframe, a column name, and a boolean value for whether or not to use the GPU. It
    #     then creates a new column in the dataframe called 'org_sorting' which is just a list of integers
    #     from 0 to the length of the dataframe. It then creates a new dataframe called df_tmp which is the
    #     value counts of the column you passed in. It then merges the two dataframes together on the
    #     column you passed in and the 'org_sorting' column. It then sorts the dataframe by the
    #     'org_sorting' column and then drops the 'org_sorting' column. </em>
        
    #     Args:
    #         df (cudf.DataFrame): the dataframe
    #         col (str): the column to be encoded
    #         gpu (boolean): whether to use GPU or not, defaults to True (optional)
    #     """
    #     if gpu:
    #         df['org_sorting'] = cp.arange(len(df), dtype="int32")
    #     else:
    #         df['org_sorting'] = np.arange(len(df), dtype="int32")
        
    #     df_tmp = df[col].value_counts().reset_index()
    #     df_tmp.columns = [col,  'CE_' + col]
    #     df_tmp = df[[col, 'org_sorting']].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    #     df['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
    #     df = df.drop('org_sorting', axis=1)
    #     return df

    def count_encode(self, df, col, gpu=True):
        """
        ** Description: ** <em> It takes a dataframe, a column name, and a boolean value for whether or not to use the GPU. It
        then creates a new column in the dataframe called 'org_sorting' which is just a list of integers
        from 0 to the length of the dataframe. It then creates a new dataframe called df_tmp which is the
        value counts of the column you passed in. It then merges the two dataframes together on the
        column you passed in and the 'org_sorting' column. It then sorts the dataframe by the
        'org_sorting' column and then drops the 'org_sorting' column. </em>
        
        Args:
            df (cudf.DataFrame): the dataframe
            col (str): the column to be encoded
            gpu (boolean): whether to use GPU or not, defaults to True (optional)
        """   
        df[col] = df[col].astype(str)  
        df_tmp = df[col].value_counts().reset_index()
        df_tmp.columns = [col,  'CE_' + col]
        df = df.merge(df_tmp, how = "left", on = col)
        df[col].fillna(0, inplace = True)
        return df

    def target_encode_attributes(self):
        """
        ** Description: ** <em> It takes the columns specified in the config file and performs target encoding on them </em>
        """
        logger.info("7. Performing target encoding")
        try:
            tar_encode_columns = self.config["target_encoding_columns"]
            for col in tar_encode_columns:
                self.df = self.target_encode(self.df, col=[col], target="target")
            logger.info("Target encoding done successfully")
            # return self.df
        except:
            logger.error(f"Error while target encoding: {traceback.format_exc()}")

    def count_encode_attributes(self):
        """
        ** Description: ** <em> It takes the columns specified in the config file and performs count encoding on them </em>
        """
        logger.info("8. Performing count encoding")
        try:
            count_attribute_columns = self.config["count_encoding_columns"]
            for col in count_attribute_columns:
                print(col)
                self.df = self.count_encode(self.df, col=col, gpu=True)
            logger.info("Count encoding done successfully")     
        except:
            logger.error(f"Error while count encoding: {traceback.format_exc()}")

    def pipeline(self):
        """
        ** Description: ** <em> The function takes in a dataframe and returns a dataframe with only the features specified in
        the config file </em>
        """
        self.df = self.df[self.config["model_features"]]

    def rename_cols(self):
        """
        ** Description: ** <em> It renames the columns of the dataframe using the column names specified in the config file </em>
        Returns:
            (cudf.DataFrame): The dataframe with the renamed columns.
        """
        logger.info("9. Renaming columns")
        try:
            self.df.rename(columns= self.config["col_rename"], inplace = True)
            logger.info("Renaming column done successfully")
            return self.df
        except:
            logger.info(f"Error while renaming columns: {traceback.format_exc()}")

if __name__ == "__main__":
    # logger.info("Connecting to SF")
    # url = URL(
    #     user=os.environ["SNOWFLAKE_USER"],
    #     password=os.environ["SNOWFLAKE_PASSWORD"],
    #     account=os.environ["SNOWFLAKE_ACCOUNT"],
    #     warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    #     database=os.environ["SNOWFLAKE_DATABASE"],
    #     schema=os.environ["SNOWFLAKE_SCHEMA"]
    # )
    # engine = create_engine(url)
    # conn = engine.connect()
    # logger.info("Connected to SF")
    logger.info("Downloading events data")
    # df = pd.read_sql(QUERY, conn)
    file_name = get_latest_file_name()
    processing_file_name = str(file_name.split("/")[-1])
    save_file_name = processing_file_name.split("raw_ranking_data_")[-1].split(".")[0]
    logger.info(f"Processing file: {processing_file_name}")
    df = cd.read_parquet(f"s3://{config['s3_bucket_name']}/{file_name}")
    df.drop_duplicates(keep = 'first', ignore_index = True, inplace = True)
    df.columns = df.columns.str.upper()
    # df = cd.from_pandas(df)
    logger.info("Data Downloaded")  
    #df = cd.read_parquet("/home/ubuntu/model_based_recommendations/ranking/data/model_data_June_Sep.parquet")
    logger.info(f"*****  PREPROCESSING SUCCESSFULLY STARTED AT: {datetime.now()}  *****")
    start = time.time()
    pre = preprocessing(df=df, config = config)
    pre.generate_target()
    pre.split_categories()
    pre.split_datetime()  
    pre.treat_missing_data()
    pre.hash_encode()
    pre.drop_columns()
    pre.target_encode_attributes()
    pre.count_encode_attributes()
    pre.pipeline()
    df = pre.rename_cols()
    logger.info("10. Saving preprocessed data to S3")
    save_preprocessed_data(df=df, save_file_name=save_file_name)
    logger.info("Saved preprocessed data to S3")
    end = time.time()
    process_time = end - start
    logger.info(f"Total time to process: {process_time}")
    logger.info(f"*****  PREPROCESSING SUCCESSFULLY COMPLETED AT: {datetime.now()}  *****")
    #### UNCOMMENT IF DATA TO BE STORED IN REDIS
    redis_df = convert_types(df)
    update_fs = UpdateFeatureStore(redis_df)
    logger.info("Updating asset feature store")
    update_fs.update_asset_feature_store()
    logger.info("Asset feature store updated")
    logger.info("Updating user feature store")
    update_fs.update_user_feature_store()
    logger.info("User feature store updated")