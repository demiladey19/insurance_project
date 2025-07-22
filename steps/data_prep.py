# Import the needed libraries
 
from zenml import step
import pandas as pd
import numpy as np
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Optional,Tuple,Dict
import joblib 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# configure our logging
logger = get_logger(__name__)

# define the cleaning steps


@step 
def encode_step(data: pd.DataFrame) -> Tuple[ 
    Annotated[Optional[pd.DataFrame], "Encoded Data"],
    Annotated[Optional[Dict], "Encoder Object"]]:
    """
    This step encodes categorical variables in the dataset using label encoding.
    """
    label_encoders = None
    try: 
        label_encoders = {}
        cat_cols = list(data.select_dtypes(include="object").columns)
        for column in cat_cols:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            label_encoders[column] = encoder
            logger.info(f"Encoded data successfully. data types are:\n {data.info()}")
    except Exception as err:
        logger.error(f"An error occurred. Detail: {err}")


    return data, label_encoders        

]
@step
def split_dataset(data: pd.DataFrame) -> Tuple[ 
    Annotated[Optiona[lpd.DataFrame], "X_train"],
    Annotated[Optiona[pd.DataFrame], "X_test"],
    Annotated[Optiona[pd.Series], "y_train"],
    Annotated[Optiona[pd.Series], "y_test"]]:
    X_train, X_test, y_train, y_test = None, None, None, None
    try:
        X = data.drop(columns=['charges'])
        y = data['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=23)
        logger.info(f"Splitting completed with shape X_train: {X_train.shape}")
    except Exception as err:
        logger.error(f"An error occured. Detail: {err}")
    return X_train, X_test, y_train, y_test

@step
def scale_datastep(X_train: pd.DataFrame,
                   X_test:pd.DataFrame) -> Tuple[ 
                       Annotated[Optional[pd.DataFrame], "Scaled X_train"],
                       Annotated[Optional[pd.DataFrame], "Scaled X_test"]]:
    try:
        scaler = StandardScaler()
        scaler.fit(X_train)
        column_names = list(X_train.columns)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = pd.DataFrame(X_train_scaled, columns=columns_names)
        X_test_final = pd.DataFrame(X_test_scaled, columns=columns_names)

        logger.info("Scaling completed successfully!")
    except Exception as err:
        X_train_final, X_test_final = None, None
        logger.error(f"An error occured. Detail: {err}")

        return X_train_final, X_test_final   