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
from sklearn,ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# configure our logging
logger = get_logger(__name__)

@step
def train_model(X_train: pd.DataFrame,
                y_train:pd.Series),
                X_test:pd.DataFrame,
                y_test:pd.Series) -> Annotated[Optional[RandomForestRegressor],
                                                "Model Object"]:
    model = None 
    try:
        model = RandomForestReggressor(random_state=23)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # compute the scores
        train_rmse = root_mean_squared_error(y_train, train_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)
                                            
        logger.info(f""" 
                    Completed training the base model with metrics:
                    train rmse: {train_rmse}
                    test rmse: {test_rmse}
                    """)

    except Exception as err:
        logger.error(f"An error occured. Detail: {err}")

        return model
