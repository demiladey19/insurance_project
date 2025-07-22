import pandas as pd
from zenml import pipeline
from steps.data_loader import load_data
from steps.data_prep import encode_step, split_dataset, scale_datastep
from zenml.logger import get_logger
from steps.model_training import train_model

logger = get_logger(__name__)

#define a pipline function

@pipeline
def insurance_pipline():
    data = load_data()
    data, label_encoders = encode_step(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train, X_test = scale_dataset(X_train, X_test)
    model = train_model(X_train, y_train, X_test, y_test)

    def insurance_pipline():
          

          return model
    
    if __name__ == "__main__":
         insurance_pipline()