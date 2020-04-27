import argparse
import pandas as pd
import os
import pickle as pkl 
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import sklearn
#from six import BytesIO


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--C', type=float, default= 1)
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--gamma', type= str, default='scale')
    parser.add_argument('--probability', type= bool, default= True)

    

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, unknown = parser.parse_known_args()

    # Take the set of files and read them all into a single pandas dataframe
    with open(os.path.join(args.train, "data.pickle"), 'rb') as handle:
        data = pkl.load(handle)

    # labels are in the first column
    train_y = data['label']
    train_X = data['data']

    # Now use scikit-learn's NN to train the model.
    model = model = SVC(C = args.C,
                        kernel= args.kernel,
                        gamma = args.gamma,
                        probability= args.probability
                       )


    
    model = model.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

# def _npy_loads(data):
#     """
#     Deserializes npy-formatted bytes into a numpy array
#     """
#     stream = BytesIO(data)
#     return np.load(stream)

# def input_fn(input_bytes, content_type):
#     """This function is called on the byte stream sent by the client, and is used to deserialize the
#     bytes into a Python object suitable for inference by predict_fn -- in this case, a NumPy array.
    
#     This implementation is effectively identical to the default implementation used in the Chainer
#     container, for NPY formatted data. This function is included in this script to demonstrate
#     how one might implement `input_fn`.
#     Args:
#         input_bytes (numpy array): a numpy array containing the data serialized by the Chainer predictor
#         content_type: the MIME type of the data in input_bytes
#     Returns:
#         a NumPy array represented by input_bytes.
#     """
#     if content_type == 'application/x-npy':
#         return _npy_loads(input_bytes)
#     else:
#         raise ValueError('Content type must be application/x-npy')


# def predict_fn(input_data, model):
#     prediction = model.predict(input_data.reshape(1,-1))
#     pred_prob = model.predict_proba(input_data.reshape(1,-1))
#     return np.array([prediction, pred_prob])

