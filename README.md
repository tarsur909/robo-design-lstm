# robo-design-lstm
## Torque Prediction

Running TorquePred.py allows user to predict dataset of actuator torque using a history of position and error in velocity. Jupyter notebooks can be used to reference model training procedure. 

## Example Usage

After downloading or cloning repo:
```python
from TorquePred import TorquePred

#load class
model = TorquePred('/Users/tarsur909/Documents/PythonStuff/lstm_scripted.pt')

#return a dataframe with torque predictions and actual torque values as columns
predictions_df = model.predict('/Users/tarsur909/Documents/PythonStuff/data/data1.0.txt')

#graph predictions of dataset predicted
model.graph_predictions()
```

## TorquePred()
| Parameter Name        | Type           | Description  |
| ------------- |:-------------:| -----:|
| model_path | str | Path to PyTorch model on user device. |
| history | int, default = 8 | Number of lookbacks {t, t-1, t-2, ..., t-n} that the LSTM uses. Don't change value without altering the history in the model code as well. |

## .predict()
| Parameter Name        | Type           | Description  |
| ------------- |:-------------:| -----:|
| dataset  | str | Path to .txt file on user device containing dataset to predict torque. There should be 7 columns, and the olumns should be des_dx, x, theta (pitch), dx, dth, force, and torque, respectively. Fill in a column of zeros for unkown columns. |
