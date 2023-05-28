import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Patch scikit-learn
from sklearnex import patch_sklearn
patch_sklearn()

class InputWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Breast Cancer Prediction')
        self.setGeometry(200, 200, 300, 200)

        label_radius = QLabel('Mean Radius:', self)
        label_radius.move(20, 20)
        self.input_radius = QLineEdit(self)
        self.input_radius.move(120, 20)

        label_texture = QLabel('Mean Texture:', self)
        label_texture.move(20, 50)
        self.input_texture = QLineEdit(self)
        self.input_texture.move(120, 50)

        label_perimeter = QLabel('Mean Perimeter:', self)
        label_perimeter.move(20, 80)
        self.input_perimeter = QLineEdit(self)
        self.input_perimeter.move(120, 80)

        label_area = QLabel('Mean Area:', self)
        label_area.move(20, 110)
        self.input_area = QLineEdit(self)
        self.input_area.move(120, 110)

        label_smoothness = QLabel('Mean Smoothness:', self)
        label_smoothness.move(20, 140)
        self.input_smoothness = QLineEdit(self)
        self.input_smoothness.move(120, 140)

        button_predict = QPushButton('Predict', self)
        button_predict.move(120, 170)
        button_predict.clicked.connect(self.predict)

        self.show()

    def predict(self):
        # Get the input values
        radius = float(self.input_radius.text())
        texture = float(self.input_texture.text())
        perimeter = float(self.input_perimeter.text())
        area = float(self.input_area.text())
        smoothness = float(self.input_smoothness.text())

        # Perform the prediction
        prediction = make_prediction(radius, texture, perimeter, area, smoothness)

        # Display the prediction result
        if prediction == 1:
            result_text = "The prediction indicates the presence of breast cancer."
        else:
            result_text = "The prediction indicates the absence of breast cancer."

        QMessageBox.information(self, 'Prediction Result', result_text, QMessageBox.Ok)

def make_prediction(radius, texture, perimeter, area, smoothness):
    dataset = pd.read_csv('Breast_cancer_data.csv')

    X = dataset[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
    y = dataset['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    user_input = pd.DataFrame(columns=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'])
    user_input.loc[0] = [radius, texture, perimeter, area, smoothness]

    prediction = model.predict(user_input)

    return prediction[0]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InputWindow()
    sys.exit(app.exec_())
