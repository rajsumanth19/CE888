from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

def evaluate(model,x_train,x_test,y_train,y_test):
    print("Train Accuracy :", accuracy_score(y_train, model.predict(x_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, model.predict(x_train)))
    print("Train classification report")
    print(classification_report(y_train, model.predict(x_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, model.predict(x_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(x_test)))
    print("Test classification report")
    print(classification_report(y_test, model.predict(x_test)))