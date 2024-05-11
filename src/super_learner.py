import numpy as np
import time
import joblib
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def main():
    start_time = time.time()
    
    # Load the data from 1d_power_spectrum_dataset.npz
    print('Loading data...')
    data = np.load('1d_power_spectrum_dataset.npz')
    pow_spect = data['pow_spect']
    flare_label = data['flare_label']

    # Split the data into training and testing set
    print('Splitting data...')
    x_train, x_test, y_train, y_test = train_test_split(pow_spect, flare_label, test_size=0.2, random_state=42)

    # Define base classifiers
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('sgd', SGDClassifier(max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('linear_svc', LinearSVC(max_iter=10000, dual=True)), 
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('nb', GaussianNB())
    ]

    # Define final esitmator
    final_estimator = LogisticRegression(max_iter=1000, n_jobs=max(multiprocessing.cpu_count() - 1, 1))

    # Initialize stacking classifier with multiple meta-learners
    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        verbose=2,
        n_jobs=-1
    )

    # Train the stacking classifier
    print('Training the stacking classifier...')
    stacking_classifier.fit(x_train, y_train)

    # Save the trained model to a file
    print('Saving stacking classifier...')
    joblib.dump(stacking_classifier, 'stacking_classifier_model.pkl')

    # Evaluate the stacking classifier
    print('Evaluating the stacking classifier...')
    y_pred = stacking_classifier.predict(x_test)
    report = classification_report(y_test, y_pred, target_names=["Non-Flare (0)", "Flare (1)"])

    print("Classification Report:")
    print(report)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", time.strftime("%H:%M:%S", time.gmtime(execution_time)))

if __name__ == "__main__":
    main()
