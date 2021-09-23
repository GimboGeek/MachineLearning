



from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = load_iris()

x=data.data
y=data.target

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
print(y_test)
print()
print(y_pred)
print()
print(accuracy_score(y_test, y_pred))