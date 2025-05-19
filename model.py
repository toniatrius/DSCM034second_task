import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # Зареждам набор от данни за цената на жилища в Калифорния
    data = fetch_california_housing()
    X, y = data.data, data.target  # X ще съдържа 8 характеристики, y ще бъде медианата на цените

    # Разделям данните на 80% за обучение и 20% за тестване, следвайки стандартната практика
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Моделът е обучен и запазен с името model.pkl")

if __name__ == "__main__":
    train_and_save_model()
