1. Създаване на Docker образ
Изпълнете следната команда в терминала, намирайки се в директорията на проекта:

docker build -t my_ml_app .

Това ще създаде Docker образ с име my_ml_app.

2. Стартиране на контейнера
След като образът е успешно създаден, стартирайте контейнера с:

docker run -p 5000:5000 my_ml_app

3. Как да изпратите заявка към API-то
След стартиране на контейнера, можете да направите POST заявка към API-то:

//powershell//
Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method POST -Headers

Ако API-то работи правилно, очакваното връщане ще изглежда така:

{
  "prediction": [4.151938398277807]
}

Обработка на грешки
Ако входните данни са невалидни, API-то ще върне съответните грешки, например:

{
  "error": "Invalid input format. Expected a JSON object with numeric features."
}

