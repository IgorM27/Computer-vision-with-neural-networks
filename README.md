# Computer-vision-with-neural-networks

## Описание проекта
Этот репозиторий содержит небольшой учебный проект в области компьютерного зрения, где я обучил нейронную сеть классифицировать изображения модных предметов с использованием классического набора данных Fashion MNIST.

## Набор данных
Используемый набор данных - Fashion MNIST, который состоит из 60 000 обучающих изображений и 10 000 тестовых изображений, относящихся к 10 различным категориям одежды. Каждое изображение имеет размер 28x28 пикселей.

## Архитектура модели
Архитектура нейронной сети для этого проекта представляет собой простую прямую нейронную сеть со следующими слоями:
- Входной слой (784 нейрона, соответствующих 28x28 пикселям)
- Скрытый слой (128 нейронов, с функцией активации ReLU)
- Выходной слой (10 нейронов, соответствующих 10 категориям одежды, с функцией активации softmax)

## Оценка
После обучения модель достигла точности на тестовом наборе данных около 85.6%, что указывает на её способность к обобщению на новых данных.

## Заключение
Этот проект служит образовательным упражнением по построению и обучению нейронных сетей для задач классификации изображений с использованием набора данных Fashion MNIST. Достигнутая точность 85.6% демонстрирует эффективность выбранной архитектуры модели и параметров обучения для данной задачи.
