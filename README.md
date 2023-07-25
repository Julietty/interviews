# interviews
Testcases from interviews

## Million Agents

1. Данные: https://drive.google.com/drive/folders/1YAe6OLhGYXvR3D8QWc1y9IvNnloeu4zW?usp=sharing

Данные представляют собой изображения товаров. Данные сгрупированы в instance, который можно получить из названия файла. Размер instanse пример 1.5 изображения.

2. Jupyter notebook содержит полноценное решение следующей задачи: принимать пути до двух изображений и выдавать вероятность, с
которой два этих изображения можно считать идентичными.

3. В пайплайне есть
   - обработка датасета, сборка train, val, test
   - обучение одного из подходов metric learning для поиска похожих изображений. Код можно легко раширить до поиска k похожих изображений через knn
   - пример функции для получения вероятности идентичности картинок
  
4. Основные идеи:
   - metric learning
   - TripleMarginLoss как лосс обучения
   - основной бэкбон - Efficientnet_b3 без классификатора + линейный слой для эмбеддера
   - итоговая вероятность `(cosine_dist(x, y) + 1 )/ 2`
