# DS
#-------------------------------------------Посторайтесь открыть All.ipynb-------------------------------------------------


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ----------------------------------------------------Не Работало----------------------------------------------------------
# Подключение билиотек для визуализации решающего дерева
from IPython.display import SVG
from IPython.display import display
from IPython.display import HTML
HTML("<style>svg{width: 70% !important; height: 70% !important;} </style>")

from graphviz import Source
#--------------------------------------------------------------------------------------------------------------------------

# Изменение размера графика
sns.set(rc={'figure.figsize':(9, 6)})

# Позволяет получить время выполнения метода
%timeit method



#--------------------------------------------------Полезные методы---------------------------------------------------------

# Функция, разбивающая входные данные на тренировочные и тестовые
from sklearn.model_selection import train_test_split

# Использование кроссвалидации
from sklearn.model_selection import cross_val_score

# Метод для эффективного обучения моделей
from sklearn.model_selection import GridSearchCV

# Метод для поиска лучшей модели, путём случайной выборки указанных параметров
from sklearn.model_selection import RandomizedSearchCV

# Random forest
from sklearn.ensemble import RandomForestClassifier

# Метод создания матрицы конфузов...
from sklearn.metrics import confusion_matrix

# Подключение Precision
from sklearn.metrics import precision_score

# Подключение Recall
from sklearn.metrics import recall_score



#-----------------------------------------------------Функции--------------------------------------------------------------

# Таким образом можно определить, какие особенности (св-ва/поля) важнее для конечного решения
def feat_importance(clf, X):
    # Получение значимости свойств (особенностей) (feature - особенность/св-во, importance - важность/значимость)
    f_imp = clf.feature_importances_

    # list(X) - получает названия колонок; ascending: True - по возрастанию/ False - по убыванию (ascending - восходящий)
    
#     return pd.DataFrame({'features': X.columns,
#                          'importance': f_imp}).sort_values('importance', ascending=False)
    return pd.DataFrame(f_imp, index= X.columns, columns=['importance']).sort_values('importance', ascending=False)

def metrics(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2*precision*recall / (precision + recall)

    return 'precision: ' + str(precision.round(2)) + ', recall: ' + str(recall.round(2)) + ', F1: ' + str(f1.round(2))


#------------------------------------------------------Ручные--------------------------------------------------------------

# Точность (Не прихватить лишнее)
def Precision(Tp, Fp):
    return Tp / (Tp + Fp)

# Полнота (Не пропустить нужное)
def Recall(Tp, Fn):
    return Tp / (Tp + Fn)

# Среднее гармоническое точности и полноты
def F1_Score(Tp, Fp, Fn):
    precis = Precision(Tp, Fp)
    recall = Recall(Tp, Fn)
    return 2 * precis * recall / (precis + recall)

#--------------------------------------------------------------------------------------------------------------------------

# Отрисовывает дерево
# names - отображение имени
def render(clf, X, names):
    plt.figure(figsize=(100, 25))
    return tree.plot_tree(clf, feature_names=list(X), class_names=names, filled=True);

# Выводит график результатов тестирования с разной максимальной глубиной дерева
def depth_test(X_train, y_train, X_test, y_test, max_depth, as_pandas=False):
    # Заводим DataFrame для группировки результатов
    scores_data = pd.DataFrame()

    # Проводим тесты для глубин от 1 до n
    for depth in range(1, max_depth):
        # Создание и обучение дерева
        work_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        work_clf.fit(X_train, y_train)

        # Тестирование дерева
        train_score = work_clf.score(X_train, y_train)
        test_score = work_clf.score(X_test, y_test)

        # Добавление в DataFrame
        temp_score_data = pd.DataFrame({'depth': [depth],
                                        'train_score': [train_score],
                                        'test_score': [test_score]})
        scores_data = scores_data.append(temp_score_data)

    # Объединение колонок в одну (создастся 2 новые колонки: название исходных и их значения)
    # id_vars - переменные, которые необходимо сохранить; value_vars - переменные, которые необходимо объеденить
    # var_name - установка ключа для переменной (value_vars переходят в эту колонку как значения)
    # value_name - устанавливает название колонки для значений (это значение раньше было в 'train_score')
    scores_data_long = pd.melt(scores_data, id_vars=['depth'], value_vars=['train_score', 'test_score'],
                               var_name='type', value_name='score')
    if(as_pandas):
        return score_data_long
    else:
        return sns.lineplot(x='depth', y='score', hue='type', data=scores_data_long)
        
        

#--------------------------------------------------Базовые действия--------------------------------------------------------

df = pd.read_csv('path') # Создание pandas Dataframe из csv файла
# Другой вариант создания
data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})

df.head() # Выводит первые n элементов
df.tail() # Вывод последних n элементов
df.describe() # Получение описательной статистики
df.dtypes # Узнать какие типы данных в Dataframe (int64 = количественная статистика, object ~ строка)

df.shape # Возвращает кол-во строк и столбцов в них
df.size # Возвращает произведение кол-ва строк на столбцы

df.iloc[0:5, 0:3] #(integer location) Позволоят отобрать данные по строкам (1 ввод) и столбцам (2 ввод)
df.index = ["Belmon", "Gloria", "Alex", "Marty"] # Присвоение index'ов DataFram'у
df.loc[['Belmon', 'Alex'], ['gender','math score']] # Отбор данных по названию строк и столбцов

# Добавление новой колонки в DataFrame
students['total score'] = students['math score'] + students['reading score'] + students['writing score']

# Другой вариант добавления новой колонки
students = students.assign(total_score_log = np.log(students['total score']))

df.drop(["species"], axis=1) # 'Сбросить', удалить столбец

# Проверяем кол-во неизвестных данных
# isnull - возвращает true если значение пропущено
df.isnull().sum()


df.loc[students.gender == 'female'] # Выводит всех участников, женского пола
math_mean = df['math score'].mean() # Вывод среднего значения
df.loc[df['math score'] > math_mean] # Выводит всех студентов, у которых math score больше среднего

# Выборка из нескольких условий; & применяется для Series объектов
students[(students['writing score'] > 90) & (students.gender == 'female')].head()
test.query('lunch == "standard"').var() # Получение дисперсного значения для студентов со стандартным обедом

# Переименовать колонки
students = students.rename(columns = 
                           {'parental level of education' : 'parental_level_of_education',
                            'test preparation course' : 'test_preparation_course',
                            'math score' : 'math_score',
                            'reading score' : 'reading_score',
                            'writing score' : 'writing_score'})

students.filter(like='score') # Получить колонки, содержащие слово 'score'; like = "содержит в себе"
# axis: 0 - строка, 1 - колонка



#-----------------------------------------------------Группировка----------------------------------------------------------

# Группировка по полу и вывод среднего значения
df.groupby('gender').aggregate({'math score' : 'mean', 'writing score' : 'mean'})

# as_index=False в groupby, позволяет изменить индексацию, т.е. убрать female и male из индексов, заменив их на 0,...,n
students.groupby(['gender', 'race/ethnicity'], as_index=False).aggregate({'math score' : 'mean', 'writing score' : 'mean'})

# sort_valuers позволяет отсортировать по колонкам; ascending позволяет изменить порядок сортировки
students.sort_values(['gender', 'math score'], ascending=False)\
    .groupby('gender').head()
# Сортируем по убыванию баллов и полу, группируем по полу, и ыводим первых 5 студентов (топ 5 юношей и девушек по math) 



#-----------------------------------------------Подготовка данных----------------------------------------------------------

# X - features (особенности), Y - results
X = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
Y = train_data.Survived

# Разбиваем данные из одной колонки по разным, удаляем лишнюю и переименовываем (0 - female, 1 - male)
X = pd.get_dummies(X).drop('Sex_female', axis=1).rename({'Sex_male': 'Sex'}, axis=1)

# Заполнение пропущенных значений возраста медианными
X.loc[X.Sex == 0, 'Age'] = X[X.Sex == 0].fillna({'Age': X[X.Sex == 0].Age.median()})
X.loc[X.Sex == 1, 'Age'] = X[X.Sex == 1].fillna({'Age': X[X.Sex == 1].Age.median()})

# Разделение данных на тестовые и проверочные
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=42)



#-------------------------------------------------------Графики------------------------------------------------------------

# Вывод гракика
students['math score'].hist()

# График корреляции между math и reading score
students.plot.scatter(x='math score', y='reading score')

# hue - группирующая переменная; fit_reg - переключает отображение регрессионных линий
ax = sns.lmplot(x='math score', y='reading score', hue='gender', data=students, fit_reg=False)
ax.set_xlabels('Math score')    # Изменение подписи x
ax.set_ylabels('Reading score') # Изменение подписи y

# Какие-то графики
sns.lmplot(x='x', y='y', data = df)
sns.scatterplot(df.iloc[:, 0], df.iloc[:, 1])

# Перебор всех столбцов, col = названию столбца
# Вывод графика
for col in df:
    sns.distplot(df[col], kde_kws={'label':col})
    
# orient='v' рисует вертикально, а не гризонтально
sns.violinplot(df['petal length'], orient='v')

# Набор странных графиков, по ним можно что-то понять...
sns.pairplot(df, hue='species')

# Тепловая карта
sns.heatmap(space.corr(), annot=True, cmap=plt.cm.Blues)

# ----------------------------------------------------Не Работало----------------------------------------------------------
# Создание графа дерева решений
graph = Source(tree.export_graphviz(clf, out_file=None,
                                    feature_names=list(X),
                                    class_names=['Negative', 'Positive'],
                                    filled=True))
# Вывод графа на экран
display(SVG(graph.pipe(format='svg')))



#--------------------------------------------------Decision Tree-----------------------------------------------------------

# criterion - критерий обучения дерева решений
# max_depth - максимальная глубина дерева
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Простое обучение дерева на тестовых данных
clf.fit(X_train,y_train)

# Обучение дерева с использованием кросс валидации
# cv - указывает на сколько частей разбивать тренировочные данные
# Сначала разбивает тренировочные данные на cv частей, затем обучается на всех, кроме 5, и предсказал 5, на всех, кроме 4..
cross_val_score(clf, X_train, y_train, cv=5).mean()

#---------------------------------------------Другой метод обучения--------------------------------------------------------

# Создаём пустое дерево
clf = tree.DecisionTreeClassifier()

# Указываем параметры, которые будут меняться в GridSearchCV
parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}

# cv - сколько подходов
grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)

# Обучаем модель на каждых значениях (перебор)
grid_search_cv_clf.fit(X_train, y_train)

# Получение дерева с лучшими параметрами
best_clf = grid_search_cv_clf.best_estimator_

# Сохраняем результат предсказаний нашего дерева (модели)
y_pred = best_clf.predict(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Получение вероятности предсказания
y_pred_prob = best_clf.predict_proba(X_test)

# Берём только 2 колонку (положительный исход) и выводим гистограмму
pd.Series(y_pred_prob[:, 1]).hist()
# Другой способ визуализации
tree.plot_tree(clf, feature_names=list(X), class_names=['Negative', 'Positive'], filled=True);



#--------------------------------------------------Random forest-----------------------------------------------------------

# Создание леса решений
clf_rf = RandomForestClassifier()

# Указываем параметры для леса решений
# n_estimators - кол-во деревьев в лесу
params = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}

search_clf_rf = GridSearchCV(clf_rf, params, cv=5)

# Обучаем лес решений
search_clf_rf.fit(X_train, y_train)

# Берём лучший лес
best_clf_rf = search_clf_rf.best_estimator_

# Проверяем полученный лес решений
best_clf_rf.score(X_test, y_test)


# Предсказание
y_pred = best_clf_rf.predict(X_test)

# Вывод результатов предсказания
metrics(y_test, y_pred)

# Выводим график важности свойств
imp = pd.DataFrame(rf.feature_importances_, index= X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))



#----------------------------------------------------Что-то ещё------------------------------------------------------------

# Перевод времени с секунд в понятную дату
events['date'] = pd.to_datetime(events.timestamp, unit='s')
submissions['date'] = pd.to_datetime(submissions.timestamp, unit='s')

# Добавление поля с отображением только даты ГГ-ММ-ДД
events['day'] = events.date.dt.date
submissions['day'] = submissions.date.dt.date
