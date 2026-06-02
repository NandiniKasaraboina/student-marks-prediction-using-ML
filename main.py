import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_absolute_error,r2_score




# data = {
#     "Hours": [1,2,3,4,5],
#     "Marks": [30,40,50,60,70]
# }


df = pd.read_csv("student_marks.csv")






# plt.scatter(df["Hours_Studied"],df["Marks"])

# plt.xlabel("Hours Studied")

# plt.ylabel("Marks")

# plt.title("Hours Studied vs Marks")


# plt.show()

# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.info())


# print(df.head())
# print(df.shape)
# print(df.columns)


x = df[["Hours_Studied"]]
y = df["Marks"]


# print(type(x))
# print(type(y))


x_train,x_test,Y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(x_train,Y_train)


# print(model.coef_)
# print(model.intercept_)


predicted_marks= model.predict([[5]])
# print(predicted_marks)


# print(x_test)
y_pred = model.predict(x_test)
# print(y_pred)


# mae = mean_absolute_error(y_test,y_pred)
# print("mae:",mae)


# r2 = r2_score(y_test,y_pred)
# print("r2 score:",r2)


comparison = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})
# print(comparison.head(10))


# print(x_train.shape)
# print(x_test.shape)
# print(Y_train.shape)
# print(y_test.shape)


hours = float(input("Enter Study Hours: "))

# prediction = model.predict([[hours]])

new_data = pd.DataFrame({
    "Hours_Studied": [hours]
})

prediction = model.predict(new_data)

print("Predicted Marks:", prediction[0])