# import pandas as pd
# import numpy as np
# from sklearn import linear_model
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
#
# df = pd.read_csv('/content/homeprices.csv')
# print(df)
#
#
# # %matplotlib inline
# plt.xlabel('area')
# plt.ylabel('price')
# plt.scatter(df.area,df.price,color='red',marker='+')
#
#
# new_df = df.drop('price',axis='columns')
# print(new_df)
#
#
#
# price = df.drop('area',axis='columns')
# print(price)
#
#
# # Create linear regression object
# reg = linear_model.LinearRegression()
# reg.fit(new_df,price)
#
#
#
#
# #Predict price of a home with area = 3300 sqr ft
# reg.predict([[3300]])
#
#
# print(reg.coef_)
#
#
# print(reg.intercept_)
#
# plt.xlabel('area',fontsize=20)
# plt.ylabel('price',fontsize=20)
# plt.scatter(df.area,df.price,color='red',marker='+')
# plt.plot(df.area,reg.predict(df[['area']]),color='blue')
#
#
# mean_squared_error(df['price'],reg.predict(df[['area']]))
#
# print(df.price)
#
# df1 = pd.read_csv('/content/canada_per_capita_income.csv')
# print(df1)
#
# df1 = df1.rename({"per capita income (US$)":"capita"}, axis='columns')
#
# year1 = df1.drop('capita',axis='columns')
# print(year1)
#
# capita1 = df1.capita
# print(capita1)
#
# # Create linear regression object
# reg1 = linear_model.LinearRegression()
# reg1.fit(year1,capita1)
#
# reg1.predict([[2020]])
#
# # %matplotlib inline
# plt.xlabel('year',fontsize=20)
# plt.ylabel('percapita',fontsize=20)
# plt.scatter(df1.year,df1.capita,color='red',marker='+')
# plt.plot(df1.year,reg1.predict(df1[['year']]),color='blue')



















import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    print(n*m_y*m_x)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
                marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

def main():
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {} \
		\nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()
