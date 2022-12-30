import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

slr = pd.read_csv(r"C:\Users\chink\Desktop\ML CA2\SLR_Salary_Data.csv")
slr.describe()
slr.info()
slr.duplicated().sum()
slr.isna().sum()
import matplotlib.pyplot as plt # mostly used for visualization purposes

plt.bar(height = slr.YearsExperience, x = np.arange(1, 30, 1))
plt.hist(slr.YearsExperience) #histogram
plt.title('YearsExperience')

plt.boxplot(slr.YearsExperience) #boxplot
plt.title('YearsExperience')


plt.bar(height = slr.Salary, x = np.arange(1, 30, 1))
plt.hist(slr.Salary) #histogram
plt.title('Salary')

plt.boxplot(slr.Salary) #boxplot
plt.title('Salary')


import seaborn as sns
sns.distplot(slr['YearsExperience'])
sns.distplot(slr['Salary'])

plt.scatter(x = slr['YearsExperience'], y = slr['Salary'], color = 'green')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')


corr1 = np.corrcoef(slr.YearsExperience, slr.Salary)

cov_output = np.cov(slr.YearsExperience, slr.Salary)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = slr).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(slr['YearsExperience']))

# Regression Line
plt.scatter(slr.YearsExperience, slr.Salary)
plt.plot(slr.YearsExperience, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res1 = slr.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# error = 5592.043608760661



######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(slr['YearsExperience']), y = slr['Salary'], color = 'brown')
corr2 = np.corrcoef(np.log(slr.YearsExperience), slr.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = slr).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(slr['YearsExperience']))

# Regression Line
plt.scatter(np.log(slr.YearsExperience), slr.Salary)
plt.plot(np.log(slr.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = slr.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# not correct

# error = 10302.893706228308


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = slr['YearsExperience'], y = np.log(slr['Salary']), color = 'orange')
corr3 = np.corrcoef(slr.YearsExperience, np.log(slr.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = slr).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(slr['YearsExperience']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(slr.YearsExperience, np.log(slr.Salary))
plt.plot(slr.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = slr.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

# error = 7213.235076620129



model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = slr).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(slr))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = slr.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(slr.YearsExperience, np.log(slr.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Observed data', 'Predicted line'])
plt.show()

corr4 = np.corrcoef(slr.YearsExperience, np.log(slr.Salary))

# Error calculation
res4 = slr.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# error = 5391.081582693624
# best model less error

data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

data1 = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([corr1, corr2, corr3, corr4])}
table_corr = pd.DataFrame(data1)
table_corr


from sklearn.model_selection import train_test_split

train, test = train_test_split(slr, test_size = 0.2)

finalmodel = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Salary = np.exp(test_pred)
pred_test_Salary

# Model Evaluation on Test data
test_res = test.Salary - pred_test_Salary
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# error = 7274.531679945997

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Salary = np.exp(train_pred)
pred_train_Salary

# Model Evaluation on train data
train_res = train.Salary - pred_train_Salary
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#Error = 4885.785578064487
