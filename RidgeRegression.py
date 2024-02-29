print("Software is working.")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

print("Reading the dataset.")
#Read dataset
cars_df = pd.read_csv("df.csv")
#Drop the row if has a empty feature
cars_df = cars_df.dropna()
#We are going to use a dummies to make features more understandable for the phyton
cars_dummies = pd.get_dummies(cars_df)

#We declared features and target columns
features = cars_dummies.drop("price", axis = 1).values
targetValues = cars_dummies["price"].values

#We split our features and target datas into 4 data set which are test and train sets for our model
features_train, features_test, target_train, target_test = train_test_split(features, targetValues, test_size=0.3, random_state=20)

print("Finding best alpha for ridge regression model.")
#Find best alpha for ridge regression model
scores=[]
values = [0.1, 1.0, 10.0, 100.0, 1000.0]
for alpha in values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(features_train, target_train)
    scores.append(ridge.score(features_test, target_test))

max_index = scores.index(max(scores))
best_alpha = values[max_index]
print(f"Best alpha is: {best_alpha}")

print("Training ridge regression model.")
ridge_best_alpha = Ridge(alpha = best_alpha)
ridge_best_alpha.fit(features_train, target_train)
target_pred_best_alpha = ridge_best_alpha.predict(features_test)

print("Which one do you want to do?\n1)Show metrics\n2)Make Prediction")
choice = int(input())
if choice == 1:
    
    RMSE = mean_squared_error(target_test, target_pred_best_alpha, squared=False)
    RSquared = r2_score(target_test, target_pred_best_alpha)
    mape = mean_absolute_percentage_error(target_test, target_pred_best_alpha) / len(target_test) # type: ignore
    mean_price = np.mean(target_pred_best_alpha)

    print("RMSE:", RMSE)
    print("MAPE:", mape)
    print("r2:", RSquared)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Bar chart for R-squared and MAPE
    labels_r2_mape = ['R-squared', 'MAPE']
    values_r2_mape = [RSquared, mape]
    axes[0].bar(labels_r2_mape, values_r2_mape, color=['blue', 'orange'])
    axes[0].set_ylabel('Values')
    axes[0].set_title('Comparison of R-squared and MAPE (Logarithmic Scale)')
    axes[0].set_yscale('log')  # Use a logarithmic scale for better visibility

    # Bar chart for Mean Price and RMSE
    labels_mean_rmse = ['Mean Price', 'RMSE']
    values_mean_rmse = [mean_price, RMSE]
    axes[1].bar(labels_mean_rmse, values_mean_rmse, color=['blue', 'orange'])
    axes[1].set_ylabel('Values')
    axes[1].set_title('Comparison of Mean Price and RMSE')

    plt.tight_layout()
    plt.show()  

    #Chart of comparison actual and predicted values
    plt.scatter(target_test, target_pred_best_alpha)
    plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], '--', color='red', label="Perfect Line")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Ridge Regression: Actual vs Predicted (alpha={best_alpha})")
    plt.legend
    plt.show()


    # Accuracy plot
    plt.plot(values, scores, marker='o')
    plt.xscale('log')  # Use a logarithmic scale for better visibility
    plt.xlabel('Alpha Values (log scale)')
    plt.ylabel('R-squared Score')
    plt.title('Ridge Regression Accuracy Across Alpha Values')
    plt.show()

    mileage_column_index = cars_dummies.columns.get_loc('mileage')

    # Kilometer car price graph
    plt.scatter(features_test[:, (mileage_column_index)], target_pred_best_alpha)
    plt.xlabel('Kilometer')
    plt.ylabel('Car Price')
    plt.title('Scatter Plot: Kilometer vs Car Price')
    plt.show()


    hp_column_index = cars_dummies.columns.get_loc('hp')

    # Scatter plot
    plt.scatter(features_test[:, (hp_column_index - 1)], target_pred_best_alpha)
    plt.xlabel('Horsepower (hp)')
    plt.ylabel('Car Price')
    plt.title('Scatter Plot: Horsepower vs Car Price')
    plt.show()


    year_column_index = cars_dummies.columns.get_loc('year')

    # Kilometer car price graph
    plt.scatter(features_test[:, (year_column_index -1)], target_pred_best_alpha)
    plt.xlabel('year')
    plt.ylabel('Car Price')
    plt.title('Scatter Plot: Year vs Car Price')
    plt.show() 
elif choice == 2:
    
    print("Brands")
    print(cars_df["make"].unique())
    make_input = input("Enter the brand of car: ")

    print("Models")
    print(cars_df[cars_df["make"] == make_input]["model"].unique())
    model_input = input("Enter model of the car: ")

    print("Fuel Types")
    print(cars_df[cars_df["model"] == model_input]["fuel"].unique())
    fuel_input = input("Enter fuel type of the car: ")

    print("Gear Types")
    print(cars_df[cars_df["model"] == model_input]["gear"].unique())
    gear_input = input("Enter gear type of the car: ")

    print("Offer Types")
    print(cars_df[(cars_df["model"] == model_input) & (cars_df["gear"] == gear_input)]["offerType"].unique())
    offer_input = input("Enter offer type: ")

    print("Horse Power")
    if cars_df[
        (cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (cars_df["offerType"] == offer_input)][
        "hp"].unique().size != 0:
        print(cars_df[(cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (
                    cars_df["offerType"] == offer_input)]["hp"].unique())
    hp_input = float(input("Enter horse power of the car: "))

    print("Year")
    if cars_df[
        (cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (cars_df["offerType"] == offer_input) & (
                cars_df["hp"] == hp_input)]["year"].unique().size != 0:
        print(cars_df[(cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (
                    cars_df["offerType"] == offer_input) & (cars_df["hp"] == hp_input)]["year"].unique())
    year_input = int(input("Enter year of the car: "))

    print("Kilometer")
    if cars_df[
        (cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (cars_df["offerType"] == offer_input) & (
                cars_df["hp"] == hp_input) & (cars_df["year"] == year_input)]["mileage"].unique().size != 0:
        print((cars_df[(cars_df["model"] == model_input) & (cars_df["gear"] == gear_input) & (
                    cars_df["offerType"] == offer_input) & (cars_df["hp"] == hp_input) & (
                                   cars_df["year"] == year_input)]["mileage"].unique()))
    km_input = int(input("Enter kilometer of the car: "))

    user_data = pd.DataFrame({
        'mileage': [km_input],
        'make': [make_input],
        'model': [model_input],
        'fuel': [fuel_input],
        'gear': [gear_input],
        'offerType': [offer_input],
        'hp': [hp_input],
        'year': [year_input]
    })

    user_predict_dummies = pd.get_dummies(user_data)
    user_predict_dummies = user_predict_dummies.reindex(columns=cars_dummies.columns.drop(["price"]), fill_value=0)

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(features_train, target_train)
    
    prediction = ridge.predict(user_predict_dummies.values)
    print(prediction)
else:
    print("You entered wrong parameter")


