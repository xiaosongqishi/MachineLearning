# total_cost = 0.0  # the cost of dream home
portion_down_payment = 0.25  # the portion of the cost needed for a down payment
current_savings = 0.0  # the amount that have saved
r = 0.04
# annual_salary = 0.0
# portion_saved = 0.0
# monthly_salary = 0.0
annual_salary = float(input("Enter your annual salary:"))
portion_saved = float(input("Enter the percent of your salary to save, as a decimal:"))
total_cost = float(input("Enter the cost of your dream house:"))
down_payment = total_cost * portion_down_payment
monthly_salary = annual_salary / 12
months = 0
while True:
    monthly_gain = current_savings * r / 12
    current_savings += monthly_salary * portion_saved + monthly_gain
    months += 1
    if current_savings >= down_payment:
        break
print("Number of months:", months)