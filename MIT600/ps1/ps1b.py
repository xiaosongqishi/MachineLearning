# -*- coding: utf-8 -*-
'''
@Time    : 2023/3/17 20:39
@Author  : qishi
@File    : ps1b.py

'''
# total_cost = 0.0  # the cost of dream home
portion_down_payment = 0.25  # the portion of the cost needed for a down payment
current_savings = 0.0  # the amount that have saved
r = 0.04
annual_salary = float(input("Enter your annual salary:"))
portion_saved = float(input("Enter the percent of your salary to save, as a decimal:"))
total_cost = float(input("Enter the cost of your dream house:"))
semi_annual_raise = float(input("Enter the semi-annual raise,as a decimal:"))
down_payment = total_cost * portion_down_payment
monthly_salary = annual_salary / 12
months = 0
while True:
    monthly_gain = current_savings * r / 12
    current_savings += monthly_salary * portion_saved + monthly_gain
    months += 1
    if months % 6 == 0:
        monthly_salary += monthly_salary * semi_annual_raise
    if current_savings >= down_payment:
        break
print("Number of months:", months)
