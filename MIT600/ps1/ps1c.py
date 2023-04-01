# -*- coding: utf-8 -*-
'''
@Time    : 2023/3/17 20:54
@Author  : qishi
@File    : ps1c.py

'''
# total_cost = 0.0  # the cost of dream home
portion_down_payment = 0.25  # the portion of the cost needed for a down payment
current_savings = 0.0  # the amount that have saved
r = 0.04
annual_salary = 150000  # float(input("Enter the starting salary:"))
# portion_saved = float(input("Enter the percent of your salary to save, as a decimal:"))
total_cost = 1000000  # float(input("Enter the cost of your dream house:"))
semi_annual_raise = .07  # float(input("Enter the semi-annual raise,as a decimal:"))
down_payment = total_cost * portion_down_payment
monthly_salary = annual_salary / 12
months = 0
step = 0
left = 0
right = 10000


def isOK(portion_saved):
    current_savings = 0
    months = 0
    monthly_salary = annual_salary / 12
    while months <= 36:
        monthly_gain = current_savings * r / 12
        current_savings += monthly_salary * portion_saved + monthly_gain
        months += 1
        if months % 6 == 0:
            monthly_salary += monthly_salary * semi_annual_raise
        if current_savings >= down_payment:
            return True
    return False


if not isOK(1):
    print("It is not possible to pay the down payment in three years.")
    exit(0)

while left < right:
    mid = (left + right) // 2
    portion_saved = mid / 10000
    if right - left <= 1:
        break
    if isOK(portion_saved):
        step += 1
        right = mid
        continue
    else:
        step += 1
        left = mid
        continue

print(portion_saved)
print(step)
