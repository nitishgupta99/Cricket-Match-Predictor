import os

class color:
   CYAN = '\033[96m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

print(color.BOLD + color.CYAN + color.UNDERLINE + '\nWelcome to Cricket Wizard! Our CMPT 353 Final Project.\n' + color.END)
print('This is an analysis of a method to predict T20 fixture results and its batting/bowling averages:\n')
print('1- 2018 IPL Match and championship prediction.')
print('2- 2018 IPL Batting/Bowling averages prediction.\n')

value = input("Please enter the results that you are intrested to see.(1 or 2):\n")

while (value != '1' and value != '2'):
	value = input(color.RED + 'Wrong entry type! Please re-enter your answer in form of "1" or "2":\n' + color.END)

if(value == "1"):
	file_name = 'python ipl_predictor_script.py'
if(value == "2"):
	file_name = 'python index.py'

secondary_check = value

print('\n')
# https://stackoverflow.com/questions/18739239/python-how-to-get-stdout-after-running-os-system
result_code = os.system(file_name + ' > output.txt')
if os.path.exists('output.txt'):
    fp = open('output.txt', "r")
    output = fp.read()
    fp.close()
    os.remove('output.txt')
    print(output)

if(value == "1"):
	value = input(color.CYAN + "Would you like to see batting averages as well?(y or n):\n" + color.END)
	if(value == '2'):
		value = '0'
if(value == "2"):
	value = input(color.CYAN + "Would you like to see IPL prediction as well?(y or n):\n" + color.END)

if (value == 'y' or 'Y'):
	if (secondary_check == '1'):
		file_name = 'python index.py'
	if (secondary_check == '2'):
		file_name = 'python ipl_predictor_script.py'

while (value != 'y' and value != 'Y' and value != 'N' and value != 'n'):
	value = input(color.RED + 'Wrong entry type! Please re-enter your answer in form of "y" or "n":\n' + color.END)

if(value != 'n' and value != 'N'):
	print('\n')
	result_code = os.system(file_name + ' > output.txt')
	if os.path.exists('output.txt'):
	    fp = open('output.txt', "r")
	    output = fp.read()
	    fp.close()
	    os.remove('output.txt')
	    print(output)


print('\nThank you for using The Cricket Wizard!\n')
print(color.BOLD + color.CYAN + 'Josie Buter\nPooya Jamali\nLakshay Sethi\n')

