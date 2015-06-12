from  datetime import timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as fm
import matplotlib.pyplot as plt
import entropy_estimators as ee
import scipy.stats as stats
import itertools

def linear_regression_analysis(dataframe, x_independent, y_dependent, report):
	formula = '%s ~ %s' % (y_dependent, x_independent)
	results = fm.ols(formula, data=dataframe).fit()

	fig_name = '%s vs. %s' % (x_independent, y_dependent)
	report.writelines([fig_name, ''])
	report.writelines(str(results.summary()))
	report.write('\n')
	report.flush()

	plt.figure(fig_name)
	plt.scatter(dataframe[x_independent], dataframe[y_dependent])
	plt.xlabel(x_independent)
	plt.ylabel(y_dependent)
	plt.savefig(fig_name + '.pdf')


def multiple_regression_analysis(dataframe, Xs_independent, y_dependent, report):
	X = sm.add_constant(dataframe[Xs_independent])
	est = sm.OLS(dataframe[y_dependent], X).fit()
	
	report.writelines(str(est.summary()))
	report.write('\n')
	report.flush()

	# length = len(Xs_independent)
	# for i in xrange(length):
	# 	for j in xrange(i, length):
	# 		print(X_variables[i].corr(Xs_independent[j]))

def non_parametric_analysis(dataframe, xs_independent, y_dependent, report):
	## Correlation Indices
	report.write('Correlacion no parametrica - Tau de Kendall (tau, p-valor)\n')
	report.write('(Valores cercanos a 1 indican alta correlacion, valores cercanos a -1 indican que no hay.)\n')

	for x in xs_independent:
		## Kendall's Tau
		ktau, p_value = stats.kendalltau(dataframe[x],dataframe[y_dependent])
		report.write('-%s y %s: (%s,%s)\n' %(x, y_dependent, ktau, p_value))

	report.write('\n=============================================================================\n')
	report.write('Correlacion no parametrica - Spearman (rho, p-valor)\n')
	report.write('(Valores cercanos a 1 y a -1 indican correlacion fuerte, para 0 no hay correlacion)\n')
	### Spearman Coefficient
	# print xs_independent
	combos = list(xs_independent) + [y_dependent] 
	report.write('-' + str(combos) + '\n')
	rho, p_value = stats.spearmanr(dataframe[combos])
	# print(rho, p_value)
	report.writelines(str(rho)[1:-1] + '\n')
	report.writelines(str(p_value)[1:-1]+ '\n')
	report.flush()
			

def mutual_information(dataframe, measure1, measure2, report):
	name = '%s - %s' % (measure1, measure2)

	vector_list_x = ee.vectorize(dataframe[measure1]) 
	vector_list_y = ee.vectorize(dataframe[measure2])

	mutual_information = ee.mi(vector_list_x, vector_list_y)
	
	report_line = 'Informacion Mutua %s: %s\n'%(name, mutual_information)
 	
 	report.write(report_line)
 	report.flush()


def get_timedelta(x):
	d, h, m, s = x.split(':')
	return timedelta(days = int(d), hours = int(h), seconds=int(s), minutes = int(m))

def main():
	report_sheet = open('report_sheet.txt', 'a+')

	dataset = pd.read_csv('dataset.csv')
	dataset = dataset[pd.notnull(dataset['JTR_Time'])]
	dataset['Time'] = dataset['JTR_Time'].apply(lambda x: get_timedelta(x).seconds)

	headers = dataset.keys()[1:-2]

	figure = 1
	for measure in dataset.keys()[1:-2]:
		
		linear_regression_analysis(dataset, measure, 'Time', report_sheet)
		figure += 1

	report_sheet.write('\n=============================================================================\n')

	multiple_regression_analysis(dataset, headers , 'Time', report_sheet)

	report_sheet.write('\n=============================================================================\n')

	non_parametric_analysis(dataset, dataset.keys()[1:-2], 'Time', report_sheet)
	
	report_sheet.write('\n=============================================================================\n')

	length = len(dataset.keys()[1:-2])
	for i in xrange(length):
		mutual_information(dataset, headers[i], 'Time', report_sheet)
		for j in xrange(i+1, length):
			mutual_information(dataset, headers[i], headers[j], report_sheet)

	report_sheet.flush()
	report_sheet.close()


if __name__ == '__main__':
	main()