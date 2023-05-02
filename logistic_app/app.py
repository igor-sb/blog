from shiny import App, render, ui, reactive
from plotnine import (
    ggplot,
	aes,
	geom_point,
	geom_line,
	geom_linerange,
	labels,
	scale_y_continuous,
)
import numpy as np
import pandas as pd
from scipy.stats import uniform, bernoulli

def logistic(x, x0, k):
	return 1 / (1 + np.exp(-k*(x - x0)))

def log_loss(p_x, y):
	return -y * np.log(p_x) - (1 - y) * np.log(1 - p_x)

def create_test_data(n0s, n1s, x0, k):
	df = pd.DataFrame({
		'x': np.concatenate((
			uniform.rvs(loc=0, scale=3, size=n0s),
			uniform.rvs(loc=2, scale=3, size=n1s),
		))
	})
	df['px_model'] = logistic(df['x'], x0, k)
	df['y'] = bernoulli.rvs(df['px_model'])
	return df

def calculate_px(x0, k):
	x = np.linspace(0, 5, 100)
	px_df = pd.DataFrame({
		'x': x,
		'y': logistic(x, x0, k)
	})
	return px_df

def plot_data_and_px(df, px_df):
	return (
		ggplot(df, aes('x', 'y'))
		+ geom_line(data=px_df)
		+ geom_point(size=4, shape='o', alpha=0.7, mapping=aes(fill='factor(y)'))
	)

def plot_logistic_fit(df, x0, k):
	x = np.linspace(0, 5, 100)
	px = logistic(x, x0, k)

	df['px'] = logistic(df['x'], x0, k)
	df['log_loss'] = log_loss(df['px'], df['y'])

	print('Total log-loss: ', np.sum(df['log_loss']))

	log_loss_df = pd.DataFrame({
		'x': x,
		'log_loss_0': log_loss(px, y=0),
		'log_loss_1': log_loss(px, y=1),
	})

	return (
		ggplot(df, aes('x', 0, color='factor(y)', fill='factor(y)'))
		+ geom_line(data=log_loss_df, mapping=aes('x', 'log_loss_0'), inherit_aes=False, linetype='dashed')
		+ geom_line(data=log_loss_df, mapping=aes('x', 'log_loss_1'), inherit_aes=False, linetype='dashed')
		+ geom_linerange(mapping=aes(ymin=0, ymax='log_loss'), size=1)
		+ geom_point(size=4, shape='o', alpha=0.7)
		+ scale_y_continuous(name='log loss')
		+ labels.ggtitle("Total log-loss: " + str(np.sum(df['log_loss'])))
		# + ylim(0, 5)
	)

app_ui = ui.page_fluid(
	ui.h2("Logistic regression fits"),
	ui.input_numeric("x0", "x offset", value=2.5),
	ui.input_numeric("k", "steepness", value=3),
	ui.output_plot("logistic_plot"),
	ui.h2("Model fit"),
	ui.input_numeric("fit_x0", "x offset", value=2.5),
	ui.input_numeric("fit_k", "steepness", value=3),
	ui.output_plot("logistic_fits")
)


def server(input, output, session):

	@reactive.Calc
	def df():
		np.random.seed(1)
		return create_test_data(20, 20, input.x0(), input.k())

	@output
	@render.plot
	def logistic_plot():
		px_df = calculate_px(input.x0(), input.k())
		return plot_data_and_px(df(), px_df)
	
	@output
	@render.plot
	def logistic_fits():
		 return plot_logistic_fit(df(), input.fit_x0(), input.fit_k())

app = App(app_ui, server)
