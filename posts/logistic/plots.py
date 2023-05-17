import plotnine as p9

def plot_smooth_logistic_curve(df):
	return (
		p9.ggplot(df, p9.aes('x', 'p_x'))
		+ p9.geom_line()
		+ p9.scale_y_continuous(name='p(x)')
		+ p9.theme_classic()
	)

def plot_naive_logistic_fit(sample_df, curve_df):
	return (
		p9.ggplot(sample_df, p9.aes('x'))
		+ p9.geom_line(
			data=curve_df,
			mapping=p9.aes(y='p_x')
		)
		+ p9.geom_point(
			mapping=p9.aes(y='y', fill='factor(y)'),
			size=4,
			alpha=0.7,
			shape='o'
		)
		+ p9.scale_y_continuous(name='y and p(x)')
		+ p9.labels.ggtitle('Naïve least-square fit')
		+ p9.theme_classic()
		+ p9.theme(legend_position='none')
	)

def plot_logistic_fit(df, logloss_df):
	return (
		p9.ggplot(df, p9.aes('x'))
		+ p9.geom_line(
			data=logloss_df,
			mapping=p9.aes(y='log_loss', color='factor(y)'),
		)
		+ p9.geom_linerange(
			mapping=p9.aes(ymin=0, ymax='log_loss', color='factor(y)'),
			linetype='dashed',
		)
		+ p9.geom_point(
			mapping=p9.aes(y=0, fill='factor(y)'),
			size=4,
			alpha=0.7,
			shape='o',
		)
		+ p9.scale_y_continuous(name='log loss')
		+ p9.theme_classic()
		+ p9.theme(legend_position='none')
	)

def plot_logistic_fit_panel(df, logloss_df, wrap):
	logistic_plot = plot_logistic_fit(df, logloss_df)
	logistic_plot += p9.theme(figure_size=(12, 4), subplots_adjust={'wspace': 0.02})
	logistic_plot += p9.facet_wrap(wrap, labeller='label_both', scales='free_y')
	return logistic_plot
