import plotnine as p9
import pandas as pd
import numpy as np
from plotnine.animation import PlotnineAnimation
from matplotlib import rc
rc('animation', html='html5')

def plot_cost_vs_beta(negloglik_sweep_betas_df, width):
    return (
        p9.ggplot(negloglik_sweep_betas_df, p9.aes('beta', 'neg_log_likelihood', fill='subject'))
        + p9.geom_col(width=width)
        + p9.scale_x_continuous(expand=[0, 0])
        + p9.scale_y_continuous(name='negative log-likelihoods ("cost")', expand=[0, 0])
        + p9.theme_classic()
    )

def animate_subject_event_times_and_mark_at_risk(df):
    events_df = df.sort_values('time').query('event == 1')
    subjects = events_df['subject']
    times = events_df['time']
    plots = (
        plot_subject_event_times_and_mark_at_risk(df, time, subject)
        for subject, time in zip(subjects, times)
    )
    ani = PlotnineAnimation(plots, interval=2000, repeat_delay=2000)
    return ani
    
def plot_subject_event_times(df, color_map='factor(x)'):
    return (
        p9.ggplot(
            df, p9.aes(x='time', y='subject', color=color_map)
        )
        + p9.geom_segment(
            p9.aes(x=0, xend='time', yend='subject'), size=2
        )
        + p9.geom_point(data=df[df['event'] == 1], shape='o', size=5, show_legend=False)
        + p9.scale_x_continuous(expand=[0, 0], breaks=range(7))
        + p9.coord_fixed(0.5)
        + p9.theme_classic()
        + p9.theme(
            axis_line_y=p9.element_blank(),
            axis_ticks_major_y=p9.element_blank()
        )
    )

# time = event_time_and_x_from_subject(df, subject)[0]
def plot_subject_event_times_and_mark_at_risk(df, time, subject, color_map='factor(x)'):
    at_risk_annotation_df = _subjects_at_risk_at_event_time(df, time)
    likelihood_label_df = pd.DataFrame({
        'x': np.max(df['time']),
        'y': 1.5,
        'likelihood': [_create_latex_expression_likelihood(df, time, subject)]
    })
    return (
        p9.ggplot(
            df, p9.aes(x='time', y='subject', color=color_map)
        )
        + p9.geom_segment(
            p9.aes(x=0, xend='time', yend='subject'), size=2
        )
        + p9.geom_point(
            data=df[df['event'] == 1], shape='o', size=5, show_legend=False
        )
        + p9.geom_point(
            data=at_risk_annotation_df, shape='o', color='black', fill='#ffffff00', size=5, stroke=0.75
        )
        + p9.geom_text(
            p9.aes('x', 'y', label='likelihood'),            
            likelihood_label_df,
            ha='right',
            inherit_aes=False,
            parse=True,
            size=20,
        )
        + p9.scale_x_continuous(
            expand=[0, 0], breaks=range(7)
        )
        + p9.coord_fixed(0.5)
        + p9.theme_classic()
        + p9.theme(
            axis_line_y=p9.element_blank(),
            axis_ticks_major_y=p9.element_blank(),
            subplots_adjust={'right': 0.8, 'bottom': 0.2}
        )
    )

def _create_latex_expression_likelihood(df, time, subject):
    subjects_at_risk_df = _subjects_at_risk_at_event_time(df, time)
    likelihoods_at_risk = ['L_' + subject_at_risk for subject_at_risk in subjects_at_risk_df['subject']]
    return '\\frac{L_' + subject + '}{' + ' + '.join(likelihoods_at_risk) + '}'

def _subjects_at_risk_at_event_time(df, time):
    subjects_at_risk = df.query(f'time >= {time}')['subject']
    return pd.DataFrame({
        'time': [time] * len(subjects_at_risk),
        'subject': subjects_at_risk
    })

def _subjects_at_risk_per_event_time(df):
    all_event_times = df.query(f'event == 1')['time']
    all_subjects_at_risk_at_time = [
        _subjects_at_risk_at_event_time(df, time)
        for time in all_event_times
    ]
    return pd.concat(all_subjects_at_risk_at_time)
