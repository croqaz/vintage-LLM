import lance
import numpy as np
import plotly.express as px
import streamlit as st

ds = lance.dataset('lance-data/text_index.lance')
df = ds.to_table(columns=['length', 'words', 'sentences', 'quality_score', 'compression_ratio', 'char_entropy', 'unique_chars']).to_pandas()

st.title('Dataset Explorer')
st.dataframe(df.describe())

st.plotly_chart(px.histogram(df, x='length', nbins=80, title='Text Length Distribution'))
st.plotly_chart(px.histogram(df, x='words', nbins=80, title='Word Count Distribution'))
st.plotly_chart(px.histogram(df, x='sentences', nbins=80, title='Sentence Count'))

# ── Quality Score scatter with median, p90, and outlier highlighting ──
qs = df['quality_score'].astype(float)
median_qs = float(np.median(qs))
p90_qs = float(np.percentile(qs, 90))
iqr = float(np.percentile(qs, 75) - np.percentile(qs, 25))
low_fence = float(np.percentile(qs, 25) - 1.5 * iqr)
high_fence = float(np.percentile(qs, 75) + 1.5 * iqr)

is_outlier = (qs < low_fence) | (qs > high_fence)
df['_qs_label'] = np.where(is_outlier, 'Outlier', 'Normal')

fig_qs = px.scatter(
    df.reset_index(),
    x='index',
    y='quality_score',
    color='_qs_label',
    color_discrete_map={'Normal': '#636EFA', 'Outlier': '#FF4136'},
    opacity=0.4,
    title='Quality Score Distribution',
    labels={'index': 'Document Index', 'quality_score': 'Quality Score'},
)
fig_qs.add_hline(
    y=median_qs,
    line_dash='dash',
    line_color='green',
    line_width=2,
    annotation_text=f'Median: {median_qs:.3f}',
    annotation_position='top left',
)
fig_qs.add_hline(
    y=p90_qs, line_dash='dot', line_color='orange', line_width=2, annotation_text=f'P90: {p90_qs:.3f}', annotation_position='top left'
)
fig_qs.update_traces(marker_size=4, selector=dict(name='Normal'))
fig_qs.update_traces(marker_size=7, selector=dict(name='Outlier'))
st.plotly_chart(fig_qs, use_container_width=True)
