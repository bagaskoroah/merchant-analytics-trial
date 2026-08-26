# ============================================================
# 06 — Plotly Dash Dashboard
# Merchant Network Analytics — Capstone Project
#
# Jalankan: python 06_dashboard.py
# Buka di browser: http://localhost:8050
# ============================================================

import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import networkx as nx
import os
import datetime
import anthropic
from urllib.parse import parse_qs

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Palet visual BNI: orange sebagai aksen aksi dan teal sebagai warna institusional.
THEME = {
    'bni_orange': '#F15A23',
    'bni_orange_accent': '#F7941D',
    'bni_teal': '#006A71',
    'bni_teal_dark': '#004B50',
    'surface': '#FFFFFF',
    'canvas': '#F3F7F7',
    'border': '#DCE8E8',
    'text_dark': '#15383C',
    'bni': '#007A7F',
    'target': '#D8892B',
    'priority': '#F15A23',
    'muted': '#68787A',
}

GRAPH_CONFIG = {
    'displayModeBar': False,
    'displaylogo': False,
    'responsive': True,
}

_BULAN_ID = ['', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
             'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']


def _format_tanggal_id(dt):
    return f"{dt.day} {_BULAN_ID[dt.month]} {dt.year}"


# ============================================================
# LOAD DATA & MODELS
# ============================================================
# Notebook 04 menghasilkan tabel readable yang mencakup seluruh merchant,
# acquisition score (jika merchant masuk affinity graph), dan rekomendasi produk.
_MERCHANTS_FULL_PATH = f'{PROJECT_ROOT}/data/processed/merchants_with_product_rec.csv'
merchants_display = pd.read_csv(_MERCHANTS_FULL_PATH)
affinity_edge_table = pd.read_csv(f'{PROJECT_ROOT}/data/processed/affinity_edge_table.csv')

DATA_UPDATED_AT = _format_tanggal_id(
    datetime.datetime.fromtimestamp(os.path.getmtime(_MERCHANTS_FULL_PATH))
)

with open(f'{PROJECT_ROOT}/models/graph_object.pkl', 'rb') as f:
    G = pickle.load(f)

with open(f'{PROJECT_ROOT}/models/community_map.pkl', 'rb') as f:
    community_map = pickle.load(f)

# Acquisition priority sekarang composite score (notebook 03), bukan trained classifier —
# tidak ada lagi model_metadata.pkl / best_model.pkl untuk acquisition priority.
priority_weights = joblib.load(f'{PROJECT_ROOT}/models/priority_weights.pkl')
product_model_metadata = joblib.load(f'{PROJECT_ROOT}/models/product_rec_metadata.pkl')

try:
    with open(f'{PROJECT_ROOT}/models/merchant_customers.pkl', 'rb') as f:
        merchant_customers = pickle.load(f)
except FileNotFoundError:
    merchant_customers = {}

LOGO_PATH = f'{PROJECT_ROOT}/assets/bni_logo.png'
HAS_LOGO = os.path.exists(LOGO_PATH)

# ============================================================
# NORMALIZE DISPLAY TABLE
# ============================================================
merchants_display['predicted_priority'] = merchants_display['predicted_priority'].fillna(0).astype(int)
merchants_display['priority_probability'] = merchants_display['priority_probability'].fillna(0)
for col in ['degree', 'weighted_degree', 'pagerank', 'betweenness', 'community_size',
            'connected_bni_ratio', 'avg_neighbor_jaccard', 'n_customers',
            'product_rec_proba_edc', 'avg_omzet_bulanan', 'community_id']:
    if col in merchants_display.columns:
        merchants_display[col] = merchants_display[col].fillna(0)
merchants_display['community_id'] = merchants_display['community_id'].astype(int)
merchants_display['product_recommendation'] = merchants_display[
    'product_recommendation'
].fillna('QRIS')

PR_Q75 = merchants_display['pagerank'].quantile(0.75)
PR_Q40 = merchants_display['pagerank'].quantile(0.40)
GRAPH_COVERAGE = int((merchants_display['degree'] > 0).sum())
PRODUCT_MODEL_AUC = float(
    product_model_metadata.get('test_metrics', {}).get('auc_roc', 0)
)
PRODUCT_MODEL_EXPERIMENTAL = PRODUCT_MODEL_AUC < 0.65

# ============================================================
# HELPER: Formatting & Business-Language Labels
# ============================================================
def format_rupiah_short(value):
    """Format angka rupiah jadi ringkas: Rp 31,1 Jt / Rp 1,2 M / Rp 850 Rb"""
    value = value or 0
    if value >= 1e9:
        s = f"{value/1e9:.1f}".replace('.', ',')
        return f"Rp {s} M"
    elif value >= 1e6:
        s = f"{value/1e6:.1f}".replace('.', ',')
        return f"Rp {s} Jt"
    elif value >= 1e3:
        return f"Rp {value/1e3:.0f} Rb"
    return f"Rp {value:.0f}"


def priority_badge(score):
    """Ganti angka priority_probability jadi badge kualitatif untuk RM."""
    if score >= 0.7:
        return ("Prioritas Tinggi", "danger")
    elif score >= 0.4:
        return ("Prioritas Sedang", "warning")
    return ("Prioritas Rendah", "secondary")


def influence_badge(pr_value):
    """Ganti angka pagerank mentah jadi badge tingkat pengaruh di jaringan."""
    if pr_value >= PR_Q75:
        return ("Tinggi", "success")
    elif pr_value >= PR_Q40:
        return ("Sedang", "warning")
    return ("Rendah", "secondary")


def get_produk_rekomendasi(row):
    """Tampilkan rekomendasi model beserta confidence untuk kelas terpilih."""
    recommendation = row.get('product_recommendation', 'QRIS') or 'QRIS'
    proba_edc = float(row.get('product_rec_proba_edc', 0.5) or 0.5)
    confidence = proba_edc if recommendation == 'QRIS + EDC' else 1 - proba_edc
    return f"{recommendation} ({confidence:.0%})"


def get_est_fee(row):
    """Estimasi potensi MDR bulanan dari omzet bulanan merchant."""
    monthly_omzet = float(row.get('avg_omzet_bulanan', 0) or 0)
    return monthly_omzet * 0.007  # asumsi MDR 0,7%


def fee_display_text(row, prefix="Est. MDR: "):
    monthly_omzet = float(row.get('avg_omzet_bulanan', 0) or 0)
    if monthly_omzet <= 0:
        return "Belum ada data omzet"
    return f"{prefix}{format_rupiah_short(get_est_fee(row))}/bulan"


def filter_non_bni_targets(kategori=None, kota=None):
    df = merchants_display[merchants_display['is_nasabah_bni'] == 'Tidak'].copy()
    if kategori and kategori != 'Semua':
        df = df[df['kategori'] == kategori]
    if kota and kota != 'Semua':
        df = df[df['kota'] == kota]
    return df.sort_values('priority_probability', ascending=False)


def legend_dot(color):
    return html.Span(style={
        "display": "inline-block", "width": "10px", "height": "10px",
        "borderRadius": "50%", "backgroundColor": color, "marginRight": "5px"
    })


def mini_stat(label, value, color=""):
    return html.Div([
        html.Div(label, className="quick-stat-label"),
        html.Div(value, className=f"quick-stat-value {color}"),
    ], className="quick-stat")


def apply_chart_theme(fig):
    """Apply presentation-only styling consistently across Plotly charts."""
    fig.update_layout(
        font=dict(family="Poppins, Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                  color=THEME['text_dark'], size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor=THEME['bni_teal_dark'],
            bordercolor=THEME['bni_teal_dark'],
            font=dict(color='#FFFFFF', family="Poppins, sans-serif", size=12),
        ),
        legend=dict(font=dict(color=THEME['muted'], size=11)),
    )
    fig.update_xaxes(
        showline=False, zeroline=False, gridcolor='#E8EFEF',
        tickfont=dict(color=THEME['muted']), title_font=dict(color=THEME['muted']),
    )
    fig.update_yaxes(
        showline=False, zeroline=False, gridcolor='#E8EFEF',
        tickfont=dict(color=THEME['muted']), title_font=dict(color=THEME['muted']),
    )
    return fig


# ============================================================
# HELPER: Ekosistem Bisnis (Community) Summary
# ============================================================
def compute_ecosystem_summary(min_size=3, exclude_unassigned=True):
    df = merchants_display.copy()
    if exclude_unassigned:
        df = df[df['community_id'] != -1]

    def summarize(g):
        customer_union = set()
        for merchant_id in g['merchant_id']:
            customer_union.update(merchant_customers.get(merchant_id, set()))
        return pd.Series({
            'size': len(g),
            'bni_count': (g['is_nasabah_bni'] == 'Ya').sum(),
            'non_bni_count': (g['is_nasabah_bni'] == 'Tidak').sum(),
            'bni_pct': (g['is_nasabah_bni'] == 'Ya').mean() * 100,
            'dominant_cat': g['kategori'].mode().iloc[0] if len(g['kategori'].mode()) > 0 else 'Mixed',
            'dominant_city': g['kota'].mode().iloc[0] if len(g['kota'].mode()) > 0 else 'Mixed',
            'avg_connections': g['degree'].mean(),
            'unique_customers': len(customer_union),
            'total_monthly_omzet': g['avg_omzet_bulanan'].sum(),
            'potential_fee': g.loc[g['is_nasabah_bni'] == 'Tidak'].apply(get_est_fee, axis=1).sum(),
        })

    per_community = df.groupby('community_id').apply(
        summarize, include_groups=False
    ).reset_index()
    per_community = per_community[per_community['size'] >= min_size].copy()

    # Beberapa community_id (hasil Louvain) berbeda bisa punya kategori & kota
    # dominan yang sama persis, sehingga muncul sebagai entri ekosistem yang
    # terlihat duplikat di dropdown. Gabungkan community_id dengan kombinasi
    # dominant_cat + dominant_city yang identik menjadi satu entri ekosistem.
    merged_ids = per_community.groupby(['dominant_cat', 'dominant_city'])['community_id'].apply(list)

    rows = []
    for (dominant_cat, dominant_city), community_ids in merged_ids.items():
        member_df = df[df['community_id'].isin(community_ids)]
        stats = summarize(member_df)
        stats['community_id'] = int(min(community_ids))
        stats['community_ids'] = [int(c) for c in community_ids]
        rows.append(stats)

    summary = pd.DataFrame(rows)
    summary['label'] = summary.apply(
        lambda r: f"{r['dominant_cat']} {r['dominant_city']} ({int(r['size'])} merchant)", axis=1
    )
    summary['community_id'] = summary['community_id'].astype(int)
    return summary.sort_values('size', ascending=False).reset_index(drop=True)


ecosystem_summary = compute_ecosystem_summary()
n_communities = len(ecosystem_summary)

# Peta community_id mentah (Louvain) -> community_id representatif di ecosystem_summary,
# dibutuhkan karena beberapa community_id mentah bisa digabung jadi satu entri ekosistem.
ECOSYSTEM_ID_MAP = {
    raw_id: int(row['community_id'])
    for _, row in ecosystem_summary.iterrows()
    for raw_id in row['community_ids']
}


def get_ecosystem_label(community_id):
    rep_id = ECOSYSTEM_ID_MAP.get(int(community_id), community_id)
    row = ecosystem_summary[ecosystem_summary['community_id'] == rep_id]
    if len(row) == 0:
        return f"Ekosistem #{community_id}"
    return row.iloc[0]['label']


# ============================================================
# NETWORK GRAPH
# ============================================================
TOP_N_NODES = 60


def get_matching_merchant_ids(view='all', kategori=None, kota=None):
    df = merchants_display
    if view not in (None, 'all'):
        if isinstance(view, (list, set, tuple)):
            df = df[df['community_id'].isin(view)]
        else:
            df = df[df['community_id'] == view]
    if kategori and kategori != 'Semua':
        df = df[df['kategori'] == kategori]
    if kota and kota != 'Semua':
        df = df[df['kota'] == kota]
    return df


def build_network_figure(view='all', kategori=None, kota=None, height=600):
    """Build plotly figure untuk network graph.
    Di halaman Overview, HANYA dikontrol oleh filter kategori/kota (satu sumber filter,
    tidak ada lagi dropdown ekosistem terpisah — lihat FIX 1 V4). Parameter `view` tetap
    ada untuk kebutuhan halaman Detail Ekosistem (menampilkan satu ekosistem spesifik).
    Kalau kategori & kota = 'Semua', dibatasi ke top N merchant paling berpengaruh supaya
    tidak overwhelming; kalau difilter, tampilkan seluruh merchant yang match.
    """
    df_match = get_matching_merchant_ids(view, kategori, kota)
    is_unfiltered_all = (view in (None, 'all')) and (not kategori or kategori == 'Semua') and (not kota or kota == 'Semua')

    if is_unfiltered_all:
        ids = df_match.nlargest(TOP_N_NODES, 'degree')['merchant_id'].tolist()
    else:
        ids = df_match['merchant_id'].tolist()
    ids = [i for i in ids if i in G.nodes()]
    sub = G.subgraph(ids).copy()

    if len(sub) == 0:
        fig = go.Figure()
        fig.update_layout(
            height=200, margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Tidak ada merchant untuk filter ini.<br>Coba pilih kategori atau kota lain.",
                               showarrow=False, font=dict(size=13, color=THEME['muted']))]
        )
        return apply_chart_theme(fig)

    if len(sub) > 1:
        layout = nx.spring_layout(
            sub, k=(2 if len(sub) < 40 else 1.3), iterations=50,
            seed=42, weight='jaccard'
        )
    else:
        layout = {list(sub.nodes())[0]: (0, 0)}

    edge_traces = []
    q90 = affinity_edge_table['jaccard'].quantile(0.9)
    for u, v, data in sub.edges(data=True):
        x0, y0 = layout[u]
        x1, y1 = layout[v]
        weight = data.get('jaccard', 0)
        shared_customers = int(data.get('shared_customers', 0))
        weight_normalized = min(1.0, weight / q90) if q90 else 0

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=max(0.25, weight_normalized * 1.1),
                color='rgba(52, 94, 98, 0.10)'
            ),
            hovertext=(f"Shared customers: {shared_customers}<br>"
                       f"Kesamaan pelanggan: {weight:.1%}"),
            hoverinfo='text',
            showlegend=False
        ))

    # Node trace — split by BNI status
    node_x, node_y, node_color, node_text, node_ids, pr_values, node_prs = [], [], [], [], [], [], []

    for node in sub.nodes():
        x, y = layout[node]
        m_info = merchants_display[merchants_display['merchant_id'] == node]

        if len(m_info) == 0:
            continue

        row = m_info.iloc[0]
        is_bni = row['is_nasabah_bni']  # string "Ya"/"Tidak"
        predicted_priority = row.get('predicted_priority', 0)
        pr = row.get('pagerank', 0)
        comm = row.get('community_id', -1)

        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        pr_values.append(pr)
        node_prs.append((node, pr, row.get('nama', '')))

        # Color logic — baca is_nasabah_bni sebagai string, BUKAN angka encoded
        if is_bni == 'Ya':
            node_color.append(THEME['bni'])
        elif predicted_priority == 1:
            node_color.append(THEME['priority'])
        else:
            node_color.append(THEME['target'])

        # Hover text — istilah bisnis, bukan istilah teknis
        nama = row.get('nama', 'Unknown')
        degree = int(row.get('degree', 0))
        bni_ratio = row.get('connected_bni_ratio', 0)
        hover = (
            f"<b>{nama}</b><br>"
            f"ID: {node}<br>"
            f"Kategori: {row.get('kategori', '-')}<br>"
            f"Kota: {row.get('kota', '-')}<br>"
            f"Status BNI: {is_bni}<br>"
            f"Tingkat pengaruh di jaringan: {pr:.4f}<br>"
            f"Merchant dengan pelanggan serupa: {degree} ({bni_ratio:.0%} sudah BNI)<br>"
            f"Jumlah pelanggan: {int(row.get('n_customers', 0))}<br>"
            f"Rata-rata kesamaan pelanggan: {row.get('avg_neighbor_jaccard', 0):.1%}<br>"
            f"Rekomendasi produk: {get_produk_rekomendasi(row)}<br>"
            f"Ekosistem bisnis: {get_ecosystem_label(int(comm)) if comm != -1 else 'Belum terklasifikasi'}"
        )
        node_text.append(hover)

    # Pilih label penting dengan collision guard. Seluruh nama tetap tersedia di hover,
    # tetapi label permanen hanya ditampilkan jika posisinya cukup jauh dari label lain.
    if len(node_prs) <= 10:
        max_label_count = len(node_prs)
    elif len(node_prs) <= 30:
        max_label_count = 9
    else:
        max_label_count = 8

    x_values = [layout[n][0] for n, _, _ in node_prs]
    y_values = [layout[n][1] for n, _, _ in node_prs]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    selected_labels = []
    for node_id, pr, nama in sorted(node_prs, key=lambda t: t[1], reverse=True):
        if pd.isna(nama) or not str(nama).strip():
            continue
        x, y = layout[node_id]
        x_norm = (x - x_min) / x_span
        y_norm = (y - y_min) / y_span
        overlaps_existing = any(
            abs(x_norm - chosen_x) < 0.22 and abs(y_norm - chosen_y) < 0.085
            for chosen_x, chosen_y, *_ in selected_labels
        )
        if not overlaps_existing:
            selected_labels.append((x_norm, y_norm, node_id, str(nama), x, y))
        if len(selected_labels) >= max_label_count:
            break

    label_annotations = []
    for _, _, node_id, nama, x, y in selected_labels:
        dx = (x - center_x) / x_span
        dy = (y - center_y) / y_span

        # Geser callout menjauh dari pusat graph agar tidak menutup node/edge utama.
        if abs(dx) > abs(dy) * 1.2:
            ax = 46 if dx >= 0 else -46
            ay = 0
            xanchor = 'left' if dx >= 0 else 'right'
            yanchor = 'middle'
        else:
            ax = 0
            ay = -34 if dy >= 0 else 34
            xanchor = 'center'
            yanchor = 'bottom' if dy >= 0 else 'top'

        display_name = nama if len(nama) <= 21 else f"{nama[:20]}…"
        label_annotations.append(dict(
            x=x, y=y, text=display_name,
            showarrow=True, arrowhead=0, arrowsize=0.7, arrowwidth=1,
            arrowcolor='rgba(61, 91, 94, 0.45)',
            ax=ax, ay=ay, xanchor=xanchor, yanchor=yanchor,
            bgcolor='rgba(255, 255, 255, 0.94)',
            bordercolor='rgba(200, 218, 218, 0.90)', borderwidth=1, borderpad=3,
            font=dict(size=9, color=THEME['text_dark']),
        ))

    # Ukuran node lebih terkendali agar cluster padat tetap terbaca.
    if pr_values:
        pr_min, pr_max = min(pr_values), max(pr_values)
        node_size = [10 + (pr - pr_min) / (pr_max - pr_min + 1e-10) * 34 for pr in pr_values]
    else:
        node_size = []

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_size, color=node_color, opacity=0.88,
            line=dict(width=1.2, color='rgba(255,255,255,0.95)')
        ),
        hovertext=node_text,
        hoverinfo='text',
        customdata=node_ids,
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=None,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=18, r=18, t=24, b=24),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[x_min - x_span * 0.16, x_max + x_span * 0.16],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[y_min - y_span * 0.18, y_max + y_span * 0.18],
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        annotations=label_annotations,
    )
    # Legend & caption ada sebagai komponen HTML terpisah di CardHeader, bukan di dalam plot
    return apply_chart_theme(fig)


def build_graph_legend():
    return html.Div([
        html.Span([legend_dot(THEME['bni']), "Sudah Nasabah BNI"], className="me-3 d-inline-flex align-items-center"),
        html.Span([legend_dot(THEME['priority']), "Target Prioritas Tinggi"], className="me-3 d-inline-flex align-items-center"),
        html.Span([legend_dot(THEME['target']), "Belum Nasabah BNI"], className="d-inline-flex align-items-center"),
    ], className="small mb-2 d-flex flex-wrap")


# ============================================================
# ANALYTICS CHARTS (halaman Analytics — statis, tanpa filter)
# ============================================================
def build_kategori_composition_figure():
    df = merchants_display.copy()
    df['Status BNI'] = df['is_nasabah_bni'].map({'Ya': 'Sudah Nasabah', 'Tidak': 'Belum Nasabah'})
    chart_data = df.groupby(['kategori', 'Status BNI']).size().reset_index(name='count')

    opp_order = (chart_data[chart_data['Status BNI'] == 'Belum Nasabah']
                 .sort_values('count', ascending=False)['kategori'].tolist())
    remaining = [k for k in chart_data['kategori'].unique() if k not in opp_order]
    kategori_order = opp_order + remaining

    fig = px.bar(chart_data, x='kategori', y='count', color='Status BNI',
                 barmode='group', text='count',
                 category_orders={'Status BNI': ['Sudah Nasabah', 'Belum Nasabah'], 'kategori': kategori_order},
                 color_discrete_map={'Sudah Nasabah': THEME['bni'], 'Belum Nasabah': THEME['target']},
                 labels={'count': 'Jumlah', 'kategori': 'Kategori'})
    fig.update_traces(textposition='outside')
    fig.update_layout(margin=dict(l=50, r=20, t=30, b=40), height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None))
    return apply_chart_theme(fig)


def build_ecosystem_opportunity_figure():
    """Peluang Akuisisi per Ekosistem — metrik = jumlah merchant belum nasabah (peluang),
    bukan penetrasi (supaya ekosistem dengan penetrasi rendah tampil sebagai peluang, bukan kesan negatif)."""
    comm_analysis = ecosystem_summary.sort_values('non_bni_count', ascending=False).head(7)
    if len(comm_analysis) == 0:
        return apply_chart_theme(go.Figure())
    comm_analysis = comm_analysis.sort_values('non_bni_count', ascending=True)

    fig = go.Figure(go.Bar(
        x=comm_analysis['non_bni_count'],
        y=comm_analysis['label'],
        orientation='h',
        marker_color=THEME['bni_orange'],
        text=comm_analysis['non_bni_count'].apply(lambda v: f"{int(v)} peluang"),
        textposition='outside',
        customdata=comm_analysis[['size', 'bni_pct']],
        hovertemplate=(
            "<b>%{y}</b><br>Merchant belum nasabah: %{x}<br>"
            "Total merchant: %{customdata[0]}<br>Penetrasi BNI saat ini: %{customdata[1]:.0f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        margin=dict(l=190, r=90, t=20, b=60), height=380,
        xaxis_title="Jumlah Merchant Belum Nasabah", yaxis_title=None,
    )
    return apply_chart_theme(fig)


def build_scatter_figure():
    """Pengaruh vs Peluang pada customer affinity graph."""
    df = merchants_display.copy()
    df['bni_connections'] = (df['degree'] * df['connected_bni_ratio']).round().astype(int)
    df['est_fee'] = df.apply(get_est_fee, axis=1)
    df['size_plot'] = df['est_fee'].clip(lower=max(df['est_fee'].max() * 0.02, 1))
    df['Status BNI'] = df['is_nasabah_bni'].map({'Ya': 'Sudah Nasabah', 'Tidak': 'Belum Nasabah'})

    fig = px.scatter(
        df, x='degree', y='bni_connections', size='size_plot', color='Status BNI',
        color_discrete_map={'Sudah Nasabah': THEME['bni'], 'Belum Nasabah': THEME['target']},
        hover_name='nama',
        hover_data={'kategori': True, 'kota': True, 'size_plot': False,
                    'degree': True, 'bni_connections': True,
                    'n_customers': True, 'avg_neighbor_jaccard': ':.1%'},
        labels={'degree': 'Merchant dengan Pelanggan Serupa',
                'bni_connections': 'Merchant Berbagi Pelanggan yang Sudah BNI'},
        opacity=0.75, size_max=32,
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=50), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title=None),
    )
    fig.add_shape(type='rect', xref='paper', yref='paper', x0=0.62, x1=1, y0=0.62, y1=1,
                  fillcolor='rgba(240,83,35,0.06)', line=dict(color='rgba(240,83,35,0.35)', dash='dash'))
    fig.add_annotation(xref='paper', yref='paper', x=0.985, y=0.97, text='★ Target Ideal', showarrow=False,
                        font=dict(size=11, color=THEME['bni_orange']), xanchor='right', yanchor='top')
    return apply_chart_theme(fig)


def build_fee_per_kota_figure():
    """Potensi MDR per kota berdasarkan omzet bulanan merchant non-BNI."""
    df = merchants_display[merchants_display['is_nasabah_bni'] == 'Tidak'].copy()
    df['est_fee'] = df.apply(get_est_fee, axis=1)
    grouped = df.groupby('kota')['est_fee'].sum().reset_index().sort_values('est_fee', ascending=False)

    fig = go.Figure(go.Bar(
        x=grouped['kota'], y=grouped['est_fee'],
        marker_color=THEME['bni_teal'],
        text=[format_rupiah_short(v) for v in grouped['est_fee']],
        textposition='outside',
        hovertext=[f"Potensi MDR: {format_rupiah_short(v)}/bulan" for v in grouped['est_fee']],
        hoverinfo='text',
    ))
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=50), height=380,
        yaxis_title="Potensi MDR (Rp/bulan)", xaxis_title=None,
    )
    return apply_chart_theme(fig)


# ============================================================
# HELPER: GenAI Agent (simplified for dashboard)
# ============================================================
CHAT_EXAMPLES = [
    "Top 5 target akuisisi",
    "Ekosistem mana yang paling potensial?",
    "Merchant prioritas di Jakarta",
    "Rekomendasi produk untuk merchant berpengaruh tinggi",
]


def get_top_targets_for_display(n=5, kategori=None, kota=None):
    return filter_non_bni_targets(kategori, kota).head(n)


def run_agent_simple(user_message):
    """
    Simplified agent untuk dashboard.
    Menggunakan Anthropic API dengan tool calling.
    Semua tool function membaca merchants_display yang sudah mencakup acquisition
    score, affinity metrics, dan product recommendation.
    PENTING: fungsi ini HANYA dipanggil dari callback tombol "Kirim" / klik contoh pertanyaan
    (lihat handle_chat) — tidak pernah dipanggil otomatis dari page load / filter / render lain,
    untuk kontrol biaya API.
    """
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
        from pathlib import Path

        # Selalu cari .env relatif terhadap lokasi file dashboard ini
        load_dotenv(Path(__file__).parent / '.env')

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return ("⚠️ API key Anthropic belum tersedia. Dashboard dan rekomendasi "
                    "lokal tetap dapat digunakan tanpa Asisten AI.")
        client = Anthropic(api_key=api_key)

        tool_schemas = [
            {
                "name": "get_top_acquisition_targets",
                "description": "Ambil daftar merchant prioritas akuisisi tertinggi",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Jumlah merchant (default 5)"},
                        "kategori": {"type": "string", "description": "Filter kategori"},
                        "kota": {"type": "string", "description": "Filter kota"}
                    }
                }
            },
            {
                "name": "get_merchant_network",
                "description": "Detail affinity merchant berdasarkan kesamaan pelanggan",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "merchant_id": {"type": "string", "description": "ID merchant"}
                    },
                    "required": ["merchant_id"]
                }
            },
            {
                "name": "calculate_acquisition_impact",
                "description": "Estimasi potensi MDR dan rekomendasi produk ML untuk akuisisi merchant",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "merchant_id": {"type": "string", "description": "ID merchant"}
                    },
                    "required": ["merchant_id"]
                }
            },
            {
                "name": "get_overview_stats",
                "description": "Statistik keseluruhan jaringan merchant",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]

        import sys
        sys.path.insert(0, PROJECT_ROOT)

        # Tool functions membaca tabel readable dan graph affinity undirected.
        def _get_top_acquisition_targets(n=5, kategori=None, kota=None):
            df = merchants_display[merchants_display['is_nasabah_bni'] == 'Tidak'].copy()
            if kategori: df = df[df['kategori'].str.lower() == kategori.lower()]
            if kota: df = df[df['kota'].str.lower() == kota.lower()]
            if len(df) == 0: return json.dumps({"error": "Tidak ada data"})
            top = df.nlargest(n, 'priority_probability').copy()
            top['rekomendasi_produk'] = top.apply(get_produk_rekomendasi, axis=1)
            return json.dumps({"targets": top[['merchant_id', 'nama', 'kategori', 'kota',
                'priority_probability', 'pagerank', 'degree', 'connected_bni_ratio',
                'n_customers', 'avg_neighbor_jaccard', 'rekomendasi_produk']].to_dict('records')},
                ensure_ascii=False, default=str)

        def _get_merchant_network(merchant_id):
            if merchant_id not in G.nodes(): return json.dumps({"error": "Not found"})
            neighbors = [{
                "id": neighbor,
                "shared_customers": int(G.edges[merchant_id, neighbor].get('shared_customers', 0)),
                "jaccard": float(G.edges[merchant_id, neighbor].get('jaccard', 0)),
            } for neighbor in G.neighbors(merchant_id)]
            neighbors.sort(key=lambda item: item['jaccard'], reverse=True)
            m = merchants_display[merchants_display['merchant_id'] == merchant_id].iloc[0]
            return json.dumps({"merchant": {"id": merchant_id, "nama": m.get('nama', ''),
                "pagerank": float(m.get('pagerank', 0)), "degree": int(m.get('degree', 0)),
                "n_customers": int(m.get('n_customers', 0))},
                "affinity_neighbors": neighbors}, ensure_ascii=False, default=str)

        def _calculate_acquisition_impact(merchant_id):
            info = merchants_display[merchants_display['merchant_id'] == merchant_id]
            if len(info) == 0: return json.dumps({"error": "Not found"})
            row = info.iloc[0]
            monthly_omzet = float(row.get('avg_omzet_bulanan', 0) or 0)
            fee = monthly_omzet * 0.007
            return json.dumps({
                "merchant_id": merchant_id,
                "monthly_omzet": monthly_omzet,
                "est_monthly_fee": float(fee),
                "est_annual_fee": float(fee * 12),
                "rekomendasi_produk": get_produk_rekomendasi(row),
                "recommendation_source": "product_rec_model",
                "affinity_neighbors": int(row.get('degree', 0)),
                "connected_bni_ratio": float(row.get('connected_bni_ratio', 0)),
            }, ensure_ascii=False, default=str)

        def _get_overview_stats():
            total = len(merchants_display); bni = (merchants_display['is_nasabah_bni'] == 'Ya').sum()
            return json.dumps({"total": total, "bni": int(bni), "non_bni": int(total - bni),
                "penetration": round(bni / total * 100, 1),
                "merchant_dalam_affinity_graph": GRAPH_COVERAGE,
                "jumlah_relasi_affinity": G.number_of_edges(),
                "product_model_experimental": PRODUCT_MODEL_EXPERIMENTAL}, default=str)

        tools_map = {
            "get_top_acquisition_targets": _get_top_acquisition_targets,
            "get_merchant_network": _get_merchant_network,
            "calculate_acquisition_impact": _calculate_acquisition_impact,
            "get_overview_stats": _get_overview_stats,
        }

        system = ("Kamu adalah AI assistant untuk tim Sales & Merchant Acquisition di Bank BNI. "
                  "Gunakan tools untuk menjawab. Jawab dalam Bahasa Indonesia yang profesional. "
                  "Sertakan angka konkret dan gunakan rekomendasi produk dari model ML. "
                  "Graph menunjukkan kesamaan pelanggan antarmerchant, bukan transfer dana langsung. "
                  "JANGAN gunakan istilah teknis seperti 'supplier', 'retailer', 'node', 'edge', 'hub', "
                  "'centrality', 'pagerank', atau 'jaccard' — gunakan bahasa bisnis seperti "
                  "'merchant dengan pelanggan serupa', 'tingkat kesamaan pelanggan', dan 'tingkat pengaruh'.")

        messages = [{"role": "user", "content": user_message}]

        for _ in range(5):
            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=2048,
                system=system, tools=tool_schemas, messages=messages
            )

            if resp.stop_reason == "tool_use":
                assistant_content = resp.content
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        fn = tools_map.get(block.name)
                        result = fn(**block.input) if fn else json.dumps({"error": "Tool not found"})
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
            else:
                return "".join(b.text for b in resp.content if hasattr(b, 'text'))

        return "Tidak dapat menyelesaikan analisis."

    except anthropic.RateLimitError:
        return ("⚠️ Kuota/rate limit Anthropic sedang tercapai. Coba kembali beberapa saat lagi; "
                "dashboard dan rekomendasi lokal tetap dapat digunakan.")
    except anthropic.AuthenticationError:
        return ("⚠️ API key Anthropic ditolak. Periksa ANTHROPIC_API_KEY pada file .env.")
    except anthropic.APIConnectionError:
        return ("⚠️ Layanan AI tidak dapat dijangkau dari jaringan saat ini. "
                "Periksa koneksi/proxy; dashboard dan rekomendasi lokal tetap tersedia.")
    except anthropic.APIStatusError as e:
        return (f"⚠️ Anthropic API mengembalikan HTTP {e.status_code}. "
                "Dashboard dan rekomendasi lokal tetap tersedia.")
    except Exception as e:
        return (f"⚠️ Asisten tidak tersedia saat ini ({type(e).__name__}). "
                "Gunakan filter dan Top Targets untuk analisis manual.")


# ============================================================
# BUILD DASH APP
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        'https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap',
    ],
    suppress_callback_exceptions=True,
    title="Merchant Network Analytics — BNI"
)

# Struktur HTML dasar. Seluruh presentation layer berada di assets/bni_theme.css.
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

# ============================================================
# PRECOMPUTE KPI VALUES
# ============================================================
total_merchants = len(merchants_display)
bni_count = (merchants_display['is_nasabah_bni'] == 'Ya').sum()
non_bni_count = total_merchants - bni_count
penetration = round(bni_count / total_merchants * 100, 1) if total_merchants else 0
high_priority = (merchants_display['predicted_priority'] == 1).sum()

# Potensi MDR bulanan: omzet bulanan merchant prioritas tinggi × asumsi MDR 0,7%.
top_priority_df = merchants_display[merchants_display['predicted_priority'] == 1]
potential_fee_income = top_priority_df.apply(get_est_fee, axis=1).sum() if len(top_priority_df) else 0

model_f1 = product_model_metadata.get('test_metrics', {}).get('f1', 0)
model_auc = product_model_metadata.get('test_metrics', {}).get('auc_roc', 0)


# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================
def kpi_card(title, value, value_color="", subtext="", link_href=None):
    card = dbc.Card(
        dbc.CardBody([
            html.H6(title, className="kpi-label"),
            html.H3(value, className=f"kpi-value {value_color}"),
            html.Small(subtext, className="kpi-meta"),
        ], className="d-flex flex-column justify-content-center h-100"),
        className="kpi-card"
    )
    if link_href:
        return dcc.Link(card, href=link_href, className="kpi-link")
    return card


def build_header():
    logo = html.Img(src='/assets/bni_logo.png', className="brand-logo") if HAS_LOGO else \
        html.Div("BNI", className="brand-mark")

    return dbc.Row([
        dbc.Col([
            html.Div([
                logo,
                html.Div([
                    html.H2("Merchant Network Analytics", className="brand-title"),
                    html.P("Merchant Acquisition Dashboard", className="brand-subtitle"),
                ]),
            ], className="brand-block"),
        ], width=12, lg=5),
        dbc.Col([
            html.Div(id='nav-links', className="nav-shell"),
        ], width=12, lg=4),
        dbc.Col([
            html.Div([
                html.Div(f"Data per: {DATA_UPDATED_AT}"),
                html.Small(
                    f"{GRAPH_COVERAGE} merchant terpetakan dalam jaringan",
                    className="text-muted"
                ),
            ], className="header-meta"),
        ], width=12, lg=3),
    ], className="app-header-row g-2 align-items-center")


def build_footer():
    """Info model teknis — hanya muncul di halaman Analytics, disembunyikan di balik
    expander supaya F1-Score/AUC tidak langsung tampil di halaman utama untuk tim non-teknis."""
    return html.Div([
        html.Hr(),
        html.Details([
            html.Summary("Lihat detail teknis", className="text-muted small",
                         style={"cursor": "pointer"}),
            html.Small(
                f"Acquisition priority: Composite Score (PageRank + BNI Ratio + Degree + Weighted Degree), bukan ML classifier · "
                f"Model rekomendasi produk: {product_model_metadata.get('model_name', '-')} · "
                f"F1-Score: {model_f1:.3f} · AUC-ROC: {model_auc:.3f}",
                className="text-muted d-block mt-1"
            )
        ])
    ], className="technical-footer")


def build_chat_empty_state():
    return html.Div([
        html.Div("Tanyakan apa saja tentang target akuisisi merchant.", className="chat-empty-title"),
        html.Div([
            dbc.Button(q, id={'type': 'chat-example', 'index': i}, size="sm", outline=True,
                       color="secondary", className="chat-suggestion")
            for i, q in enumerate(CHAT_EXAMPLES)
        ], className="chat-suggestions"),
    ], className="chat-empty-state")


def quick_summary_panel(n_filtered, n_target, n_priority, fee):
    return dbc.Row([
        dbc.Col(mini_stat("Merchant Tercakup", f"{n_filtered}"), width=6),
        dbc.Col(mini_stat("Target Akuisisi", f"{n_target}", "text-danger"), width=6),
        dbc.Col(mini_stat("Prioritas Tinggi", f"{n_priority}", "text-warning"), width=6),
        dbc.Col(mini_stat("Estimasi MDR", format_rupiah_short(fee), "text-primary"), width=6),
    ], className="quick-summary-grid")


# ============================================================
# PAGE 1: OVERVIEW ("Aksi Hari Ini")
# ============================================================
def overview_layout():
    return html.Div([
        dbc.Alert(
            "Rekomendasi produk masih bersifat eksperimental dan digunakan sebagai arahan awal, "
            "bukan keputusan final RM.",
            color="warning", className="model-alert", dismissable=True,
        ) if PRODUCT_MODEL_EXPERIMENTAL else None,

        # KPI Cards
        dbc.Row([
            dbc.Col(kpi_card("Total Merchant", f"{total_merchants}",
                             subtext=f"{GRAPH_COVERAGE} masuk peta jaringan"), width=6, md=4, lg=2),
            dbc.Col(kpi_card("Sudah Nasabah BNI", f"{bni_count}", value_color="text-success",
                              subtext=f"{penetration}% penetrasi"), width=6, md=4, lg=2),
            dbc.Col(kpi_card("Belum Nasabah", f"{non_bni_count}", value_color="text-danger",
                              subtext="peluang akuisisi"), width=6, md=4, lg=2),
            dbc.Col(kpi_card("Prioritas Tinggi", f"{high_priority}", value_color="text-warning",
                              subtext="siap digarap"), width=6, md=4, lg=2),
            dbc.Col(kpi_card("Potensi MDR", format_rupiah_short(potential_fee_income),
                              value_color="text-primary", subtext="estimasi dari volume transaksi merchant"), width=6, md=4, lg=2),
            dbc.Col(kpi_card("Ekosistem Teridentifikasi", f"{n_communities}", value_color="text-primary",
                              subtext="klik untuk detail →", link_href="/ekosistem"), width=6, md=4, lg=2),
        ], className="mb-3 g-3"),

        # Satu set filter untuk seluruh halaman — mengontrol graph + top targets + ringkasan
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Div("FILTER", className="section-eyebrow"),
                    html.H5("Fokus analisis", className="filter-title"),
                    html.P("Persempit tampilan untuk menyorot peluang yang paling relevan.",
                           className="filter-description"),
                ], className="filter-intro"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Kategori", className="fw-bold small mb-1"),
                        dcc.Dropdown(
                            id='filter-kategori',
                            options=[{'label': 'Semua', 'value': 'Semua'}] +
                                    [{'label': k, 'value': k} for k in sorted(merchants_display['kategori'].dropna().unique())],
                            value='Semua', clearable=False,
                        ),
                    ], width=6, md=3),
                    dbc.Col([
                        html.Label("Kota", className="fw-bold small mb-1"),
                        dcc.Dropdown(
                            id='filter-kota',
                            options=[{'label': 'Semua', 'value': 'Semua'}] +
                                    [{'label': k, 'value': k} for k in sorted(merchants_display['kota'].dropna().unique())],
                            value='Semua', clearable=False,
                        ),
                    ], width=6, md=3),
                    dbc.Col([
                        html.Div("Filter ini mengontrol network graph, top targets, dan ringkasan di bawah.",
                                 className="text-muted small mt-4")
                    ], width=12, md=6),
                ], className="g-3 align-items-end"),
            ])
        ], className="filter-card mb-3"),

        # ROW 1 (paling actionable): Network Graph + Asisten AI sejajar
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Peta Jaringan Pelanggan", className="mb-1"),
                        html.Small("Garis = kesamaan pelanggan · ukuran lingkaran = tingkat pengaruh · arahkan kursor untuk detail",
                                   className="text-muted d-block mb-2"),
                        build_graph_legend(),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='network-graph', figure=build_network_figure(), config=GRAPH_CONFIG),
                        html.Div(
                            html.P("Klik salah satu merchant di graph untuk melihat detail.", className="text-muted mb-0"),
                            id='node-detail-panel', className="mt-2"
                        ),
                    ])
                ], className="dashboard-card network-card")
            ], width=12, lg=8, className="mb-3 mb-lg-0"),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div("AI ASSISTANT", className="section-eyebrow"),
                        html.H5("Asisten Rekomendasi Akuisisi", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            html.Div(build_chat_empty_state(), id='chat-history', className="chat-history"),
                            type="dot", color=THEME['bni_orange']
                        ),
                        dbc.InputGroup([
                            dbc.Input(
                                id='chat-input', type='text',
                                placeholder='Tanya: "Merchant mana prioritas di Bogor kategori F&B?"',
                            ),
                            dbc.Button("Kirim", id='chat-send', color="primary", className="bni-button ms-2"),
                        ]),
                    ])
                ], className="dashboard-card assistant-card")
            ], width=12, lg=4),
        ], className="mb-3 g-3"),

        # ROW 2: Top Targets + Ringkasan Cepat
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.H5("Top Targets", className="mb-0"), width=12, md=6),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='target-limit-selector',
                                    options=[{'label': 'Tampilkan Top 5', 'value': 5}, {'label': 'Tampilkan Top 10', 'value': 10},
                                             {'label': 'Tampilkan Top 15', 'value': 15}, {'label': 'Tampilkan Semua', 'value': -1}],
                                    value=5, clearable=False,
                                ),
                            ], width=12, md=6),
                        ], className="g-2 align-items-center")
                    ]),
                    dbc.CardBody([
                        html.Div(id='target-context', className="mb-2"),
                        html.Div(id='target-table'),
                    ])
                ], className="dashboard-card h-100")
            ], width=12, lg=8, className="mb-3 mb-lg-0"),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Ringkasan Cepat", className="mb-0")),
                    dbc.CardBody([html.Div(id='quick-summary')])
                ], className="dashboard-card h-100 quick-summary-card")
            ], width=12, lg=4),
        ], className="mb-3 g-3"),

        dcc.Store(id='chat-store', data=[]),
    ])


# ============================================================
# PAGE 2: DETAIL EKOSISTEM ("Deep Dive")
# ============================================================
def ecosystem_detail_layout(default_community=None):
    if len(ecosystem_summary) == 0:
        return dbc.Alert("Belum ada ekosistem bisnis yang teridentifikasi.", color="warning")

    options = [{'label': row['label'], 'value': int(row['community_id'])} for _, row in ecosystem_summary.iterrows()]
    valid_ids = set(ecosystem_summary['community_id'])
    default_community = ECOSYSTEM_ID_MAP.get(default_community, default_community)
    default_value = default_community if default_community in valid_ids else int(ecosystem_summary.iloc[0]['community_id'])

    return html.Div([
        html.Div("ECOSYSTEM INTELLIGENCE", className="page-eyebrow"),
        dbc.Row([
            dbc.Col([
                html.H4("Detail Ekosistem Bisnis", className="page-title"),
                html.P("Ekosistem mana yang paling potensial, dan siapa anggotanya?",
                       className="page-description"),
                html.Label("Pilih Ekosistem", className="fw-bold"),
                dcc.Dropdown(id='ecosystem-selector', options=options, value=default_value,
                             clearable=False, className="mb-3"),
            ], width=12)
        ]),

        html.Div(id='ecosystem-summary-cards', className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Peta Jaringan Ekosistem", className="mb-0")),
                    dbc.CardBody([dcc.Graph(id='ecosystem-mini-graph', config=GRAPH_CONFIG)])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Rekomendasi Strategi Akuisisi", className="mb-0")),
                    dbc.CardBody([html.Div(id='ecosystem-narrative')])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=5),
        ], className="mb-3 g-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.H5("Anggota Ekosistem", className="mb-0"), width=12, md=6),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='ecosystem-member-limit',
                                    options=[{'label': 'Tampilkan 5', 'value': 5}, {'label': 'Tampilkan 10', 'value': 10},
                                             {'label': 'Tampilkan 15', 'value': 15}, {'label': 'Tampilkan Semua', 'value': -1}],
                                    value=5, clearable=False,
                                )
                            ], width=12, md=6),
                        ], className="g-2 align-items-center")
                    ]),
                    dbc.CardBody([html.Div(id='ecosystem-member-table')])
                ], className="shadow-sm border-0", style={"borderRadius": "12px"})
            ], width=12)
        ]),
    ])


# ============================================================
# PAGE 3: ANALYTICS & INSIGHT ("Gambaran Besar")
# ============================================================
def analytics_layout():
    return html.Div([
        html.Div("PORTFOLIO VIEW", className="page-eyebrow"),
        html.H4("Analytics & Insight", className="page-title"),
        html.P("Gambaran besar peta merchant dan area dengan peluang akuisisi terbesar.",
               className="page-description"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Pengaruh vs Peluang Akuisisi", className="mb-0")),
                    dbc.CardBody([
                        html.Small(
                            "Titik merah di kuadran kanan-atas = target ideal: pengaruh tinggi di jaringan "
                            "dan berbagi pelanggan dengan banyak merchant BNI, tapi belum menjadi nasabah.",
                            className="text-muted d-block mb-2"
                        ),
                        dcc.Graph(figure=build_scatter_figure(), config=GRAPH_CONFIG),
                    ])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=6, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Potensi MDR per Kota", className="mb-0")),
                    dbc.CardBody([dcc.Graph(figure=build_fee_per_kota_figure(), config=GRAPH_CONFIG)])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=6, className="mb-3"),
        ], className="g-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Komposisi Merchant per Kategori Bisnis", className="mb-0")),
                    dbc.CardBody([dcc.Graph(figure=build_kategori_composition_figure(), config=GRAPH_CONFIG)])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=6, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Peluang Akuisisi per Ekosistem", className="mb-0")),
                    dbc.CardBody([dcc.Graph(figure=build_ecosystem_opportunity_figure(), config=GRAPH_CONFIG)])
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "12px"})
            ], width=12, lg=6, className="mb-3"),
        ], className="g-3"),

        build_footer(),
    ])


# ============================================================
# APP LAYOUT (multi-page shell, navbar sticky)
# ============================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        dbc.Container(build_header(), fluid=True),
        className="app-header"
    ),
    dbc.Container([
        html.Div(id='page-content'),
    ], fluid=True, className="app-main")
], className="app-shell")


# ============================================================
# CALLBACKS — ROUTING & NAV
# ============================================================
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('url', 'search')
)
def render_page(pathname, search):
    if pathname == '/ekosistem':
        default_comm = None
        if search:
            qs = parse_qs(search.lstrip('?'))
            vals = qs.get('community')
            if vals:
                try:
                    default_comm = int(vals[0])
                except (TypeError, ValueError):
                    default_comm = None
        return ecosystem_detail_layout(default_comm)
    if pathname == '/analytics':
        return analytics_layout()
    return overview_layout()


@app.callback(
    Output('nav-links', 'children'),
    Input('url', 'pathname')
)
def update_nav(pathname):
    is_overview = pathname in (None, '/', '')
    is_eko = pathname == '/ekosistem'
    is_analytics = pathname == '/analytics'
    return html.Div([
        dcc.Link("Overview", href="/", className=f"nav-pill-link{' active' if is_overview else ''}"),
        dcc.Link("Detail Ekosistem", href="/ekosistem", className=f"nav-pill-link{' active' if is_eko else ''}"),
        dcc.Link("Analytics & Insight", href="/analytics", className=f"nav-pill-link{' active' if is_analytics else ''}"),
    ], className="d-flex flex-wrap")


# ============================================================
# CALLBACKS — OVERVIEW PAGE
# ============================================================
@app.callback(
    [Output('target-table', 'children'),
     Output('target-context', 'children'),
     Output('quick-summary', 'children')],
    [Input('filter-kategori', 'value'),
     Input('filter-kota', 'value'),
     Input('target-limit-selector', 'value')]
)
def update_target_table(kategori, kota, limit):
    df_all_targets = filter_non_bni_targets(kategori, kota)
    top = df_all_targets if limit == -1 else df_all_targets.head(limit)

    # Ringkasan cepat (semua merchant yang match filter, bukan hanya target)
    filtered_all = merchants_display.copy()
    if kategori and kategori != 'Semua':
        filtered_all = filtered_all[filtered_all['kategori'] == kategori]
    if kota and kota != 'Semua':
        filtered_all = filtered_all[filtered_all['kota'] == kota]
    n_filtered = len(filtered_all)
    n_target = len(filtered_all[filtered_all['is_nasabah_bni'] == 'Tidak'])
    filtered_priority_df = filtered_all[filtered_all['predicted_priority'] == 1]
    filtered_fee = filtered_priority_df.apply(get_est_fee, axis=1).sum() if len(filtered_priority_df) else 0

    summary_panel = quick_summary_panel(n_filtered, n_target, len(filtered_priority_df), filtered_fee)

    # Pesan kontekstual saat filter menghasilkan target lemah
    n_high = (df_all_targets['priority_probability'] >= 0.7).sum()
    n_medium = ((df_all_targets['priority_probability'] >= 0.4) & (df_all_targets['priority_probability'] < 0.7)).sum()

    context_msg, context_color = None, THEME['muted']
    if len(df_all_targets) == 0:
        context_msg = None
    elif n_high == 0 and n_medium == 0:
        context_msg = "Tidak ada target prioritas tinggi/sedang di filter ini. Menampilkan target terbaik yang tersedia:"
        context_color = THEME['muted']
    elif n_high == 0:
        context_msg = f"Tidak ada prioritas tinggi, namun ada {n_medium} target prioritas sedang:"
        context_color = "#B45309"
    else:
        context_msg = f"Ditemukan {n_high} target prioritas tinggi:"
        context_color = THEME['bni']

    context_el = html.Small(context_msg, style={"color": context_color, "fontWeight": "600"}) if context_msg else ""

    if len(top) == 0:
        return html.P("Tidak ada target yang sesuai filter.", className="text-muted"), "", summary_panel

    high_med_rows = []
    low_rows = []
    for i, (_, row) in enumerate(top.iterrows()):
        score = row.get('priority_probability', 0)
        badge_label, badge_color = priority_badge(score)
        degree = int(row.get('degree', 0))
        bni_ratio = row.get('connected_bni_ratio', 0)
        bni_partners = int(degree * bni_ratio)
        produk = get_produk_rekomendasi(row)
        nama = str(row.get('nama', 'Unknown'))[:28]

        if score >= 0.4:
            # Prioritas Tinggi/Sedang — tampil menonjol lengkap dengan estimasi fee
            high_med_rows.append(
                html.Div([
                    html.Div([
                        html.Strong(f"#{i+1} ", className="text-muted"),
                        html.Strong(nama),
                        dbc.Badge(f"{badge_label} ({score*100:.0f}%)", color=badge_color, className="ms-2"),
                    ]),
                    html.Small(f"{row.get('kategori','-')} · {row.get('kota','-')}",
                               className="text-muted d-block"),
                    html.Small(f"{degree} merchant dengan pelanggan serupa ({bni_partners} sudah BNI)",
                               className="text-muted d-block"),
                    html.Small(fee_display_text(row), className="text-success d-block fw-bold"),
                    html.Small(f"Rekomendasi produk: {produk}", className="text-primary d-block"),
                    html.Hr(className="my-1"),
                ], className="target-row target-row--priority")
            )
        else:
            # Prioritas Rendah — dibuat subtle, tanpa fee income, agar tidak mendominasi
            low_rows.append(
                html.Div([
                    html.Span(f"#{i+1} {nama} ", className="text-muted", style={"fontSize": "0.82rem"}),
                    dbc.Badge(badge_label, color=badge_color, className="ms-1", style={"fontSize": "0.65rem"}),
                    html.Span(f" · {row.get('kategori','-')} · {row.get('kota','-')}",
                              className="text-muted", style={"fontSize": "0.78rem"}),
                ], className="target-row target-row--low")
            )

    children = list(high_med_rows)
    if low_rows:
        children.append(html.Div("Prioritas Rendah", className="text-muted small fw-bold mt-2 mb-1"))
        children.extend(low_rows)

    return children, context_el, summary_panel


@app.callback(
    Output('network-graph', 'figure'),
    [Input('filter-kategori', 'value'),
     Input('filter-kota', 'value')]
)
def update_network_graph(kategori, kota):
    return build_network_figure(view='all', kategori=kategori, kota=kota, height=600)


@app.callback(
    Output('node-detail-panel', 'children'),
    Input('network-graph', 'clickData')
)
def on_node_click(clickData):
    if not clickData or 'points' not in clickData or not clickData['points']:
        return html.P("Klik salah satu merchant di graph untuk melihat detail.", className="text-muted mb-0")
    point = clickData['points'][0]
    merchant_id = point.get('customdata')
    if not merchant_id:
        return html.P("Klik salah satu merchant di graph untuk melihat detail.", className="text-muted mb-0")

    m_info = merchants_display[merchants_display['merchant_id'] == merchant_id]
    if len(m_info) == 0:
        return html.P("Detail tidak ditemukan.", className="text-muted mb-0")
    row = m_info.iloc[0]

    degree = int(row.get('degree', 0))
    bni_ratio = row.get('connected_bni_ratio', 0)
    bni_partners = int(degree * bni_ratio)
    produk = get_produk_rekomendasi(row)
    comm = int(row.get('community_id', -1))

    detail_children = [
        html.H6(f"Detail: {row.get('nama', merchant_id)}", className="mb-1"),
        html.Small(f"{row.get('kategori','-')} · {row.get('kota','-')} · Status BNI: {row.get('is_nasabah_bni','-')}",
                   className="text-muted d-block"),
        html.Small(f"{degree} merchant dengan pelanggan serupa ({bni_partners} sudah BNI)", className="d-block mt-1"),
        html.Small(f"{int(row.get('n_customers', 0))} pelanggan unik di merchant ini · "
                   f"rata-rata kesamaan pelanggan {row.get('avg_neighbor_jaccard', 0):.1%}",
                   className="text-muted d-block"),
        html.Small(fee_display_text(row), className="text-success d-block fw-bold"),
        html.Small(f"Rekomendasi produk: {produk}", className="text-primary d-block"),
    ]
    if comm != -1:
        detail_children.append(
            dcc.Link(
                dbc.Button("Lihat di Ekosistem →", size="sm", color="secondary", outline=True, className="mt-2"),
                href=f"/ekosistem?community={comm}"
            )
        )

    return dbc.Card([dbc.CardBody(detail_children)], className="node-detail-card")


@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-store', 'data'),
     Output('chat-input', 'value')],
    [Input('chat-send', 'n_clicks'),
     Input({'type': 'chat-example', 'index': ALL}, 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-store', 'data')],
    prevent_initial_call=True
)
def handle_chat(send_clicks, example_clicks, user_input, chat_history):
    # PENTING: API Claude (run_agent_simple) hanya dipanggil di sini, dari klik tombol
    # Kirim atau klik salah satu contoh pertanyaan — tidak pernah otomatis.
    trigger = callback_context.triggered[0] if callback_context.triggered else None
    if not trigger or trigger.get('value') is None:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = trigger['prop_id'].split('.')[0]
    if trigger_id != 'chat-send':
        try:
            comp_id = json.loads(trigger_id)
            idx = comp_id.get('index', 0)
            if 0 <= idx < len(CHAT_EXAMPLES):
                user_input = CHAT_EXAMPLES[idx]
        except (ValueError, TypeError):
            pass

    if not user_input or not user_input.strip():
        return dash.no_update, dash.no_update, dash.no_update

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    response = run_agent_simple(user_input)
    chat_history.append({"role": "assistant", "content": response})

    chat_display = []
    for msg in chat_history:
        if msg['role'] == 'user':
            chat_display.append(
                html.Div([
                    html.Strong("Anda: ", className="text-primary"),
                    html.Span(msg['content']),
                ], className="chat-message chat-message--user")
            )
        else:
            chat_display.append(
                html.Div([
                    html.Strong("💬 Asisten: ", className="text-success"),
                    html.Div(
                        dcc.Markdown(msg['content']),
                        className="mt-1"
                    ),
                ], className="chat-message chat-message--assistant")
            )

    return chat_display, chat_history, ""


# ============================================================
# CALLBACKS — DETAIL EKOSISTEM PAGE
# ============================================================
@app.callback(
    [Output('ecosystem-summary-cards', 'children'),
     Output('ecosystem-mini-graph', 'figure'),
     Output('ecosystem-narrative', 'children')],
    Input('ecosystem-selector', 'value')
)
def update_ecosystem_detail(community_id):
    if community_id is None:
        return "", go.Figure(), ""

    summary_row = ecosystem_summary[ecosystem_summary['community_id'] == community_id]
    if len(summary_row) == 0:
        return dbc.Alert("Data ekosistem tidak ditemukan.", color="warning"), go.Figure(), ""

    summary_row = summary_row.iloc[0]
    community_ids = summary_row['community_ids']
    members_df = merchants_display[merchants_display['community_id'].isin(community_ids)].copy()
    if len(members_df) == 0:
        return dbc.Alert("Data ekosistem tidak ditemukan.", color="warning"), go.Figure(), ""

    size = int(summary_row['size'])
    bni_pct = summary_row['bni_pct']
    avg_connections = summary_row['avg_connections']
    unique_customers = int(summary_row['unique_customers'])
    total_monthly_omzet = summary_row['total_monthly_omzet']
    potential_fee = summary_row['potential_fee']
    label = summary_row['label']

    cards = dbc.Row([
        dbc.Col(kpi_card("Jumlah Merchant", f"{size}", subtext="anggota ekosistem"), xs=6, md=4, lg=2),
        dbc.Col(kpi_card("Penetrasi BNI", f"{bni_pct:.0f}%", value_color="text-success",
                          subtext="sudah nasabah"), xs=6, md=4, lg=2),
        dbc.Col(kpi_card("Rata-rata Merchant Serupa", f"{avg_connections:.0f}", value_color="text-primary",
                          subtext="per anggota ekosistem"), xs=6, md=4, lg=2),
        dbc.Col(kpi_card("Jangkauan Pelanggan", f"{unique_customers:,}", value_color="text-primary",
                          subtext="pelanggan unik ekosistem"), xs=6, md=4, lg=2),
        dbc.Col(kpi_card("Omzet Bulanan", format_rupiah_short(total_monthly_omzet), value_color="text-primary",
                          subtext="estimasi seluruh merchant"), xs=6, md=4, lg=2),
        dbc.Col(kpi_card("Potensi MDR Tambahan", format_rupiah_short(potential_fee), value_color="text-warning",
                          subtext="jika non-BNI diakuisisi/bln"), xs=6, md=4, lg=2),
    ], className="g-3")

    mini_fig = build_network_figure(view=community_ids, height=380)

    non_bni_members = members_df[members_df['is_nasabah_bni'] == 'Tidak']
    hub_candidates = non_bni_members.nlargest(3, 'degree')

    narrative_parts = [
        f"Ekosistem {label} terdiri dari {size} merchant dengan penetrasi BNI {bni_pct:.0f}%."
    ]
    if len(hub_candidates) > 0:
        approx_reach = int((hub_candidates['degree'] * hub_candidates['connected_bni_ratio']).sum())
        nama_list = ', '.join(str(n) for n in hub_candidates['nama'].tolist())
        narrative_parts.append(
            f"Terdapat {len(hub_candidates)} merchant berpengaruh tinggi yang belum menjadi nasabah ({nama_list}) — "
            f"mengakuisisi mereka berpotensi memperluas jaringan ke sekitar {approx_reach} merchant nasabah existing."
        )
    else:
        narrative_parts.append("Tidak ada merchant berpengaruh tinggi non-BNI yang menonjol di ekosistem ini saat ini.")

    if potential_fee > 0:
        narrative_parts.append(
            f"Estimasi potensi MDR tambahan: {format_rupiah_short(potential_fee)}/bulan "
            f"jika seluruh merchant non-BNI di ekosistem ini diakuisisi."
        )

    recommendation_counts = non_bni_members['product_recommendation'].value_counts().to_dict()
    recommendation_summary = ", ".join(
        f"{label}: {count} merchant" for label, count in recommendation_counts.items()
    ) or "Tidak ada merchant non-BNI"
    narrative = html.Div([
        html.P(p, className="mb-2") for p in narrative_parts
    ] + [
        html.Small(
            f"Rekomendasi produk model untuk merchant non-BNI: {recommendation_summary}.",
            className="text-primary d-block mt-2"
        )
    ])

    return cards, mini_fig, narrative


@app.callback(
    Output('ecosystem-member-table', 'children'),
    [Input('ecosystem-selector', 'value'),
     Input('ecosystem-member-limit', 'value')]
)
def update_ecosystem_member_table(community_id, limit):
    if community_id is None:
        return ""
    summary_row = ecosystem_summary[ecosystem_summary['community_id'] == community_id]
    community_ids = summary_row.iloc[0]['community_ids'] if len(summary_row) > 0 else [community_id]
    members_df = merchants_display[merchants_display['community_id'].isin(community_ids)].copy()
    if len(members_df) == 0:
        return html.P("Tidak ada anggota.", className="text-muted")

    members_sorted = members_df.sort_values('priority_probability', ascending=False)
    if limit != -1:
        members_sorted = members_sorted.head(limit)

    table_header = html.Thead(html.Tr([
        html.Th("Nama"), html.Th("Kesamaan Pelanggan"), html.Th("Status BNI"),
        html.Th("Tingkat Pengaruh"), html.Th("Estimasi MDR/Bulan")
    ]))
    table_rows = []
    for _, row in members_sorted.iterrows():
        bni_val = row.get('is_nasabah_bni', 'Tidak')
        inf_label, inf_color = influence_badge(row.get('pagerank', 0))
        table_rows.append(html.Tr([
            html.Td(str(row.get('nama', '-'))[:32]),
            html.Td(f"Berbagi pelanggan dengan {int(row.get('degree', 0))} merchant"),
            html.Td(dbc.Badge(bni_val, color="success" if bni_val == 'Ya' else "danger")),
            html.Td(dbc.Badge(inf_label, color=inf_color)),
            html.Td(fee_display_text(row, prefix="")),
        ], className="target-row"))
    return dbc.Table([table_header, html.Tbody(table_rows)],
                      bordered=False, hover=True, responsive=True, size="sm", className="merchant-table mb-0")


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("Merchant Network Analytics Dashboard")
    print("="*60)
    print(f"Merchants: {total_merchants} | Nasabah BNI: {bni_count} ({penetration}%) | Ekosistem: {n_communities}")
    print(f"Product rec model: {product_model_metadata.get('model_name', '-')} | F1: {model_f1:.3f} | AUC: {model_auc:.3f}")
    print(f"Acquisition priority: Composite Score (bukan ML classifier), bobot: {priority_weights}")
    print(f"\nBuka di browser: http://localhost:8050")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=8050)
