# =============================================================================
# DCMA Schedule Analyzer — Módulos Analíticos Avançados
# Módulo 1: Análise de Rede PERT (Simulação Probabilística)
# Módulo 2: Análise Crítica de Recursos
# Módulo 3: Curva S de Progresso com Escala Dinâmica
#
# Todo o processamento é vetorizado em Pandas — sem iterrows — para suportar
# XERs de dezenas de milhares de linhas com boa performance.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# Reaproveita o layout escuro do app principal. Se este módulo for importado
# isoladamente, cai num fallback equivalente.
try:
    from app import DARK_LAYOUT, STATUS_COLOR
except ImportError:
    DARK_LAYOUT = dict(
        paper_bgcolor='#0a1520', plot_bgcolor='#0d1f2d',
        font=dict(color='#c8d8e8', family='Inter, sans-serif'),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    STATUS_COLOR = {'OK': '#00D4AA', 'ALERTA': '#F5A623', 'CRÍTICO': '#E5534B', 'N/A': '#555555'}

ACCENT = '#00D4AA'
ACCENT_2 = '#0099cc'
WARN = '#F5A623'
CRIT = '#E5534B'


# =============================================================================
# MÓDULO 1 — ANÁLISE DE REDE PERT (SIMULAÇÃO PROBABILÍSTICA)
# =============================================================================
def run_pert_analysis(tasks_df: pd.DataFrame) -> dict:
    """
    Simula os 3 tempos PERT (Otimista, Mais Provável, Pessimista) a partir da
    duração determinística do P6, calcula o tempo esperado, desvio padrão e
    variância do caminho crítico, e devolve tudo pronto para consulta de
    probabilidade e plotagem.

    Premissas de simulação (já que o P6 padrão só guarda 1 duração):
      O (otimista)   = M × 0.90   (-10%)
      M (mais provável) = target_drtn_hr_cnt (duração atual do P6)
      P (pessimista) = M × 1.20   (+20%)

    Parameters
    ----------
    tasks_df : DataFrame
        Deve conter as colunas 'target_drtn_hr_cnt' e 'total_float_hr_cnt'.
        Idealmente já filtrado para excluir marcos/LOE/concluídas
        (mesmo filtro usado no motor DCMA).

    Returns
    -------
    dict com:
        'df'              -> DataFrame com colunas O, M, P, TE, DP, Variancia, is_critical
        'project_te_hours' -> soma do TE ao longo do caminho crítico (horas)
        'project_sd_hours' -> desvio padrão do projeto (raiz da soma das variâncias)
        'critical_count'   -> nº de atividades no caminho crítico usadas na simulação
        'prob_fn'          -> função prob_fn(target_hours) -> probabilidade % (Z-score)
        'dist_fig'         -> Figura Plotly (curva de Gauss) pronta para st.plotly_chart
    """
    df = tasks_df.copy()

    # ── Duração base (M) ──
    M = pd.to_numeric(df.get('target_drtn_hr_cnt', 0), errors='coerce').fillna(0.0)
    df['M'] = M
    df['O'] = M * 0.90   # otimista: -10%
    df['P'] = M * 1.20   # pessimista: +20%

    # ── Cálculos estocásticos (vetorizados) ──
    df['TE'] = (df['O'] + 4 * df['M'] + df['P']) / 6.0
    df['DP'] = (df['P'] - df['O']) / 6.0
    df['Variancia'] = df['DP'] ** 2

    # ── Isola o caminho crítico (folga <= 0) ──
    tf = pd.to_numeric(df.get('total_float_hr_cnt', 0), errors='coerce').fillna(0.0)
    df['is_critical'] = tf <= 0

    critical = df[df['is_critical']]
    critical_count = len(critical)

    # Soma estocástica ao longo do caminho crítico:
    #   TE do projeto  = soma dos TE das atividades críticas
    #   Variância proj = soma das variâncias das atividades críticas (independência assumida)
    #   DP do projeto  = raiz quadrada da variância total
    project_te_hours = float(critical['TE'].sum()) if critical_count else 0.0
    project_var_hours = float(critical['Variancia'].sum()) if critical_count else 0.0
    project_sd_hours = float(np.sqrt(project_var_hours)) if project_var_hours > 0 else 0.0

    def prob_fn(target_hours: float) -> float:
        """
        Probabilidade (%) de concluir o projeto até `target_hours`,
        usando o Z-score sobre a distribuição normal acumulada do TE do projeto.
        """
        if project_sd_hours <= 0:
            # Sem variabilidade — resposta determinística
            return 100.0 if target_hours >= project_te_hours else 0.0
        z = (target_hours - project_te_hours) / project_sd_hours
        return float(norm.cdf(z) * 100.0)

    dist_fig = _pert_distribution_chart(project_te_hours, project_sd_hours)

    return {
        'df': df,
        'project_te_hours': project_te_hours,
        'project_sd_hours': project_sd_hours,
        'project_var_hours': project_var_hours,
        'critical_count': critical_count,
        'prob_fn': prob_fn,
        'dist_fig': dist_fig,
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converte '#RRGGBB' em 'rgba(r,g,b,a)' — Plotly recente rejeita hex de 8 dígitos (#RRGGBBAA)."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _pert_distribution_chart(mean_hours: float, sd_hours: float, target_hours: float | None = None) -> go.Figure:
    """Curva de Gauss da duração do projeto, com marcação opcional da Data Alvo."""
    if sd_hours <= 0:
        sd_hours = max(mean_hours * 0.05, 1.0)  # evita curva degenerada

    x = np.linspace(mean_hours - 4 * sd_hours, mean_hours + 4 * sd_hours, 400)
    y = norm.pdf(x, loc=mean_hours, scale=sd_hours)

    # Converte horas -> dias (8h/dia) só para leitura no eixo, mantendo cálculo em horas
    x_days = x / 8.0
    mean_days = mean_hours / 8.0
    sd_days = sd_hours / 8.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_days, y=y, mode='lines', name='Distribuição (TE)',
        line=dict(color=ACCENT, width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,170,0.12)',
    ))

    # Linha do TE médio
    fig.add_vline(x=mean_days, line_dash='dash', line_color='rgba(255,255,255,0.53)',
                   annotation_text=f'TE = {mean_days:.1f}d', annotation_font_color='#c8d8e8')

    # Faixas de confiança ±1σ / ±2σ
    for n_sigma, opacity in [(1, 0.10), (2, 0.05)]:
        fig.add_vrect(
            x0=mean_days - n_sigma * sd_days, x1=mean_days + n_sigma * sd_days,
            fillcolor=ACCENT_2, opacity=opacity, line_width=0,
        )

    if target_hours is not None:
        target_days = target_hours / 8.0
        prob = norm.cdf((target_hours - mean_hours) / sd_hours) * 100.0
        color = ACCENT if prob >= 70 else WARN if prob >= 40 else CRIT
        fig.add_vline(
            x=target_days, line_color=color, line_width=3,
            annotation_text=f'Data Alvo · {prob:.1f}%', annotation_font_color=color,
        )
        # área sob a curva até a data alvo (probabilidade de sucesso)
        x_fill = x_days[x_days <= target_days]
        y_fill = y[x_days <= target_days]
        if len(x_fill):
            fig.add_trace(go.Scatter(
                x=x_fill, y=y_fill, mode='lines', fill='tozeroy',
                fillcolor=_hex_to_rgba(color, 0.2), line=dict(width=0), showlegend=False, hoverinfo='skip',
            ))

    fig.update_layout(
        **DARK_LAYOUT, height=340,
        title=dict(text='Distribuição de Probabilidade — Conclusão do Projeto (PERT)', font=dict(size=14)),
        xaxis=dict(title='Duração do Caminho Crítico (dias)', gridcolor='#1e3a52', linecolor='#1e3a52'),
        yaxis=dict(title='Densidade', gridcolor='#1e3a52', linecolor='#1e3a52'),
        showlegend=False,
    )
    return fig


def pert_probability_chart(pert_result: dict, target_hours: float) -> go.Figure:
    """Wrapper público: gera a curva já com a Data Alvo marcada."""
    return _pert_distribution_chart(
        pert_result['project_te_hours'],
        pert_result['project_sd_hours'],
        target_hours=target_hours,
    )


# =============================================================================
# MÓDULO 2 — ANÁLISE CRÍTICA DE RECURSOS
# =============================================================================
def run_resource_analysis(
    tasks_df: pd.DataFrame,
    rsrc_df: pd.DataFrame,
    daily_limit_hours: float = 8.0,
) -> dict:
    """
    Cruza TASK x TASKRSRC para identificar superalocação de recursos e
    gargalos no caminho crítico. Tudo vetorizado — sem laços linha-a-linha.

    Parameters
    ----------
    tasks_df : DataFrame
        Atividades já filtradas (mesmo conjunto usado no motor DCMA).
        Precisa de: task_id, early_start_date, early_end_date,
        target_drtn_hr_cnt, total_float_hr_cnt.
    rsrc_df : DataFrame
        Tabela TASKRSRC bruta do XER. Precisa de: task_id, rsrc_id,
        target_qty (ou target_cost) — usamos target_qty como horas alocadas.
    daily_limit_hours : float
        Limite diário de alocação por recurso antes de considerar superalocação.

    Returns
    -------
    dict com:
        'daily_allocation' -> DataFrame (rsrc_id, date, allocated_hours, is_overallocated)
        'overallocated'    -> subset de daily_allocation onde is_overallocated
        'critical_bottlenecks' -> DataFrame de recursos alocados em atividades críticas
        'histogram_fig'    -> Figura Plotly (barras empilhadas) por recurso ao longo do tempo
    """
    if rsrc_df is None or rsrc_df.empty or tasks_df is None or tasks_df.empty:
        empty = pd.DataFrame()
        return {
            'daily_allocation': empty, 'overallocated': empty,
            'critical_bottlenecks': empty, 'histogram_fig': _empty_fig('Sem dados de recursos disponíveis'),
        }

    tasks = tasks_df.copy()
    rsrc = rsrc_df.copy()

    # ── normaliza tipos ──
    tasks['early_start_date'] = pd.to_datetime(tasks.get('early_start_date'), errors='coerce')
    tasks['early_end_date'] = pd.to_datetime(tasks.get('early_end_date'), errors='coerce')
    tasks['target_drtn_hr_cnt'] = pd.to_numeric(tasks.get('target_drtn_hr_cnt', 0), errors='coerce').fillna(0.0)
    tasks['total_float_hr_cnt'] = pd.to_numeric(tasks.get('total_float_hr_cnt', 0), errors='coerce').fillna(0.0)

    qty_col = 'target_qty' if 'target_qty' in rsrc.columns else None
    if qty_col is None:
        rsrc['target_qty'] = 8.0  # fallback: assume 1 unidade de recurso = 8h/dia
        qty_col = 'target_qty'
    rsrc[qty_col] = pd.to_numeric(rsrc[qty_col], errors='coerce').fillna(0.0)

    # ── junta recurso + atividade ──
    merged = rsrc.merge(
        tasks[['task_id', 'early_start_date', 'early_end_date', 'target_drtn_hr_cnt', 'total_float_hr_cnt']],
        on='task_id', how='inner',
    )
    merged = merged.dropna(subset=['early_start_date', 'early_end_date'])
    merged = merged[merged['early_end_date'] >= merged['early_start_date']]

    if merged.empty:
        empty = pd.DataFrame()
        return {
            'daily_allocation': empty, 'overallocated': empty,
            'critical_bottlenecks': empty,
            'histogram_fig': _empty_fig('Sem alocações com datas válidas para análise'),
        }

    # ── horas/dia alocadas por linha de recurso (qty total / nº dias) ──
    merged['n_days'] = (merged['early_end_date'] - merged['early_start_date']).dt.days.clip(lower=1) + 1
    merged['hours_per_day'] = merged[qty_col] / merged['n_days']

    # ── explode em uma linha por (recurso, dia) — vetorizado via date_range + repeat ──
    daily_alloc = _explode_date_range_vectorized(merged)

    # ── agrega por recurso + dia ──
    daily_summary = (
        daily_alloc.groupby(['rsrc_id', 'date'], as_index=False)['hours_per_day']
        .sum()
        .rename(columns={'hours_per_day': 'allocated_hours'})
    )
    daily_summary['is_overallocated'] = daily_summary['allocated_hours'] > daily_limit_hours

    overallocated = daily_summary[daily_summary['is_overallocated']].sort_values(
        'allocated_hours', ascending=False
    )

    # ── gargalos do caminho crítico (folga <= 0) ──
    critical_bottlenecks = merged[merged['total_float_hr_cnt'] <= 0].copy()
    if not critical_bottlenecks.empty:
        critical_bottlenecks = (
            critical_bottlenecks.groupby('rsrc_id', as_index=False)
            .agg(
                atividades_criticas=('task_id', 'nunique'),
                horas_alocadas_total=(qty_col, 'sum'),
            )
            .sort_values('atividades_criticas', ascending=False)
        )

    histogram_fig = _resource_histogram_chart(daily_summary, daily_limit_hours)

    return {
        'daily_allocation': daily_summary,
        'overallocated': overallocated,
        'critical_bottlenecks': critical_bottlenecks,
        'histogram_fig': histogram_fig,
    }


def _explode_date_range_vectorized(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Expande cada linha (recurso alocado num período) em uma linha por dia,
    de forma totalmente vetorizada: usa np.repeat para replicar as linhas-base
    e soma um offset de dias (np.arange por grupo, via cumsum de um reset) —
    sem laços Python sobre as datas, o que mantém custo ~O(total_dias) com
    overhead mínimo mesmo em XERs com dezenas de milhares de atividades.
    """
    n_days = merged['n_days'].values.astype(int)
    starts = merged['early_start_date'].values.astype('datetime64[D]')

    total_rows = int(n_days.sum())
    if total_rows == 0:
        return merged.iloc[0:0][['rsrc_id', 'date', 'hours_per_day', 'task_id']].assign(date=pd.NaT)

    # índice de origem replicado (linha base -> N dias)
    repeat_idx = np.repeat(np.arange(len(merged)), n_days)

    # offset de dia dentro de cada grupo: 0,1,2,...,n-1 para cada linha, concatenado
    # construído via subtração de um arange global pelo "início do grupo" repetido
    group_starts = np.concatenate(([0], np.cumsum(n_days)[:-1])) if len(n_days) else np.array([])
    day_offsets = np.arange(total_rows) - np.repeat(group_starts, n_days)

    exploded = merged.iloc[repeat_idx].reset_index(drop=True)
    exploded['date'] = starts[repeat_idx] + day_offsets.astype('timedelta64[D]')
    return exploded[['rsrc_id', 'date', 'hours_per_day', 'task_id']]


def _resource_histogram_chart(daily_summary: pd.DataFrame, daily_limit_hours: float) -> go.Figure:
    """Histograma de recursos (barras empilhadas) ao longo do tempo."""
    if daily_summary.empty:
        return _empty_fig('Sem dados de alocação diária')

    pivot = daily_summary.pivot_table(
        index='date', columns='rsrc_id', values='allocated_hours', aggfunc='sum', fill_value=0
    ).sort_index()

    # limita a top N recursos por carga total, para não poluir o gráfico
    top_n = 12
    totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_resources = totals.head(top_n).index
    pivot_top = pivot[top_resources]

    fig = go.Figure()
    palette = ['#00D4AA', '#0099cc', '#F5A623', '#E5534B', '#8e6fce', '#4dabf7',
               '#ffa94d', '#69db7c', '#ff8787', '#748ffc', '#e599f7', '#63e6be']

    for i, rsrc_id in enumerate(pivot_top.columns):
        fig.add_trace(go.Bar(
            x=pivot_top.index, y=pivot_top[rsrc_id],
            name=str(rsrc_id), marker_color=palette[i % len(palette)],
        ))

    # linha de limite diário
    fig.add_hline(
        y=daily_limit_hours, line_dash='dash', line_color='#E5534B',
        annotation_text=f'Limite diário ({daily_limit_hours:.0f}h)', annotation_font_color='#E5534B',
    )

    fig.update_layout(
        **DARK_LAYOUT, height=380, barmode='stack',
        title=dict(text='Histograma de Recursos — Alocação Diária', font=dict(size=14)),
        xaxis=dict(title='Data', gridcolor='#1e3a52', linecolor='#1e3a52'),
        yaxis=dict(title='Horas Alocadas/Dia', gridcolor='#1e3a52', linecolor='#1e3a52'),
        legend=dict(orientation='h', y=1.15, bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    return fig


# =============================================================================
# MÓDULO 3 — CURVA S DE PROGRESSO COM ESCALA DINÂMICA
# =============================================================================
GROUPING_OPTIONS = {
    'Diário/Semanal': 'D',
    'Semanal/Mensal': 'W',
    'Mensal/Anual': 'ME',   # 'M' foi descontinuado no pandas 2.2+; 'ME' = Month End
}


def generate_s_curve(
    tasks_df: pd.DataFrame,
    weight_col: str = 'target_drtn_hr_cnt',
    freq: str = 'W',
) -> dict:
    """
    Distribui linearmente o peso (duração, custo ou %) de cada atividade entre
    early_start_date e early_end_date, agrega no período escolhido e devolve
    a Curva S cumulativa.

    Parameters
    ----------
    tasks_df : DataFrame
        Precisa de: early_start_date, early_end_date, e a coluna de peso
        (default 'target_drtn_hr_cnt'; pode ser custo, ex. 'target_cost').
    weight_col : str
        Coluna numérica a distribuir (duração, custo, etc).
    freq : str
        Frequência de agrupamento do Pandas: 'D' (diário), 'W' (semanal),
        'M' (mensal). Use GROUPING_OPTIONS para mapear rótulos amigáveis.

    Returns
    -------
    dict com:
        'periodic_df'  -> DataFrame (period, period_value, cumulative_value, cumulative_pct)
        'fig'          -> Figura Plotly (linha cumulativa + barras periódicas)
    """
    df = tasks_df.copy()
    if freq == 'M':  # compat: aceita o alias antigo e converte para o atual
        freq = 'ME'
    df['early_start_date'] = pd.to_datetime(df.get('early_start_date'), errors='coerce')
    df['early_end_date'] = pd.to_datetime(df.get('early_end_date'), errors='coerce')
    df[weight_col] = pd.to_numeric(df.get(weight_col, 0), errors='coerce').fillna(0.0)

    df = df.dropna(subset=['early_start_date', 'early_end_date'])
    df = df[df['early_end_date'] >= df['early_start_date']]
    df = df[df[weight_col] > 0]

    if df.empty:
        return {
            'periodic_df': pd.DataFrame(),
            'fig': _empty_fig('Sem atividades com datas/peso válidos para a Curva S'),
        }

    df['n_days'] = (df['early_end_date'] - df['early_start_date']).dt.days.clip(lower=0) + 1
    df['value_per_day'] = df[weight_col] / df['n_days']

    # ── expansão vetorizada diária (np.repeat + offset de dias, sem laço por linha) ──
    n_days = df['n_days'].values.astype(int)
    starts = df['early_start_date'].values.astype('datetime64[D]')
    total_rows = int(n_days.sum())

    if total_rows == 0:
        return {
            'periodic_df': pd.DataFrame(),
            'fig': _empty_fig('Sem atividades com datas/peso válidos para a Curva S'),
        }

    repeat_idx = np.repeat(np.arange(len(df)), n_days)
    group_starts = np.concatenate(([0], np.cumsum(n_days)[:-1]))
    day_offsets = np.arange(total_rows) - np.repeat(group_starts, n_days)

    exploded = df.iloc[repeat_idx].reset_index(drop=True)
    exploded['date'] = starts[repeat_idx] + day_offsets.astype('timedelta64[D]')

    # ── agrega no período escolhido (D / W / M) ──
    exploded = exploded.set_index('date')
    periodic = exploded['value_per_day'].resample(freq).sum().reset_index()
    periodic.columns = ['period', 'period_value']
    periodic = periodic.sort_values('period')

    periodic['cumulative_value'] = periodic['period_value'].cumsum()
    total = periodic['period_value'].sum()
    periodic['cumulative_pct'] = (periodic['cumulative_value'] / total * 100.0) if total > 0 else 0.0

    fig = _s_curve_chart(periodic, freq, weight_col)

    return {'periodic_df': periodic, 'fig': fig}


def _s_curve_chart(periodic: pd.DataFrame, freq: str, weight_col: str) -> go.Figure:
    freq_label = {'D': 'Diário', 'W': 'Semanal', 'M': 'Mensal', 'ME': 'Mensal'}.get(freq, freq)

    fig = go.Figure()

    # barras: esforço periódico
    fig.add_trace(go.Bar(
        x=periodic['period'], y=periodic['period_value'],
        name=f'Esforço Periódico ({freq_label})',
        marker_color='rgba(0,153,204,0.45)',
        yaxis='y1',
    ))

    # linha: curva S cumulativa (%)
    fig.add_trace(go.Scatter(
        x=periodic['period'], y=periodic['cumulative_pct'],
        name='Avanço Cumulativo (%)', mode='lines+markers',
        line=dict(color=ACCENT, width=3),
        marker=dict(size=5, color=ACCENT),
        yaxis='y2',
    ))

    fig.update_layout(
        **DARK_LAYOUT, height=380,
        title=dict(text=f'Curva S de Progresso — Base: {weight_col}', font=dict(size=14)),
        xaxis=dict(title='Período', gridcolor='#1e3a52', linecolor='#1e3a52'),
        yaxis=dict(title='Esforço por Período', gridcolor='#1e3a52', linecolor='#1e3a52', side='left'),
        yaxis2=dict(title='Avanço Cumulativo (%)', overlaying='y', side='right',
                     range=[0, 105], gridcolor='rgba(0,0,0,0)'),
        legend=dict(orientation='h', y=1.15, bgcolor='rgba(0,0,0,0)'),
        barmode='overlay',
    )
    return fig


# =============================================================================
# HELPERS
# =============================================================================
def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref='paper', yref='paper', x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color='#6699bb'),
    )
    fig.update_layout(**DARK_LAYOUT, height=300,
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
