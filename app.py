# =============================================================================
# DCMA 14-Point Schedule Quality Analyzer
# Primavera P6 XER · Streamlit App
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO, StringIO
import re
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCMA Schedule Analyzer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a1520;
    color: #c8d8e8;
  }
  .main { background-color: #0a1520; }
  .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

  /* Header */
  .dcma-header {
    background: linear-gradient(135deg, #0d1f2d 0%, #0f2a40 100%);
    border: 1px solid #1e3a52;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .dcma-title { font-size: 24px; font-weight: 700; color: #e8f4ff; margin: 0; }
  .dcma-subtitle { font-size: 12px; color: #6699bb; font-family: 'JetBrains Mono', monospace; margin: 2px 0 0; }

  /* Cards */
  .metric-card {
    background: #0d1f2d;
    border: 1px solid #1e3a52;
    border-radius: 12px;
    padding: 18px 20px;
  }
  .status-ok    { border-color: #00D4AA55; background: #00D4AA0a; }
  .status-alert { border-color: #F5A62355; background: #F5A6230a; }
  .status-crit  { border-color: #E5534B55; background: #E5534B0a; }

  /* Badges */
  .badge-ok   { background:#00D4AA22; color:#00D4AA; border:1px solid #00D4AA55;
                padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700;
                font-family:'JetBrains Mono',monospace; }
  .badge-alerta { background:#F5A62322; color:#F5A623; border:1px solid #F5A62355;
                  padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700;
                  font-family:'JetBrains Mono',monospace; }
  .badge-crit { background:#E5534B22; color:#E5534B; border:1px solid #E5534B55;
                padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700;
                font-family:'JetBrains Mono',monospace; }
  .badge-na   { background:#55555522; color:#888; border:1px solid #55555555;
                padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700;
                font-family:'JetBrains Mono',monospace; }

  /* Streamlit overrides */
  .stFileUploader > div { border: 2px dashed #1e3a52 !important; border-radius: 12px !important;
                          background: #0d1f2d !important; }
  .stFileUploader > div:hover { border-color: #00D4AA !important; }
  .stButton > button {
    background: linear-gradient(135deg, #00D4AA, #0099cc) !important;
    color: #000 !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 28px !important; font-size: 15px !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
  div[data-testid="stTabs"] button { color: #6699bb !important; font-weight: 600; }
  div[data-testid="stTabs"] button[aria-selected="true"] { color: #00D4AA !important; border-bottom-color: #00D4AA !important; }
  .stDataFrame { border-radius: 10px; overflow: hidden; }
  .stExpander { border: 1px solid #1e3a52 !important; border-radius: 10px !important; background: #0d1f2d !important; }
  hr { border-color: #1e3a52; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# XER PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_xer(content: str) -> tuple[dict, list]:
    """
    Parse a Primavera P6 XER file into a dict of DataFrames.
    Returns (tables_dict, log_messages).
    """
    tables = {}
    log = []
    current_table = None
    headers = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip('\r')
        if not line.strip() or line.startswith('%E'):
            continue
        if line.startswith('%T'):
            current_table = line[2:].strip()
            tables[current_table] = []
            headers = []
        elif line.startswith('%F'):
            headers = line[2:].strip().split('\t')
        elif line.startswith('%R'):
            values = line[2:].strip().split('\t')
            record = {headers[i]: values[i] if i < len(values) else '' for i in range(len(headers))}
            if current_table:
                tables[current_table].append(record)

    # Convert to DataFrames
    dfs = {}
    for name, rows in tables.items():
        if rows:
            dfs[name] = pd.DataFrame(rows)
            log.append(f"✓ Tabela **{name}**: {len(rows)} registros")
        else:
            dfs[name] = pd.DataFrame()

    missing = [t for t in ['TASK', 'TASKPRED', 'PROJECT'] if t not in dfs]
    for m in missing:
        log.append(f"⚠ Tabela **{m}** não encontrada no arquivo")

    return dfs, log


# ─────────────────────────────────────────────────────────────────────────────
# DCMA ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(series, col):
    if col not in series.columns:
        return pd.Series([0.0] * len(series))
    return pd.to_numeric(series[col], errors='coerce').fillna(0.0)

def safe_dt(series, col):
    if col not in series.columns:
        return pd.Series([pd.NaT] * len(series))
    return pd.to_datetime(series[col], errors='coerce')


def run_dcma(dfs: dict) -> dict | None:
    """Run all 14 DCMA checks. Returns result dict or None if no tasks."""

    # ── raw tables ──
    task_raw = dfs.get('TASK', pd.DataFrame())
    pred_raw = dfs.get('TASKPRED', pd.DataFrame())
    rsrc_raw = dfs.get('TASKRSRC', pd.DataFrame())
    proj_raw = dfs.get('PROJECT', pd.DataFrame())

    if task_raw.empty:
        return None

    # ── filter tasks: exclude milestones, LOE, completed ──
    tasks = task_raw.copy()
    exclude_types  = ['TT_Mile', 'TT_FinMile', 'TT_LOE', 'TT_WBS']
    exclude_status = ['TK_Complete']
    if 'task_type' in tasks.columns:
        tasks = tasks[~tasks['task_type'].isin(exclude_types)]
    if 'status_code' in tasks.columns:
        tasks = tasks[~tasks['status_code'].isin(exclude_status)]
    tasks = tasks.reset_index(drop=True)

    preds = pred_raw.copy() if not pred_raw.empty else pd.DataFrame()
    rsrc  = rsrc_raw.copy() if not rsrc_raw.empty else pd.DataFrame()

    total_tasks = len(tasks)
    total_rels  = len(preds)

    if total_tasks == 0:
        return None

    task_ids = set(tasks['task_id']) if 'task_id' in tasks.columns else set()

    # only keep rels that reference active tasks
    if not preds.empty and 'task_id' in preds.columns:
        preds_active = preds[preds['task_id'].isin(task_ids) | preds.get('pred_task_id', pd.Series()).isin(task_ids)]
    else:
        preds_active = preds

    has_pred = set(preds_active['task_id'])             if 'task_id' in preds_active.columns      else set()
    has_succ = set(preds_active['pred_task_id'])        if 'pred_task_id' in preds_active.columns  else set()
    has_rsrc = set(rsrc['task_id'])                     if 'task_id' in rsrc.columns               else set()

    lag_vals = safe_float(preds_active, 'lag_hr_cnt')

    results = []

    # ── 1. Missing predecessors ──
    miss_pred = tasks[~tasks['task_id'].isin(has_pred)] if 'task_id' in tasks.columns else pd.DataFrame()
    pct1 = len(miss_pred) / total_tasks * 100
    results.append({
        'id': 1, 'name': 'Sem Predecessoras', 'short': 'S/Pred',
        'value': pct1, 'goal': 5, 'unit': '%',
        'affected': miss_pred,
        'formula': 'Atividades sem predecessoras / Total × 100',
        'rec': 'Verifique se estas atividades realmente não dependem de nenhuma outra. Adicione predecessoras lógicas onde aplicável.',
        'is_rel': False,
    })

    # ── 2. Missing successors ──
    miss_succ = tasks[~tasks['task_id'].isin(has_succ)] if 'task_id' in tasks.columns else pd.DataFrame()
    pct2 = len(miss_succ) / total_tasks * 100
    results.append({
        'id': 2, 'name': 'Sem Sucessoras', 'short': 'S/Suc',
        'value': pct2, 'goal': 5, 'unit': '%',
        'affected': miss_succ,
        'formula': 'Atividades sem sucessoras / Total × 100',
        'rec': 'Atividades sem sucessoras indicam pontas abertas. Conecte-as ao término do projeto.',
        'is_rel': False,
    })

    # ── 3. Constraints ──
    cstr_hard = ['CS_MSO', 'CS_MSOB', 'CS_MSF', 'CS_MFOB', 'CS_MANDSTART', 'CS_MANDFINISH']
    if 'cstr_type' in tasks.columns:
        constrained = tasks[tasks['cstr_type'].isin(cstr_hard)]
    else:
        constrained = pd.DataFrame()
    pct3 = len(constrained) / total_tasks * 100
    results.append({
        'id': 3, 'name': 'Com Restrições', 'short': 'Restr',
        'value': pct3, 'goal': 5, 'unit': '%',
        'affected': constrained,
        'formula': 'Atividades com restrições impostas / Total × 100',
        'rec': 'Restrições de datas comprometem a integridade lógica. Substitua por dependências sempre que possível.',
        'is_rel': False,
    })

    # ── 4. Lags (positive) ──
    if not preds_active.empty:
        lag_rels = preds_active[lag_vals > 0]
        pct4 = len(lag_rels) / max(total_rels, 1) * 100
    else:
        lag_rels = pd.DataFrame(); pct4 = 0
    results.append({
        'id': 4, 'name': 'Com Lag Positivo', 'short': 'Lag+',
        'value': pct4, 'goal': 5, 'unit': '% rel',
        'affected': lag_rels,
        'formula': 'Relações com lag > 0 / Total de relações × 100',
        'rec': 'Lags ocultam lógica real. Crie atividades explícitas para representar períodos de espera.',
        'is_rel': True,
    })

    # ── 5. Non-FS relationships ──
    if not preds_active.empty and 'pred_type' in preds_active.columns:
        non_fs = preds_active[preds_active['pred_type'] != 'PR_FS']
        pct5 = len(non_fs) / max(total_rels, 1) * 100
    else:
        non_fs = pd.DataFrame(); pct5 = 0
    results.append({
        'id': 5, 'name': 'Relações não-FS', 'short': 'N-FS',
        'value': pct5, 'goal': 10, 'unit': '% rel',
        'affected': non_fs,
        'formula': 'Relações SS/FF/SF / Total de relações × 100',
        'rec': 'Relações não-FS podem mascarar problemas de lógica. Revise a necessidade de cada uma.',
        'is_rel': True,
    })

    # ── 6. Leads (negative lag) ──
    if not preds_active.empty:
        lead_rels = preds_active[lag_vals < 0]
        pct6 = len(lead_rels) / max(total_rels, 1) * 100
    else:
        lead_rels = pd.DataFrame(); pct6 = 0
    results.append({
        'id': 6, 'name': 'Com Lead (lag negativo)', 'short': 'Lead',
        'value': pct6, 'goal': 0, 'unit': '% rel',
        'affected': lead_rels,
        'formula': 'Relações com lag < 0 / Total de relações × 100',
        'rec': 'Leads (lags negativos) são proibidos pela DCMA. Elimine-os e substitua pela lógica correta.',
        'is_rel': True,
    })

    # ── 7. Long durations (> 44 days = 352 hours @ 8h/day) ──
    dur = safe_float(tasks, 'target_drtn_hr_cnt')
    long_dur = tasks[dur > 352]
    pct7 = len(long_dur) / total_tasks * 100
    results.append({
        'id': 7, 'name': 'Duração > 44 dias', 'short': 'Dur+',
        'value': pct7, 'goal': 5, 'unit': '%',
        'affected': long_dur,
        'formula': 'Atividades com duração > 352 h (44 dias úteis) / Total × 100',
        'rec': 'Atividades longas dificultam o controle. Considere decompor em pacotes menores de trabalho.',
        'is_rel': False,
    })

    # ── 8. High float (> 44 days) ──
    tf = safe_float(tasks, 'total_float_hr_cnt')
    high_float = tasks[tf > 352]
    pct8 = len(high_float) / total_tasks * 100
    results.append({
        'id': 8, 'name': 'Alta Folga (> 44 dias)', 'short': 'F.Alta',
        'value': pct8, 'goal': 5, 'unit': '%',
        'affected': high_float,
        'formula': 'Atividades com folga total > 352 h / Total × 100',
        'rec': 'Alta folga pode indicar lógica ausente ou sequenciamento incorreto. Revise as dependências.',
        'is_rel': False,
    })

    # ── 9. Negative float ──
    neg_float = tasks[tf < 0]
    pct9 = len(neg_float) / total_tasks * 100
    results.append({
        'id': 9, 'name': 'Folga Negativa', 'short': 'F.Neg',
        'value': pct9, 'goal': 0, 'unit': '%',
        'affected': neg_float,
        'formula': 'Atividades com folga total < 0 / Total × 100',
        'rec': 'Folga negativa indica cronograma em atraso ou restrições incompatíveis. Ação imediata necessária.',
        'is_rel': False,
    })

    # ── 10. Date inconsistencies ──
    es = safe_dt(tasks, 'early_start_date')
    ef = safe_dt(tasks, 'early_end_date')
    # Also check actual dates if present
    act_s = safe_dt(tasks, 'act_start_date')
    act_f = safe_dt(tasks, 'act_end_date')
    start_dt = act_s.where(act_s.notna(), es)
    end_dt   = act_f.where(act_f.notna(), ef)
    date_inc_mask = (start_dt.notna() & end_dt.notna() & (end_dt < start_dt))
    date_inc = tasks[date_inc_mask]
    pct10 = len(date_inc) / total_tasks * 100
    results.append({
        'id': 10, 'name': 'Inconsistências de Data', 'short': 'Inc.Dt',
        'value': pct10, 'goal': 0, 'unit': '%',
        'affected': date_inc,
        'formula': 'Atividades com término < início / Total × 100',
        'rec': 'Inconsistências de data indicam erros graves de entrada de dados. Corrija imediatamente.',
        'is_rel': False,
    })

    # ── 11. Missing resources ──
    if not rsrc.empty:
        miss_rsrc = tasks[~tasks['task_id'].isin(has_rsrc)] if 'task_id' in tasks.columns else pd.DataFrame()
        pct11 = len(miss_rsrc) / total_tasks * 100
        na11 = False
    else:
        miss_rsrc = pd.DataFrame(); pct11 = 0; na11 = True
    results.append({
        'id': 11, 'name': 'Sem Recursos', 'short': 'S/Rec',
        'value': pct11, 'goal': 10, 'unit': '%',
        'affected': miss_rsrc,
        'formula': 'Atividades sem recursos / Total × 100 (quando há recursos no projeto)',
        'rec': 'Atividades sem recursos não permitem análise de custo e carga de trabalho adequada.',
        'is_rel': False,
        'na': na11,
    })

    # ── 12. Open ends (missing pred OR succ) ──
    open_ends = tasks[
        (~tasks['task_id'].isin(has_pred)) | (~tasks['task_id'].isin(has_succ))
    ] if 'task_id' in tasks.columns else pd.DataFrame()
    pct12 = len(open_ends) / total_tasks * 100
    results.append({
        'id': 12, 'name': 'Extremidades Abertas', 'short': 'Ext.Ab',
        'value': pct12, 'goal': 5, 'unit': '%',
        'affected': open_ends,
        'formula': 'Atividades sem pred. OU sem suc. / Total × 100',
        'rec': 'Toda atividade deve ter ao menos uma predecessora e uma sucessora (exceto marcos de início/fim).',
        'is_rel': False,
    })

    # ── 13. Critical path breaks ──
    crit_tasks = tasks[tf <= 0]
    crit_ids = set(crit_tasks['task_id']) if 'task_id' in crit_tasks.columns else set()
    breaks = 0
    if not preds_active.empty and 'task_id' in preds_active.columns and len(crit_ids) > 0:
        crit_preds = preds_active[
            preds_active['task_id'].isin(crit_ids) & preds_active['pred_task_id'].isin(crit_ids)
        ]
        connected_as_succ = set(crit_preds['task_id'])
        # Count critical tasks that have no critical predecessor (except the first)
        isolated = crit_ids - connected_as_succ
        breaks = max(0, len(isolated) - 1)  # allow 1 start point
    pct13 = (breaks / max(len(crit_ids), 1)) * 100
    results.append({
        'id': 13, 'name': 'Desvio do Caminho Crítico', 'short': 'CP',
        'value': pct13, 'goal': 5, 'unit': '%',
        'affected': pd.DataFrame(),
        'formula': 'Quebras na cadeia crítica / Total de atividades críticas × 100',
        'rec': 'O caminho crítico deve ser contínuo do início ao fim do projeto. Identifique e corrija quebras.',
        'is_rel': False,
    })

    # ── 14. Logic density ──
    density = total_rels / total_tasks if total_tasks > 0 else 0
    results.append({
        'id': 14, 'name': 'Densidade Lógica', 'short': 'Dens',
        'value': density, 'goal': 2.0, 'unit': 'rel/at',
        'goal_min': 1.5, 'goal_max': 3.0,
        'affected': pd.DataFrame(),
        'formula': 'Total de relações / Total de atividades',
        'rec': (
            'Baixa densidade lógica — cronograma pode ter lógica insuficiente.' if density < 1.5
            else 'Alta densidade — possíveis relações redundantes.' if density > 3.0
            else 'Densidade adequada.'
        ),
        'is_rel': False,
    })

    # ── score ──
    score_total, score_count = 0, 0
    for r in results:
        if r.get('na'):
            continue
        score_count += 1
        if r['id'] == 14:
            gmin, gmax = r.get('goal_min', 1.5), r.get('goal_max', 3.0)
            if gmin <= r['value'] <= gmax:
                s = 100
            elif r['value'] < gmin:
                s = max(0, (r['value'] / gmin) * 100)
            else:
                s = max(0, 100 - ((r['value'] - gmax) / gmax) * 50)
        else:
            ratio = r['value'] / max(r['goal'], 0.01)
            s = max(0, 100 - ratio * 100)
        score_total += s

    final_score = score_total / score_count if score_count > 0 else 0

    proj_name = ''
    if not proj_raw.empty and 'proj_short_name' in proj_raw.columns:
        proj_name = proj_raw.iloc[0].get('proj_short_name', '')

    return {
        'indicators': results,
        'total_tasks': total_tasks,
        'total_rels': total_rels,
        'critical_count': len(crit_tasks),
        'final_score': final_score,
        'project_name': proj_name or 'Projeto',
        'tasks_df': tasks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATUS & COLORS
# ─────────────────────────────────────────────────────────────────────────────
def get_status(ind: dict) -> str:
    if ind.get('na'):
        return 'N/A'
    if ind['id'] == 14:
        gmin, gmax = ind.get('goal_min', 1.5), ind.get('goal_max', 3.0)
        if gmin <= ind['value'] <= gmax:
            return 'OK'
        if ind['value'] < 1.0 or ind['value'] > 4.0:
            return 'CRÍTICO'
        return 'ALERTA'
    ratio = ind['value'] / max(ind['goal'], 0.01)
    if ratio <= 0.5:
        return 'OK'
    if ratio <= 1.0:
        return 'ALERTA'
    return 'CRÍTICO'

STATUS_COLOR  = {'OK': '#00D4AA', 'ALERTA': '#F5A623', 'CRÍTICO': '#E5534B', 'N/A': '#555555'}
STATUS_EMOJI  = {'OK': '✅', 'ALERTA': '⚠️', 'CRÍTICO': '🚨', 'N/A': '➖'}
STATUS_CLASS  = {'OK': 'badge-ok', 'ALERTA': 'badge-alerta', 'CRÍTICO': 'badge-crit', 'N/A': 'badge-na'}


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='#0a1520', plot_bgcolor='#0d1f2d',
    font=dict(color='#c8d8e8', family='Inter, sans-serif'),
    margin=dict(t=30, b=10, l=10, r=10),
)

def score_gauge(score: float) -> go.Figure:
    color = '#00D4AA' if score >= 75 else '#F5A623' if score >= 50 else '#E5534B'
    label = 'BOM' if score >= 75 else 'REGULAR' if score >= 50 else 'CRÍTICO'
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(score, 1),
        number={'suffix': '/100', 'font': {'size': 36, 'color': color, 'family': 'JetBrains Mono'}},
        title={'text': f'Score de Qualidade<br><span style="font-size:14px;color:{color}">{label}</span>',
               'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#1e3a52', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1e3a52',
            'bordercolor': '#0d1f2d',
            'steps': [
                {'range': [0, 50], 'color': '#E5534B22'},
                {'range': [50, 75], 'color': '#F5A62322'},
                {'range': [75, 100], 'color': '#00D4AA22'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': score},
        }
    ))
    fig.update_layout(**DARK_LAYOUT, height=260)
    return fig


def radar_chart(indicators: list) -> go.Figure:
    subset = indicators[:8]
    status_score = {'OK': 100, 'ALERTA': 50, 'CRÍTICO': 10, 'N/A': 0}
    cats = [i['short'] for i in subset]
    vals = [status_score[get_status(i)] for i in subset]
    cats.append(cats[0]); vals.append(vals[0])  # close polygon

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(0,212,170,0.15)',
        line=dict(color='#00D4AA', width=2),
        marker=dict(color='#00D4AA', size=6),
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=280,
        polar=dict(
            bgcolor='#0d1f2d',
            radialaxis=dict(visible=True, range=[0, 100], tickcolor='#1e3a52',
                            gridcolor='#1e3a52', linecolor='#1e3a52'),
            angularaxis=dict(tickcolor='#1e3a52', gridcolor='#1e3a52', linecolor='#1e3a52'),
        ),
        showlegend=False,
    )
    return fig


def bar_chart(indicators: list) -> go.Figure:
    names  = [i['short'] for i in indicators]
    values = [round(i['value'], 2) for i in indicators]
    goals  = [i['goal'] for i in indicators]
    colors = [STATUS_COLOR[get_status(i)] for i in indicators]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Valor', x=names, y=values,
        marker_color=colors, marker_line_color='#0a1520', marker_line_width=1,
        text=[f"{v:.1f}" for v in values], textposition='outside',
        textfont=dict(size=10, color='#c8d8e8'),
    ))
    fig.add_trace(go.Scatter(
        name='Meta', x=names, y=goals,
        mode='markers', marker=dict(symbol='line-ew', size=14,
                                    color='#ffffff66', line=dict(width=2, color='#ffffff88')),
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=300,
        xaxis=dict(gridcolor='#1e3a52', linecolor='#1e3a52'),
        yaxis=dict(gridcolor='#1e3a52', linecolor='#1e3a52'),
        legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
        barmode='group',
    )
    return fig


def pie_chart(indicators: list) -> go.Figure:
    from collections import Counter
    counts = Counter(get_status(i) for i in indicators)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [STATUS_COLOR[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#0a1520', width=2)),
        textfont=dict(color='#e8f4ff', size=13),
        hole=0.45,
    ))
    fig.update_layout(**DARK_LAYOUT, height=280, showlegend=True,
                      legend=dict(orientation='h', y=-0.1, bgcolor='rgba(0,0,0,0)'))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_excel(result: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        # Summary sheet
        summary_rows = []
        for ind in result['indicators']:
            s = get_status(ind)
            summary_rows.append({
                'ID': ind['id'],
                'Indicador': ind['name'],
                'Valor': round(ind['value'], 2),
                'Unidade': ind['unit'],
                'Meta': ind['goal'],
                'Status': s,
                'N° Afetados': len(ind['affected']) if not isinstance(ind['affected'], pd.DataFrame) or not ind['affected'].empty else 0,
                'Recomendação': ind['rec'],
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Resumo DCMA', index=False)

        # Score sheet
        score_df = pd.DataFrame([{
            'Projeto': result['project_name'],
            'Score Final': round(result['final_score'], 1),
            'Total Atividades': result['total_tasks'],
            'Total Relações': result['total_rels'],
            'Atividades Críticas': result['critical_count'],
            'Gerado em': datetime.now().strftime('%d/%m/%Y %H:%M'),
        }])
        score_df.to_excel(writer, sheet_name='Score', index=False)

        # Detail sheets per indicator
        for ind in result['indicators']:
            aff = ind['affected']
            if isinstance(aff, pd.DataFrame) and not aff.empty:
                cols_want = ['task_id', 'task_code', 'task_name', 'target_drtn_hr_cnt',
                             'total_float_hr_cnt', 'early_start_date', 'early_end_date',
                             'cstr_type', 'status_code']
                cols_avail = [c for c in cols_want if c in aff.columns]
                aff[cols_avail].to_excel(
                    writer, sheet_name=f"#{ind['id']} {ind['short']}"[:31], index=False
                )
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge_html(status: str) -> str:
    cls = STATUS_CLASS.get(status, 'badge-na')
    return f'<span class="{cls}">{status}</span>'


def render_indicator_card(ind: dict):
    status = get_status(ind)
    color  = STATUS_COLOR[status]
    css_cls = {'OK': 'status-ok', 'ALERTA': 'status-alert', 'CRÍTICO': 'status-crit', 'N/A': ''}.get(status, '')

    val_str = 'N/A' if ind.get('na') else f"{ind['value']:.2f} {ind['unit']}"
    goal_str = (f"{ind.get('goal_min', 1.5):.1f}–{ind.get('goal_max', 3.0):.1f}"
                if ind['id'] == 14 else str(ind['goal']))

    n_affected = 0
    aff = ind['affected']
    if isinstance(aff, pd.DataFrame):
        n_affected = len(aff)
    elif isinstance(aff, list):
        n_affected = len(aff)

    pct_bar = 0
    if not ind.get('na') and ind['id'] != 14:
        pct_bar = min(ind['value'] / max(ind['goal'], 0.01) * 100, 100)

    st.markdown(f"""
    <div class="metric-card {css_cls}" style="margin-bottom:0">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
        <span style="color:#6699bb;font-size:11px;font-family:'JetBrains Mono',monospace">#{ind['id']}</span>
        {badge_html(status)}
      </div>
      <div style="font-weight:700;color:#e8f4ff;font-size:14px;margin-bottom:10px;line-height:1.3">{ind['name']}</div>
      <div style="font-size:26px;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;margin-bottom:4px">{val_str}</div>
      <div style="font-size:11px;color:#6699bb;margin-bottom:10px">Meta: {goal_str} {ind['unit'] if ind['id']!=14 else ''} · {n_affected} afetado(s)</div>
      <div style="background:#1e3a52;border-radius:3px;height:5px;overflow:hidden">
        <div style="width:{pct_bar:.1f}%;height:100%;background:{color};border-radius:3px;transition:width 0.8s"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="dcma-header">
      <span style="font-size:32px">📐</span>
      <div>
        <p class="dcma-title">DCMA Schedule Analyzer</p>
        <p class="dcma-subtitle">14-Point Schedule Quality Assessment · Primavera P6 XER</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ──
    uploaded = st.file_uploader(
        "Carregue o arquivo XER exportado do Primavera P6",
        type=['xer'],
        help="File → Export → Primavera PM (XER) no P6",
    )

    if not uploaded:
        st.info("👆 Faça upload de um arquivo **.XER** para iniciar a análise DCMA.")
        st.markdown("""
        **Como exportar o arquivo XER do Primavera P6:**
        1. Abra o P6 e selecione o projeto
        2. Vá em **File → Export**
        3. Escolha o formato **Primavera PM - (XER)**
        4. Clique em **Finish** e salve o arquivo
        5. Faça upload do arquivo `.xer` aqui
        """)
        return

    # ── Parse ──
    with st.spinner("🔍 Lendo e analisando o arquivo XER..."):
        try:
            raw_text = uploaded.read().decode('utf-8', errors='replace')
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            return

        dfs, parse_log = parse_xer(raw_text)
        result = run_dcma(dfs)

    if result is None:
        st.error("❌ Nenhuma atividade válida encontrada no arquivo XER.")
        with st.expander("Log de parsing"):
            for line in parse_log:
                st.markdown(line)
        return

    indicators = result['indicators']

    # ── Project banner ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Projeto", result['project_name'])
    with c2:
        st.metric("Atividades", result['total_tasks'])
    with c3:
        st.metric("Relações", result['total_rels'])
    with c4:
        st.metric("Críticas", result['critical_count'])
    with c5:
        ok_count = sum(1 for i in indicators if get_status(i) == 'OK')
        st.metric("Indicadores OK", f"{ok_count}/14")

    st.markdown("---")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Indicadores", "🔍 Detalhes", "📝 Log & Export"])

    # ════════════════════════════════════════
    # TAB 1 — DASHBOARD
    # ════════════════════════════════════════
    with tab1:
        col_gauge, col_pie, col_radar = st.columns(3)
        with col_gauge:
            st.plotly_chart(score_gauge(result['final_score']), use_container_width=True)
        with col_pie:
            st.markdown("**Distribuição de Status**")
            st.plotly_chart(pie_chart(indicators), use_container_width=True)
        with col_radar:
            st.markdown("**Radar de Qualidade**")
            st.plotly_chart(radar_chart(indicators), use_container_width=True)

        st.markdown("**Valor vs Meta por Indicador**")
        st.plotly_chart(bar_chart(indicators), use_container_width=True)

        # Critical & Alert callouts
        crit_inds  = [i for i in indicators if get_status(i) == 'CRÍTICO']
        alert_inds = [i for i in indicators if get_status(i) == 'ALERTA']
        if crit_inds or alert_inds:
            ca, cb = st.columns(2)
            with ca:
                if crit_inds:
                    st.error(f"🚨 **{len(crit_inds)} indicador(es) CRÍTICO(s)**")
                    for i in crit_inds:
                        st.markdown(f"- **#{i['id']} {i['name']}** — {i['value']:.1f} {i['unit']} (meta: {i['goal']})")
            with cb:
                if alert_inds:
                    st.warning(f"⚠️ **{len(alert_inds)} indicador(es) em ALERTA**")
                    for i in alert_inds:
                        st.markdown(f"- **#{i['id']} {i['name']}** — {i['value']:.1f} {i['unit']} (meta: {i['goal']})")

    # ════════════════════════════════════════
    # TAB 2 — INDICATORS GRID
    # ════════════════════════════════════════
    with tab2:
        rows = [indicators[i:i+2] for i in range(0, len(indicators), 2)]
        for row in rows:
            cols = st.columns(2)
            for col, ind in zip(cols, row):
                with col:
                    render_indicator_card(ind)
                    status = get_status(ind)
                    with st.expander(f"ℹ️ Fórmula e recomendação"):
                        st.markdown(f"**Fórmula:** `{ind['formula']}`")
                        st.markdown(f"**Recomendação:** {ind['rec']}")
                    st.markdown("")

    # ════════════════════════════════════════
    # TAB 3 — DETAILS TABLE + AFFECTED TASKS
    # ════════════════════════════════════════
    with tab3:
        # Summary table
        summary_rows = []
        for ind in indicators:
            s = get_status(ind)
            n_aff = len(ind['affected']) if isinstance(ind['affected'], pd.DataFrame) else 0
            summary_rows.append({
                '#': ind['id'],
                'Indicador': ind['name'],
                'Valor': f"{ind['value']:.2f} {ind['unit']}" if not ind.get('na') else 'N/A',
                'Meta': (f"{ind.get('goal_min',1.5):.1f}–{ind.get('goal_max',3.0):.1f}"
                         if ind['id'] == 14 else str(ind['goal'])),
                'Status': f"{STATUS_EMOJI[s]} {s}",
                'Afetados': n_aff,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Atividades Problemáticas por Indicador")

        for ind in indicators:
            if ind.get('na') or ind.get('is_rel'):
                continue
            aff = ind['affected']
            if not isinstance(aff, pd.DataFrame) or aff.empty:
                continue

            status = get_status(ind)
            with st.expander(f"{STATUS_EMOJI[status]} #{ind['id']} {ind['name']} — {len(aff)} atividade(s)"):
                cols_want = ['task_id', 'task_code', 'task_name', 'target_drtn_hr_cnt',
                             'total_float_hr_cnt', 'early_start_date', 'early_end_date']
                cols_avail = [c for c in cols_want if c in aff.columns]
                rename_map = {
                    'task_id': 'ID', 'task_code': 'Código', 'task_name': 'Nome',
                    'target_drtn_hr_cnt': 'Duração (h)', 'total_float_hr_cnt': 'Folga Total (h)',
                    'early_start_date': 'Início Cedo', 'early_end_date': 'Término Cedo',
                }
                display = aff[cols_avail].rename(columns=rename_map)
                st.dataframe(display.head(200), use_container_width=True, hide_index=True)
                if len(aff) > 200:
                    st.caption(f"Exibindo 200 de {len(aff)} registros. Exporte o Excel para ver todos.")

        # Relationship details
        for ind in indicators:
            if not ind.get('is_rel'):
                continue
            aff = ind['affected']
            if not isinstance(aff, pd.DataFrame) or aff.empty:
                continue
            status = get_status(ind)
            with st.expander(f"{STATUS_EMOJI[status]} #{ind['id']} {ind['name']} — {len(aff)} relação(ões)"):
                cols_want = ['task_id', 'pred_task_id', 'pred_type', 'lag_hr_cnt']
                cols_avail = [c for c in cols_want if c in aff.columns]
                rename_map = {
                    'task_id': 'ID Tarefa', 'pred_task_id': 'ID Predecessora',
                    'pred_type': 'Tipo', 'lag_hr_cnt': 'Lag (h)',
                }
                st.dataframe(aff[cols_avail].rename(columns=rename_map).head(200),
                             use_container_width=True, hide_index=True)

    # ════════════════════════════════════════
    # TAB 4 — LOG & EXPORT
    # ════════════════════════════════════════
    with tab4:
        st.markdown("### Log de Parsing")
        with st.container():
            for line in parse_log:
                if line.startswith('⚠'):
                    st.warning(line)
                else:
                    st.markdown(line)

        st.markdown("---")
        st.markdown("### Exportar Resultados")
        col_xl, col_csv = st.columns(2)

        with col_xl:
            try:
                excel_bytes = export_excel(result)
                st.download_button(
                    label="⬇️ Baixar Relatório Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"DCMA_{result['project_name']}_{datetime.now():%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Erro ao gerar Excel: {e}")

        with col_csv:
            csv_rows = []
            for ind in indicators:
                s = get_status(ind)
                n_aff = len(ind['affected']) if isinstance(ind['affected'], pd.DataFrame) else 0
                csv_rows.append({
                    'ID': ind['id'], 'Indicador': ind['name'],
                    'Valor': round(ind['value'], 2), 'Unidade': ind['unit'],
                    'Meta': ind['goal'], 'Status': s, 'Afetados': n_aff,
                    'Recomendação': ind['rec'],
                })
            csv_str = pd.DataFrame(csv_rows).to_csv(index=False, sep=';')
            st.download_button(
                label="⬇️ Baixar Resumo CSV",
                data=csv_str.encode('utf-8-sig'),
                file_name=f"DCMA_{result['project_name']}_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
            )


if __name__ == '__main__':
    main()
