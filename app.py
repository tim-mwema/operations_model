import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- PAGE CONFIG & THEME ---
st.set_page_config(page_title="🌎 Regional OPS Dashboard", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    * { font-family: 'Times New Roman', Times, serif !important; }
    :root { --wm-blue: #4A90E2; --wm-light: #EBF5FB; --wm-dark: #1B4F72; }
    
    .stMetric label { color: var(--wm-dark) !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 0.5px; }
    .stMetric div[data-testid="stMetricValue"] { color: var(--wm-blue) !important; font-size: 32px !important; font-weight: bold; }
    
    .location-metric div[data-testid="stMetricValue"] { 
        font-size: 16px !important; font-style: italic !important; line-height: 1.3 !important; 
        white-space: normal !important; word-wrap: break-word !important;
    }
    
    .comp-card { 
        border: 1px solid #d6eaf8; padding: 15px; border-radius: 8px; background: white; 
        box-shadow: 0 2px 4px rgba(74,144,226,0.1); min-height: 220px; 
        display: flex; flex-direction: column; justify-content: space-between;
    }
    
    /* REINFORCED TABLE WRAPPING */
    [data-testid="stDataFrame"] { width: 100% !important; overflow-x: hidden !important; }
    [data-testid="stDataFrame"] table { 
        width: 100% !important; table-layout: fixed !important; border-collapse: collapse !important;
    }
    [data-testid="stDataFrame"] th, 
    [data-testid="stDataFrame"] td { 
        white-space: pre-wrap !important; word-wrap: break-word !important; overflow-wrap: break-word !important;
        vertical-align: top !important; padding: 10px 8px !important; line-height: 1.4 !important;
        min-width: 80px !important; max-width: 300px !important; text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        white-space: normal !important; vertical-align: middle !important;
        line-height: 1.3 !important; min-width: 100px !important; max-width: 300px !important;
    }
    
    /* Hide scrollbar completely */
    [data-testid="stDataFrame"]::-webkit-scrollbar { display: none !important; }
    [data-testid="stDataFrame"] { -ms-overflow-style: none !important; scrollbar-width: none !important; }
    
    @media print {
        .no-print { display: none !important; }
        body { font-family: 'Times New Roman', serif; color: black; }
        .stMetric div[data-testid="stMetricValue"] { color: black !important; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Professionalized Services Dashboard")
st.caption("Kenyan Prototype | Predictive Maintenance | Strategic Investment Tool | Project Asset & ROI | SI Analysis")

# --- DATA ACQUISITION FROM GOOGLE SHEETS ---
@st.cache_data(ttl=300)
def load_data_from_sheets():
    try:
        sheet_id = st.secrets["google_sheets"]["sheet_id"]
    except Exception:
        sheet_id = sheet_id
        
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sheet: {str(e)}")
        return None

df = load_data_from_sheets()

if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Asset Health & Uptime", 
        " Investment Simulator + SI",
        " Compliance Tracker",
        "🤖 SI Portfolio Segmentation"
    ])

    # ==========================================
    # TAB 1: ASSET HEALTH & UPTIME 
    # ==========================================
    with tab1:
        st.sidebar.header("👆 Project Selection")
        selected_project_name = st.sidebar.selectbox("Select a Water System:", df['Project_Name'].tolist())
        row = df[df['Project_Name'] == selected_project_name].iloc[0]

        def get_risk_label(score, risk_type):
            if score == 0:
                labels = {"shock": "No Climate Shocks [Score: 0]", 
                         "theft": "No Theft Reported [Score: 0]", 
                         "vandalism": "No Vandalism [Score: 0]"}
                return labels[risk_type]
            if risk_type == "shock":
                if score == 1: return f"Minor Climate Shock [Score: {score}]"
                elif score <= 2: return f"Moderate Drought/Flood [Score: {score}]"
                else: return f"Major Climate Event [Score: {score}]"
            elif risk_type == "theft":
                if score == 1: return f"Minor Theft [Score: {score}]"
                elif score == 2: return f"Moderate Component Loss [Score: {score}]"
                else: return f"Critical Asset Theft [Score: {score}]"
            else:
                if score == 1: return f"Minor Physical Damage [Score: {score}]"
                elif score == 2: return f"Moderate Structural Damage [Score: {score}]"
                else: return f"Severe Infrastructure Damage [Score: {score}]"

        def calc_effective_age(age, shocks, theft, vandalism):
            return age + (shocks * 0.8) + (theft * 1.2) + (vandalism * 1.0)

        def calc_remaining_life(eff_age, expected_life):
            return max(0, ((1 - (eff_age / expected_life)) * 100))

        comps = {
            'Pump': {'age': 'Pump_Calendar_Age', 'life': 'Pump_Expected_Life', 'shock': 'Pump_Shocks_Index', 
                     'theft': 'Pump_Theft_Score', 'vand': 'Pump_Vandalism_Score', 'status': 'Pump_Status'},
            'Solar': {'age': 'Solar_Calendar_Age', 'life': 'Solar_Expected_Life', 'shock': 'Solar_Shocks_Index', 
                      'theft': 'Solar_Theft_Score', 'vand': 'Solar_Vandalism_Score', 'status': 'Solar_Status'},
            'Treatment': {'age': 'Treatment_Calendar_Age', 'life': 'Treatment_Expected_Life', 'shock': 'Treatment_Shocks_Index', 
                          'theft': 'Treatment_Theft_Score', 'vand': 'Treatment_Vandalism_Score', 'status': 'Treatment_Status'},
            'Access Points': {'age': 'AccessPoints_Calendar_Age', 'life': 'AccessPoints_Expected_Life', 'shock': 'AccessPoints_Shocks_Index', 
                              'theft': 'AccessPoints_Theft_Score', 'vand': 'AccessPoints_Vandalism_Score', 'status': 'AccessPoints_Status'}
        }

        metrics = {}
        degraded_assets = []
        
        for name, cols in comps.items():
            eff_age = calc_effective_age(row[cols['age']], row[cols['shock']], row[cols['theft']], row[cols['vand']])
            rem_life = calc_remaining_life(eff_age, row[cols['life']])
            
            age_ratio = eff_age / row[cols['life']]
            if age_ratio > 0.7:
                primary_risk = "Climate Shocks" if row[cols['shock']] >= row[cols['theft']] and row[cols['shock']] >= row[cols['vand']] else \
                              ("Theft" if row[cols['theft']] >= row[cols['vand']] else "Vandalism")
                degraded_assets.append((name, eff_age, row[cols['life']], primary_risk, age_ratio))
            
            metrics[name] = {
                'eff_age': eff_age, 'rem_life': rem_life, 'exp_life': row[cols['life']],
                'status': row[cols['status']],
                'shock_label': get_risk_label(row[cols['shock']], "shock"),
                'theft_label': get_risk_label(row[cols['theft']], "theft"),
                'vand_label': get_risk_label(row[cols['vand']], "vandalism")
            }

        overall_health = np.mean([m['rem_life'] for m in metrics.values()])
        
        is_core_up = (row['Pump_Status'] == 'Working' and 
                      row['Solar_Status'] == 'Working' and 
                      row['Treatment_Status'] == 'Working')
        current_uptime = (row['AccessPoints_Working_Count'] / row['AccessPoints_Total_Count'] * 100) if is_core_up else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric(label="Monthly Service Uptime", value=f"{current_uptime:.1f}%")
        with col2: st.metric(label="Overall System Health", value=f"{overall_health:.1f}%")
        with col3: 
            st.markdown('<div class="location-metric">', unsafe_allow_html=True)
            st.metric(label="Location", value=f"{row['County_Region']}, {row['Country']}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4: st.metric(label="Active Beneficiaries", value=f"{row['Active_Beneficiaries']:,}")

        st.divider()
        st.subheader("🔍 Component-Level Diagnostics (Risk-Adjusted)")
        c1, c2, c3, c4 = st.columns(4)
        
        for idx, (name, m) in enumerate(metrics.items()):
            color = "#27AE60" if m['rem_life'] > 50 else ("#F39C12" if m['rem_life'] > 20 else "#C0392B")
            bg = '#FADBD8' if m['status']=='Not Working' else 'white'
            col_ref = [c1, c2, c3, c4][idx]
            with col_ref:
                st.markdown(f"""
                <div class="comp-card" style="background:{bg}; border-top: 4px solid {color};">
                    <div>
                        <b style="font-size:16px;">{name}</b> 
                        <span style="float:right; color:{color}; font-weight:bold; font-size:18px;">{m['rem_life']:.1f}% Life</span><br>
                        <small>Eff. Age: {m['eff_age']:.1f} / Exp: {m['exp_life']} yrs</small><br>
                        <small>Status: <b>{m['status']}</b></small><hr style="margin:8px 0; border:0; border-top:1px solid #d6eaf8;">
                        <small style="color:#D68910">⚠️ {m['shock_label']}</small><br>
                        <small style="color:#C0392B">🚨 {m['theft_label']}</small><br>
                        <small style="color:#7D3C98">🔧 {m['vand_label']}</small>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("📋 Strategic Maintenance Recommendation")
        
        rec_title = ""
        rec_body = ""
        rec_color = "#166ED1"
        
        if overall_health < 30 or any(m['status'] == 'Not Working' for m in metrics.values()):
            rec_title = "🚨 MAJOR OVERHAUL / REPLACEMENT REQUIRED"
            rec_body = f"System health critical ({overall_health:.1f}% remaining). Immediate capital intervention required."
            rec_color = "#C52E1E"
        elif degraded_assets:
            rec_title = "⚠️ TARGETED INTERVENTION REQUIRED"
            asset_details = "<br>".join([
                f"• <b>{a[0]}</b>: Effective Age {a[1]:.1f}/{a[2]} yrs ({a[4]*100:.0f}% consumed). Primary risk driver: <b>{a[3]}</b>. Schedule repair and replacement assessment."
                for a in sorted(degraded_assets, key=lambda x: x[4], reverse=True)
            ])
            rec_body = f"The following assets are at imminent risk of degradation:<br><br>{asset_details}"
            rec_color = "#F39C12"
        else:
            rec_title = "✅ KEEP UNDER CLOSE MONITORING"
            rec_body = "All components within healthy parameters. Continue routine preventive maintenance. Verify climate shock/vandalism scores remain stable at next assessment."
            rec_color = "#27AE60"

        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {rec_color}15; border-left: 5px solid {rec_color};">
            <h3 style="margin-top:0; color: {rec_color};">{rec_title}</h3>
            <p style="font-size: 16px; margin-bottom:0; line-height: 1.6;">{rec_body}</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        if st.button("️ Generate Printable Report (PDF)", key="pdf_btn_tab1"):
            html_table = ""
            for name, m in metrics.items():
                html_table += f"<tr><td>{name}</td><td>{m['eff_age']:.1f}</td><td>{m['exp_life']}</td><td>{m['rem_life']:.1f}%</td><td>{m['status']}</td></tr>"
            
            printable_html = f"""
            <html><head><title>Asset Health Report - {selected_project_name}</title>
            <style>body{{font-family:'Times New Roman';padding:40px;}} table{{width:100%;border-collapse:collapse;margin-top:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#4A90E2;color:white;}}</style>
            </head><body>
                <h1 style="color:#1B4F72;">Water Mission: Asset Health Report</h1>
                <p><b>Project:</b> {selected_project_name} | <b>Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
                <p><b>Overall Health:</b> {overall_health:.1f}% | <b>Uptime:</b> {current_uptime:.1f}%</p>
                <table><tr><th>Component</th><th>Eff. Age</th><th>Exp. Life</th><th>Life Left</th><th>Status</th></tr>{html_table}</table>
                <br><p><b>Recommendation:</b> {rec_title}<br>{rec_body.replace('<br>','<br>')}</p>
            </body></html>"""
            
            st.components.v1.html(f'<script>window.open("","_blank").document.write(`{printable_html.replace("`","\\`")}`);</script>', height=0)
            st.success("✅ Printable report opened in new tab. Use Ctrl+P to Save as PDF.")

    # ==========================================
    # TAB 2: INVESTMENT SIMULATOR + UPDATED SI CLUSTERING
    # ==========================================
    with tab2:
        st.header("4-Project Strategic Investment Simulator + SI Clustering")
        st.markdown("Simulate investment scenarios. Risk-adjusted SI clustering uses **Effective Life**, **Replacement Frequency**, and **Cost/Person** to identify true viability tiers.")

        projects = ['A', 'B', 'C', 'D']
        proj_data = {}
        
        cols = st.columns(4)
        for i, p in enumerate(projects):
            with cols[i]:
                st.subheader(f"Project {p}")
                init = st.number_input(f"Init Cost ($)", key=f"{p}_init", value=[5000,25000,15000,8000][i])
                om = st.number_input(f"Annual O&M Expense ($)", key=f"{p}_om", value=[2000,500,1000,1500][i])
                revenue = st.number_input(f"Est. Annual Revenue ($)", key=f"{p}_rev", value=[800,3000,1500,600][i])
                base_life = st.number_input(f"Base Life (Yrs)", key=f"{p}_life", value=[10,15,12,8][i])
                ben = st.number_input(f"Beneficiaries", key=f"{p}_ben", value=[1000,4000,2500,1500][i])
                
                st.markdown("**Risk Profile**")
                shocks = st.slider("Climate Shocks (0-5)", 0, 5, [4,1,2,3][i], key=f"{p}_shocks")
                theft = st.slider("Theft (0-3)", 0, 3, [2,0,1,2][i], key=f"{p}_theft")
                vand = st.slider("Vandalism (0-3)", 0, 3, [1,0,1,2][i], key=f"{p}_vand")
                
                proj_data[p] = {'init': init, 'om': om, 'revenue': revenue, 'base_life': base_life, 'ben': ben, 
                                'shocks': shocks, 'theft': theft, 'vand': vand}

        # --- CALCULATIONS ---
        ANALYSIS_PERIOD = 15
        
        results = []
        for p, d in proj_data.items():
            eff_life = max(1, d['base_life'] - (d['shocks']*0.8) - (d['theft']*1.2) - (d['vand']*1.0))
            replacements_needed = int(np.ceil(ANALYSIS_PERIOD / eff_life)) - 1
            simple_roi = (d['init'] + d['om']) / d['ben']
            net_annual = d['revenue'] - d['om']
            years_to_recover = d['init'] / net_annual if net_annual > 0 else 999
            
            results.append({
                'Project': f"Project {p}", 
                'Simple_ROI': simple_roi, 
                'Years_to_Recover': years_to_recover,
                'Eff_Life': eff_life,
                'Replacements_Needed': replacements_needed,
                'Total_Lifecycle_Cost': d['init'] + (d['init'] * replacements_needed) + (d['om'] * ANALYSIS_PERIOD)
            })
        
        df_results = pd.DataFrame(results).sort_values('Simple_ROI', ascending=True)

        st.divider()
        
        st.subheader(" SI Investment Segmentation (Risk-Adjusted)")
        k_val_tab2 = st.slider("Number of Investment Tiers (K):", min_value=2, max_value=4, value=3, step=1, key="k_slider_tab2")
        
        current_label_map = {}
        if len(df_results) >= k_val_tab2:
            features_t2 = df_results[['Eff_Life', 'Replacements_Needed', 'Simple_ROI']].values
            
            if np.std(features_t2[:, 0]) == 0 and np.std(features_t2[:, 1]) == 0 and np.std(features_t2[:, 2]) == 0:
                st.warning("⚠️ All projects have identical profiles. Adjust sliders to create differentiation for meaningful SI clustering.")
                df_results['Inv_Cluster'] = 0
                df_results['Tier'] = " UNIFORM PROFILE"
                centroids_t2 = np.array([[np.mean(features_t2[:,0]), np.mean(features_t2[:,1]), np.mean(features_t2[:,2])]])
                current_label_map = {0: "🟡 UNIFORM PROFILE"}
            else:
                scaler_t2 = StandardScaler()
                scaled_t2 = scaler_t2.fit_transform(features_t2)
                
                kmeans_t2 = KMeans(n_clusters=k_val_tab2, random_state=42, n_init=20)
                df_results['Inv_Cluster'] = kmeans_t2.fit_predict(scaled_t2)
                
                centroids_t2 = scaler_t2.inverse_transform(kmeans_t2.cluster_centers_)
                inv_labels = {}
                roi_weight = 0.5
                
                for i, c in enumerate(centroids_t2):
                    life_val, repl_val, roi_val = c
                    quality_score = life_val - (repl_val * 2.0) - (roi_val * roi_weight)
                    inv_labels[i] = quality_score
                
                sorted_keys = sorted(inv_labels.keys(), key=lambda x: inv_labels[x], reverse=True)
                current_label_map = {sorted_keys[0]: "🏆 BEST INVESTMENT", sorted_keys[-1]: "❌ WORST INVESTMENT"}
                if len(sorted_keys) == 3: 
                    current_label_map[sorted_keys[1]] = "️ MODERATE REVIEW"
                elif len(sorted_keys) == 4: 
                    current_label_map[sorted_keys[1]] = "💎 HIGH POTENTIAL"
                    current_label_map[sorted_keys[2]] = "🟡 HIGH RISK"
                
                df_results['Tier'] = df_results['Inv_Cluster'].map(current_label_map)
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                fig_t2 = go.Figure()
                colors_t2 = ['#27AE60', '#F39C12', '#C0392B', '#8E44AD']
                for i in range(k_val_tab2):
                    mask = df_results['Inv_Cluster'] == i
                    fig_t2.add_trace(go.Scatter(
                        x=df_results[mask]['Eff_Life'], y=df_results[mask]['Replacements_Needed'],
                        mode='markers',
                        marker=dict(size=20, color=colors_t2[i % 4], line=dict(width=2, color='white')),
                        name=current_label_map.get(i, f"Cluster {i}")
                    ))
                fig_t2.update_layout(
                    title="Investment Viability: Effective Life vs Replacement Frequency",
                    xaxis_title="Risk-Adjusted Effective Life (Years)", 
                    yaxis_title=f"Required Replacements ({ANALYSIS_PERIOD}-Yr Horizon)",
                    plot_bgcolor='white', font=dict(family="Times New Roman"), height=650
                )
                st.plotly_chart(fig_t2, use_container_width=True)
            
            with col_v2:
                if 'scaled_t2' in locals():
                    sil = silhouette_score(scaled_t2, df_results['Inv_Cluster'])
                    st.metric("Silhouette Score", f"{sil:.3f}")
                    st.caption(">0.5 = Distinct tiers. <0.2 = Overlapping groups.")
                    
                    st.markdown("**Cluster Centroids (Risk + Cost Averages):**")
                    for i in range(k_val_tab2):
                        c_life = centroids_t2[i][0]
                        c_repl = centroids_t2[i][1]
                        c_roi = centroids_t2[i][2]
                        st.caption(f"{current_label_map.get(i, f'Cluster {i}')}: Avg Life={c_life:.1f}yrs, Avg Repl={c_repl:.0f}, Avg Cost=${c_roi:.2f}/person")

            st.dataframe(df_results[['Project', 'Tier', 'Simple_ROI', 'Years_to_Recover', 
                                     'Eff_Life', 'Replacements_Needed']].style.format({
                'Simple_ROI': '${:,.2f} / person', 
                'Years_to_Recover': '{:.1f} yrs', 
                'Eff_Life': '{:.1f} yrs',
                'Replacements_Needed': '{:.0f}'
            }).map(
                lambda x: 'background-color: #D5F5E3' if 'BEST' in str(x) else 
                         ('background-color: #FEF9E7' if 'MODERATE' in str(x) or 'POTENTIAL' in str(x) else 
                          ('background-color: #FADBD8' if 'WORST' in str(x) or 'RISK' in str(x) else '')),
                subset=['Tier']
            ), use_container_width=True, hide_index=True)

        st.divider()
        if st.button("️ Generate Printable Investment Report (PDF)", key="pdf_btn_tab2"):
            rows_html = ""
            for _, r in df_results.iterrows():
                rows_html += f"<tr><td>{r['Project']}</td><td>{r['Tier']}</td><td>${r['Simple_ROI']:,.2f}</td><td>{r['Eff_Life']:.1f} yrs</td><td>{r['Replacements_Needed']}</td><td>${r['Total_Lifecycle_Cost']:,.0f}</td></tr>"
            
            printable_html = f"""
            <html><head><title>SI Investment Simulator Report</title>
            <style>body{{font-family:'Times New Roman';padding:40px;}} table{{width:100%;border-collapse:collapse;margin-top:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#4A90E2;color:white;}}</style>
            </head><body>
                <h1 style="color:#1B4F72;">Strategic Investment Simulator Report</h1>
                <p><b>Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')} | <b>Analysis Period:</b> {ANALYSIS_PERIOD} Years</p>
                <table>
                    <tr><th>Project</th><th>SI Tier</th><th>Cost/Person</th><th>Eff. Life</th><th>Replacements</th><th>Total Lifecycle Cost</th></tr>
                    {rows_html}
                </table>
                <br><p><i>Note: SI Clustering based on Effective Life, Replacement Frequency, and Cost Efficiency.</i></p>
            </body></html>"""
            
            st.components.v1.html(f'<script>window.open("","_blank").document.write(`{printable_html.replace("`","\\`")}`);</script>', height=0)
            st.success("✅ Investment report opened in new tab. Use Ctrl+P to Save as PDF.")

    # ==========================================
    # TAB 3: COMPLIANCE TRACKER (CLEAN, NO CLUSTERING)
    # ==========================================
    with tab3:
        st.header("Regional Performance & Compliance Tracker")
        st.markdown("Real-time Preventive Maintenance (PM) compliance tracking and Revenue Integrity analysis. Calculations automated based on today's date (**{}**).".format(date.today()))

        today = date.today()
        
        st.subheader("💰 Revenue Integrity & Non-Revenue Water (NRW) Analysis")
        nrw_rows = []
        for _, r in df.iterrows():
            if all(col in r.index for col in ['Tariff_Per_Liter', 'Monthly_Production_Liters', 'Actual_Monthly_Collected']):
                try:
                    tariff = float(r['Tariff_Per_Liter'])
                    prod_liters = float(r['Monthly_Production_Liters'])
                    actual_coll = float(r['Actual_Monthly_Collected'])
                    
                    true_expected_rev = tariff * prod_liters
                    deviation = true_expected_rev - actual_coll
                    deviation_pct = (deviation / true_expected_rev * 100) if true_expected_rev > 0 else 0
                    
                    nrw_status = "✅ Low NRW (<5%)" if deviation_pct < 5 else f"️ High NRW ({deviation_pct:.1f}%)"

                    nrw_rows.append({
                        'Project': r['Project_Name'], 'Tariff ($/L)': f"${tariff:.4f}",
                        'Monthly Production (L)': f"{prod_liters:,.0f}",
                        'True Expected Revenue ($)': f"${true_expected_rev:,.2f}",
                        'Actual Collected ($)': f"${actual_coll:,.2f}",
                        'Deviation ($)': f"${deviation:,.2f}", 'Deviation %': f"{deviation_pct:.1f}%",
                        'NRW Status': nrw_status
                    })
                except: pass

        if nrw_rows:
            df_nrw = pd.DataFrame(nrw_rows)
            st.dataframe(df_nrw.style.map(
                lambda x: 'background-color: #FADBD8; color: #C0392B; font-weight: bold' if 'High NRW' in str(x) else 
                         ('background-color: #D5F5E3; color: #27AE60; font-weight: bold' if 'Low NRW' in str(x) else ''),
                subset=['NRW Status']
            ), use_container_width=True, hide_index=True)
            
            n1, n2 = st.columns(2)
            n1.metric("Projects with High NRW", len(df_nrw[df_nrw['NRW Status'].str.contains('High')]))
            n2.metric("Total Monthly Leakage", f"${df_nrw['Deviation ($)'].str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float).sum():,.2f}")
        else:
            st.warning("️ Financial data columns missing.")

        st.divider()

        st.subheader("🔧 Preventive Maintenance (PM) Compliance Tracker")
        
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            selected_project = st.selectbox("Select Project:", df['Project_Name'].unique())
        with sel_col2:
            asset_types = ["Pump", "Solar", "Treatment"]
            selected_asset = st.selectbox("Select Asset Type:", asset_types)

        proj_row = df[df['Project_Name'] == selected_project].iloc[0]
        prefix = selected_asset
        last_pm_col = f"{prefix}_Last_PM_Date"
        next_due_col = f"{prefix}_Next_PM_Due"
        
        days_overdue = None; risk_score = None; status_text = "No Date Set"
        last_pm_date_str = "N/A"; next_due_date_str = "N/A"

        if last_pm_col in proj_row.index and next_due_col in proj_row.index:
            try:
                last_pm_date = pd.to_datetime(proj_row[last_pm_col]).date() if pd.notna(proj_row[last_pm_col]) else None
                next_due_date = pd.to_datetime(proj_row[next_due_col]).date() if pd.notna(proj_row[next_due_col]) else None
                
                if last_pm_date: last_pm_date_str = last_pm_date.strftime('%Y-%m-%d')
                if next_due_date: next_due_date_str = next_due_date.strftime('%Y-%m-%d')

                if next_due_date:
                    days_overdue = (today - next_due_date).days
                    if days_overdue <= 0: risk_score, status_text = 0, "✅ On Track"
                    elif days_overdue <= 10: risk_score, status_text = 10, "⚠️ Low Risk (≤10 days)"
                    elif days_overdue <= 21: risk_score, status_text = 20, "🟠 Medium Risk (11-21 days)"
                    else: risk_score, status_text = 30, "🔴 CRITICAL (>21 days)"
            except: status_text = "Invalid Date Format"

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1: st.metric("Status", status_text)
        with res_col2: st.metric("Days Overdue", days_overdue if days_overdue is not None else "N/A")
        with res_col3: st.metric("Risk Score", risk_score if risk_score is not None else "N/A")
        with res_col4: st.metric("Last PM", last_pm_date_str)

        if risk_score is not None:
            st.progress(min(risk_score / 30, 1.0), text=f"Risk Level: {risk_score}/30")
        st.info(f"**Next Due:** {next_due_date_str} | **Today:** {today}")

        st.divider()
        st.subheader("📈 Full Regional PM Compliance Overview")
        
        pm_rows = []
        for _, r in df.iterrows():
            for asset in asset_types:
                nxt_col = f"{asset}_Next_PM_Due"
                lst_col = f"{asset}_Last_PM_Date"
                if nxt_col in r.index and pd.notna(r[nxt_col]):
                    try:
                        nxt_dt = pd.to_datetime(r[nxt_col]).date()
                        lst_dt = pd.to_datetime(r[lst_col]).date() if pd.notna(r[lst_col]) else None
                        ovrd = (today - nxt_dt).days
                        
                        if ovrd <= 0: sc, st_txt = 0, "✅ On Track"
                        elif ovrd <= 10: sc, st_txt = 10, "⚠️ Low Risk"
                        elif ovrd <= 21: sc, st_txt = 20, "🟠 Medium Risk"
                        else: sc, st_txt = 30, "🔴 CRITICAL"
                        
                        pm_rows.append({'Project': r['Project_Name'], 'Asset': asset, 'Last PM': lst_dt.strftime('%Y-%m-%d') if lst_dt else 'N/A',
                                       'Next Due': nxt_dt.strftime('%Y-%m-%d'), 'Days Overdue': ovrd, 'Risk Score': sc, 'Status': st_txt})
                    except: pass

        if pm_rows:
            df_pm = pd.DataFrame(pm_rows).sort_values(['Risk Score', 'Days Overdue'], ascending=[False, False])
            st.dataframe(df_pm.style.map(
                lambda x: 'background-color: #FADBD8; color: #C0392B; font-weight: bold' if 'CRITICAL' in str(x) else 
                         ('background-color: #FEF9E7; color: #F39C12; font-weight: bold' if 'Medium' in str(x) else 
                          ('background-color: #D5F5E3; color: #27AE60; font-weight: bold' if 'On Track' in str(x) else '')),
                subset=['Status']
            ), use_container_width=True, hide_index=True)
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Assets On Track", len(df_pm[df_pm['Risk Score'] == 0]))
            s2.metric("Low Risk", len(df_pm[df_pm['Risk Score'] == 10]))
            s3.metric("Medium Risk", len(df_pm[df_pm['Risk Score'] == 20]))
            s4.metric(" Critical", len(df_pm[df_pm['Risk Score'] == 30]))

        st.divider()
        if st.button("🖨️ Generate Printable Compliance Report (PDF)", key="pdf_btn_tab3"):
            nrw_html = ""
            if nrw_rows:
                nrw_html = "<h3>Revenue Integrity Summary</h3><table><tr><th>Project</th><th>NRW %</th><th>Status</th></tr>"
                for r in nrw_rows:
                    nrw_html += f"<tr><td>{r['Project']}</td><td>{r['Deviation %']}</td><td>{r['NRW Status']}</td></tr>"
                nrw_html += "</table><br>"
            
            pm_html = "<h3>PM Compliance Summary</h3><table><tr><th>Project</th><th>Asset</th><th>Overdue Days</th><th>Risk Score</th><th>Status</th></tr>"
            for r in pm_rows:
                pm_html += f"<tr><td>{r['Project']}</td><td>{r['Asset']}</td><td>{r['Days Overdue']}</td><td>{r['Risk Score']}</td><td>{r['Status']}</td></tr>"
            pm_html += "</table>"

            printable_html = f"""
            <html><head><title>Compliance Tracker Report</title>
            <style>body{{font-family:'Times New Roman';padding:40px;}} table{{width:100%;border-collapse:collapse;margin-top:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#4A90E2;color:white;}}</style>
            </head><body>
                <h1 style="color:#1B4F72;">Regional Compliance & NRW Report</h1>
                <p><b>Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')} | <b>Selected Project:</b> {selected_project}</p>
                {nrw_html}
                {pm_html}
            </body></html>"""
            
            st.components.v1.html(f'<script>window.open("","_blank").document.write(`{printable_html.replace("`","\\`")}`);</script>', height=0)
            st.success("✅ Compliance report opened in new tab. Use Ctrl+P to Save as PDF.")

    # ==========================================
    # TAB 4: TRUE UNSUPERVISED SI SEGMENTATION
    # ==========================================
    with tab4:
        st.header("🤖 SI Portfolio Segmentation: Pattern Discovery Engine")
        st.markdown("SI discovers natural project groupings based on NRW % vs Max PM Overdue Days. Labels and recommendations are dynamically generated from cluster centroids—no predefined quadrants.")

        cluster_data = []
        for _, r in df.iterrows():
            nrw_pct = 0
            if all(col in r.index for col in ['Tariff_Per_Liter', 'Monthly_Production_Liters', 'Actual_Monthly_Collected']):
                try:
                    exp_rev = float(r['Tariff_Per_Liter']) * float(r['Monthly_Production_Liters'])
                    act_coll = float(r['Actual_Monthly_Collected'])
                    nrw_pct = ((exp_rev - act_coll) / exp_rev * 100) if exp_rev > 0 else 0
                except: nrw_pct = 0
            
            max_overdue = 0
            for asset in ["Pump", "Solar", "Treatment"]:
                nxt_col = f"{asset}_Next_PM_Due"
                if nxt_col in r.index and pd.notna(r[nxt_col]):
                    try:
                        nxt_dt = pd.to_datetime(r[nxt_col]).date()
                        ovrd = (today - nxt_dt).days
                        max_overdue = max(max_overdue, ovrd)
                    except: pass
            
            cluster_data.append({'Project': r['Project_Name'], 'NRW_Percent': nrw_pct, 'Max_PM_Overdue_Days': max_overdue})

        if len(cluster_data) >= 2:  
            df_cluster = pd.DataFrame(cluster_data)
            
            k_val_tab4 = st.slider("Number of Clusters (K):", min_value=2, max_value=min(6, len(df_cluster)), value=3, step=1, key="k_slider_tab4",
                                   help="Let SI discover natural groupings. Higher K reveals niche patterns.")
            
            scaler = StandardScaler()
            features = scaler.fit_transform(df_cluster[['NRW_Percent', 'Max_PM_Overdue_Days']])
            
            kmeans = KMeans(n_clusters=k_val_tab4, random_state=42, n_init=10)
            df_cluster['Cluster'] = kmeans.fit_predict(features)
            
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_labels = {}
            cluster_reasons = {}
            
            for i, centroid in enumerate(centroids):
                nrw_val, overdue_val = centroid
                
                nrw_desc = "High NRW" if nrw_val > 10 else "Low NRW"
                overdue_desc = "High PM Risk" if overdue_val > 21 else ("Medium PM Risk" if overdue_val > 10 else "Low PM Risk")
                
                cluster_labels[i] = f"Cluster {i}: {nrw_desc} + {overdue_desc}"
                
                if nrw_val > 10 and overdue_val > 21:
                    cluster_reasons[i] = "🔴 CRITICAL: Both financial leakage AND operational neglect detected. Requires emergency joint task force (finance + maintenance). Immediate site audit within 7 days."
                elif nrw_val > 10 and overdue_val <= 10:
                    cluster_reasons[i] = "🟡 REVENUE CRISIS: Technical team is performing well (low overdue), but billing/collection systems are failing. Deploy revenue audit team. Do NOT divert maintenance resources."
                elif nrw_val <= 10 and overdue_val > 21:
                    cluster_reasons[i] = " OPERATIONAL DECLINE: Finances are healthy but maintenance is slipping dangerously. Schedule immediate catch-up PM before asset failures cause revenue loss. Review technician capacity."
                elif nrw_val <= 10 and overdue_val <= 10:
                    cluster_reasons[i] = "✅ MODEL PERFORMER: Excellent financial AND operational health. Document best practices. Use as training site. No intervention needed."
                else:
                    if nrw_val > np.median(df_cluster['NRW_Percent']):
                        cluster_reasons[i] = "️ FINANCIAL FOCUS: Above-average NRW requires billing system review. Monitor PM compliance to prevent compounding risks."
                    else:
                        cluster_reasons[i] = "⚠️ OPERATIONAL FOCUS: Above-average PM overdue requires maintenance sprint. Strong finances provide buffer for corrective action."
            
            df_cluster['Segment'] = df_cluster['Cluster'].map(cluster_labels)
            df_cluster['Recommendation'] = df_cluster['Cluster'].map(cluster_reasons)
            
            col_v1, col_v2 = st.columns([2, 1])
            
            with col_v1:
                fig_q = go.Figure()
                colors_q = ['#C0392B', '#F1C40F', '#E67E22', '#27AE60', '#8E44AD', '#2980B9']
                
                for i in range(k_val_tab4):
                    mask = df_cluster['Cluster'] == i
                    fig_q.add_trace(go.Scatter(
                        x=df_cluster[mask]['NRW_Percent'], y=df_cluster[mask]['Max_PM_Overdue_Days'],
                        mode='markers',
                        marker=dict(size=20, color=colors_q[i % len(colors_q)], line=dict(width=2, color='white')),
                        name=cluster_labels[i],
                        hovertemplate=f"<b>{cluster_labels[i]}</b><br>Avg NRW: {centroids[i][0]:.1f}%<br>Avg Overdue: {centroids[i][1]:.0f} days<extra></extra>"
                    ))
                
                fig_q.update_layout(
                    title="SI-Discovered Portfolio Segments: NRW % vs Max PM Overdue Days",
                    xaxis_title="Non-Revenue Water (%)", yaxis_title="Max PM Overdue (Days)",
                    plot_bgcolor='white', font=dict(family="Times New Roman"), height=750,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig_q, use_container_width=True)
            
            with col_v2:
                sil = silhouette_score(features, df_cluster['Cluster'])
                st.metric("Silhouette Score", f"{sil:.3f}")
                st.caption(">0.5 = Well-separated clusters. <0.2 = Overlapping groups; try different K.")
                
                st.markdown("**Cluster Centroids (SI-Discovered Averages):**")
                for i in range(k_val_tab4):
                    st.caption(f"{cluster_labels[i]}: Avg NRW={centroids[i][0]:.1f}%, Avg Overdue={centroids[i][1]:.0f} days")

            st.divider()
            st.subheader("SI-Generated Segment Analysis & Tailored Support")
            
            action_rows = []
            for i in range(k_val_tab4):
                c_df = df_cluster[df_cluster['Cluster'] == i]
                avg_nrw = c_df['NRW_Percent'].mean()
                avg_ovr = c_df['Max_PM_Overdue_Days'].mean()
                
                action_rows.append({
                    'Segment': cluster_labels[i], 
                    '# Projects': len(c_df),
                    'Avg NRW %': f"{avg_nrw:.1f}%", 
                    'Avg Max Overdue': f"{avg_ovr:.0f} days",
                    'Tailored Support Recommendation': cluster_reasons[i]
                })
            
            df_actions = pd.DataFrame(action_rows)
            
            st.dataframe(df_actions.style.map(
                lambda x: 'background-color: #FADBD8; color: #C0392B; font-weight: bold' if 'CRITICAL' in str(x) else 
                         ('background-color: #FEF9E7; color: #F1C40F; font-weight: bold' if 'REVENUE CRISIS' in str(x) else 
                          ('background-color: #FDEBD0; color: #E67E22; font-weight: bold' if 'OPERATIONAL DECLINE' in str(x) else 
                           ('background-color: #D5F5E3; color: #27AE60; font-weight: bold' if 'MODEL PERFORMER' in str(x) else ''))),
                subset=['Segment']
            ), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("📊 Project Assignments by SI-Discovered Segment")
            st.dataframe(df_cluster[['Project', 'Segment', 'NRW_Percent', 'Max_PM_Overdue_Days', 'Recommendation']].style.format({
                'NRW_Percent': '{:.1f}%', 'Max_PM_Overdue_Days': '{:.0f} days'
            }).map(
                lambda x: 'background-color: #FADBD8' if 'CRITICAL' in str(x) else 
                         ('background-color: #FEF9E7' if 'REVENUE CRISIS' in str(x) else 
                          ('background-color: #FDEBD0' if 'OPERATIONAL DECLINE' in str(x) else 
                           ('background-color: #D5F5E3' if 'MODEL PERFORMER' in str(x) else ''))),
                subset=['Segment']
            ), use_container_width=True, hide_index=True)

            st.divider()
            if st.button("🖨️ Generate Printable SI Segmentation Report (PDF)", key="pdf_btn_tab4"):
                seg_html = "<h3>SI Cluster Assignments & Recommendations</h3><table><tr><th>Project</th><th>Segment</th><th>NRW %</th><th>Max Overdue</th><th>Recommendation</th></tr>"
                for _, r in df_cluster.iterrows():
                    seg_html += f"<tr><td>{r['Project']}</td><td>{r['Segment']}</td><td>{r['NRW_Percent']:.1f}%</td><td>{r['Max_PM_Overdue_Days']:.0f} days</td><td>{r['Recommendation']}</td></tr>"
                seg_html += "</table>"

                printable_html = f"""
                <html><head><title>SI Portfolio Segmentation Report</title>
                <style>body{{font-family:'Times New Roman';padding:40px;}} table{{width:100%;border-collapse:collapse;margin-top:20px;}} th,td{{border:1px solid #ddd;padding:8px;text-align:left;}} th{{background:#4A90E2;color:white;}}</style>
                </head><body>
                    <h1 style="color:#1B4F72;">SI Portfolio Segmentation Report</h1>
                    <p><b>Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')} | <b>K Value:</b> {k_val_tab4} | <b>Silhouette Score:</b> {sil:.3f}</p>
                    {seg_html}
                </body></html>"""
                
                st.components.v1.html(f'<script>window.open("","_blank").document.write(`{printable_html.replace("`","\\`")}`);</script>', height=0)
                st.success("✅ SI segmentation report opened in new tab. Use Ctrl+P to Save as PDF.")
                
        else:
            st.warning("️ Need at least 2 projects with complete financial and PM data for clustering.")

else:
    st.stop()