import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================
# ENTERPRISE PLATFORM CONFIGURATION
# ================================

class EnterpriseConfig:
    """Enterprise-grade configuration management"""
    PAGE_TITLE = "ULTRA MARATHONS ENTERPRISE PLATFORM"
    PAGE_ICON = "🚀"
    LAYOUT = "wide"
    
    # Professional color schemes
    THEMES = {
        'light': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'dark': '#343a40',
            'light': '#f8f9fa',
            'background': '#ffffff',
            'card_bg': '#f8f9fa'
        }
    }

# ================================
# ENTERPRISE DATA ENGINE
# ================================

class DataEngine:
    """Enterprise data processing engine with ML capabilities"""
    
    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def load_enterprise_data(file_paths):
        """Load data with enterprise-grade error handling"""
        try:
            # Try multiple file paths
            for path in file_paths:
                if os.path.exists(path):
                    data = pd.read_csv(path)
                    st.success(f"✅ Enterprise data loaded: {len(data):,} records")
                    return DataEngine._clean_enterprise_data(data)
            
            st.error("❌ No data files found")
            return None
            
        except Exception as e:
            st.error(f"❌ Data engine error: {str(e)}")
            return None
    
    @staticmethod
    def _clean_enterprise_data(data):
        """Enterprise-grade data cleaning pipeline"""
        # Column standardization
        data.columns = data.columns.str.replace(' ', '_').str.lower().str.strip()
        
        # Data quality pipeline
        data = DataEngine._clean_demographics(data)
        data = DataEngine._clean_performance(data)
        data = DataEngine._clean_events(data)
        
        # Feature engineering
        data = DataEngine._create_enterprise_features(data)
        
        return data
    
    @staticmethod
    def _clean_demographics(data):
        """Clean demographic data with advanced validation"""
        # Age processing
        if 'athlete_year_of_birth' in data.columns:
            data = data.dropna(subset=['athlete_year_of_birth'])
            data['age'] = data['year_of_event'] - data['athlete_year_of_birth']
            
            # Fix data inconsistencies
            condition = data['athlete_year_of_birth'] > data['year_of_event']
            data.loc[condition, ['athlete_year_of_birth','year_of_event']] = \
                data.loc[condition, ['year_of_event','athlete_year_of_birth']].values
            
            data['age'] = data['year_of_event'] - data['athlete_year_of_birth']
            data = data[(data['age'] >= 12) & (data['age'] <= 75)]
        
        # Gender processing
        if 'athlete_gender' in data.columns:
            data = data[data['athlete_gender'].isin(['M', 'F'])]
            data['athlete_gender'] = data['athlete_gender'].map({'F': 0, 'M': 1})
        
        return data
    
    @staticmethod
    def _clean_performance(data):
        """Clean performance metrics"""
        # Distance standardization
        if 'event_distance/length' in data.columns:
            data['event_distance/length'] = data['event_distance/length'].astype(str)
            
            # Convert miles to km
            mile_mask = data['event_distance/length'].str.contains('mi', na=False)
            if mile_mask.any():
                data.loc[mile_mask, 'event_distance/length'] = \
                    pd.to_numeric(data.loc[mile_mask, 'event_distance/length'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce') * 1.60934
            
            # Extract numeric distances
            data['event_distance/length'] = pd.to_numeric(
                data['event_distance/length'].str.extract(r'(\d+\.?\d*)')[0], 
                errors='coerce'
            )
            data = data[(data['event_distance/length'] >= 5) & (data['event_distance/length'] <= 500)]
        
        # Speed processing
        if 'athlete_average_speed' in data.columns:
            data['athlete_average_speed'] = pd.to_numeric(data['athlete_average_speed'], errors='coerce')
            
            # Unit conversion for historical data
            if 'year_of_event' in data.columns:
                early_mask = data['year_of_event'] <= 1995
                data.loc[early_mask, 'athlete_average_speed'] *= 3.6
        
        return data
    
    @staticmethod
    def _clean_events(data):
        """Clean event-related data"""
        # Remove time-based entries
        if 'event_distance/length' in data.columns:
            time_mask = data['event_distance/length'].astype(str).str.contains('h', na=False)
            data = data[~time_mask]
        
        return data
    
    @staticmethod
    def _create_enterprise_features(data):
        """Create advanced features for analytics"""
        # Performance tiers
        if 'athlete_average_speed' in data.columns:
            data['performance_tier'] = pd.cut(
                data['athlete_average_speed'],
                bins=[0, 8, 12, 15, 100],
                labels=['Beginner', 'Intermediate', 'Advanced', 'Elite']
            )
        
        # Age groups
        if 'age' in data.columns:
            data['age_group'] = pd.cut(
                data['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
        
        # Event difficulty
        if 'event_distance/length' in data.columns:
            data['event_difficulty'] = pd.cut(
                data['event_distance/length'],
                bins=[0, 42, 80, 160, 500],
                labels=['Marathon', 'Ultra', 'Extreme', 'Ultra-Extreme']
            )
        
        return data
    
    @staticmethod
    def get_ml_insights(data):
        """Generate machine learning insights"""
        insights = {}
        
        # Performance predictions
        if len(data) > 1000:
            insights['performance_trend'] = [DataEngine._calculate_performance_trend(data)]
            insights['participation_forecast'] = [DataEngine._forecast_participation(data)]
            insights['demographic_shifts'] = DataEngine._analyze_demographic_shifts(data)
        
        return insights
    
    @staticmethod
    def _calculate_performance_trend(data):
        """Calculate performance improvement trends"""
        if 'athlete_average_speed' in data.columns and 'year_of_event' in data.columns:
            trend_data = data.groupby('year_of_event')['athlete_average_speed'].mean().reset_index()
            trend_data = trend_data.dropna()
            if len(trend_data) > 1:
                slope = np.polyfit(trend_data['year_of_event'], trend_data['athlete_average_speed'], 1)[0]
                return f"{'Improving' if slope > 0 else 'Declining'} by {abs(slope):.3f} km/h per year"
        return "Insufficient data for trend analysis"
    
    @staticmethod
    def _forecast_participation(data):
        """Simple participation forecasting"""
        if 'year_of_event' in data.columns:
            yearly = data['year_of_event'].value_counts().sort_index()
            if len(yearly) > 5:
                growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
                return f"Historical growth: {growth:.1f}%"
        return "Insufficient data for forecasting"
    
    @staticmethod
    def _analyze_demographic_shifts(data):
        """Analyze demographic changes"""
        insights = []
        if 'age' in data.columns and 'year_of_event' in data.columns:
            recent_data = data[data['year_of_event'] >= 2015]
            older_data = data[data['year_of_event'] <= 2000]
            
            if not recent_data.empty and not older_data.empty:
                recent_avg_age = recent_data['age'].mean()
                older_avg_age = older_data['age'].mean()
                if not pd.isna(recent_avg_age) and not pd.isna(older_avg_age):
                    change = recent_avg_age - older_avg_age
                    insights.append(f"Average age change: {change:+.1f} years")
        
        return insights if insights else ["Stable demographic patterns"]

# ================================
# ENTERPRISE VISUALIZATION ENGINE
# ================================

class VisualizationEngine:
    """Professional visualization engine for enterprise dashboards"""
    
    def __init__(self, theme):
        self.theme = theme
        self.chart_height = 450
    
    def create_executive_summary(self, data, kpis):
        """Create executive summary dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Participation Growth', 'Performance Trends', 
                          'Gender Distribution', 'Age Demographics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Participation growth
        if 'year_of_event' in data.columns:
            participation = data['year_of_event'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=participation.index, y=participation.values, name="Participants"),
                row=1, col=1
            )
        
        # Performance trends
        if 'athlete_average_speed' in data.columns and 'year_of_event' in data.columns:
            performance_data = data.dropna(subset=['athlete_average_speed'])
            if not performance_data.empty:
                performance = performance_data.groupby('year_of_event')['athlete_average_speed'].mean()
                fig.add_trace(
                    go.Scatter(x=performance.index, y=performance.values, 
                              name="Avg Speed", line=dict(color='orange')),
                    row=1, col=2
                )
        
        # Gender distribution
        if 'athlete_gender' in data.columns:
            gender_data = data.dropna(subset=['athlete_gender'])
            if not gender_data.empty:
                gender_counts = gender_data['athlete_gender'].value_counts()
                fig.add_trace(
                    go.Pie(labels=['Male', 'Female'], values=gender_counts.values,
                          name="Gender"),
                    row=2, col=1
                )
        
        # Age distribution
        if 'age' in data.columns:
            age_data = data.dropna(subset=['age'])
            if not age_data.empty:
                fig.add_trace(
                    go.Histogram(x=age_data['age'], nbinsx=20, name="Age Distribution"),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text="<b>Executive Summary Dashboard</b>")
        return fig
    
    def create_performance_metrics(self, data):
        """Create comprehensive performance metrics dashboard"""
        # Performance by decade
        if 'year_of_event' in data.columns and 'athlete_average_speed' in data.columns:
            data['decade'] = (data['year_of_event'] // 10) * 10
            decade_performance = data.groupby('decade')['athlete_average_speed'].agg(['mean', 'std']).reset_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=decade_performance['decade'],
                y=decade_performance['mean'],
                name='Average Speed',
                error_y=dict(type='data', array=decade_performance['std'], visible=True)
            ))
            fig1.update_layout(
                title='<b>Performance Evolution by Decade</b>',
                xaxis_title='Decade',
                yaxis_title='Average Speed (km/h)',
                height=400
            )
        else:
            fig1 = self._create_empty_chart("Performance data unavailable")
        
        # Speed distribution by performance tier
        if 'performance_tier' in data.columns and 'athlete_average_speed' in data.columns:
            performance_data = data.dropna(subset=['performance_tier', 'athlete_average_speed'])
            if not performance_data.empty:
                fig2 = px.violin(
                    performance_data, 
                    x='performance_tier', 
                    y='athlete_average_speed',
                    title='<b>Speed Distribution by Performance Tier</b>',
                    color='performance_tier'
                )
            else:
                fig2 = self._create_empty_chart("Performance tier data unavailable")
        else:
            fig2 = self._create_empty_chart("Performance tier data unavailable")
        
        # Finish rate analysis
        if 'event_number_of_finishers' in data.columns and 'event_name' in data.columns:
            finish_rates = data.groupby('event_name').agg({
                'event_number_of_finishers': 'sum',
                'athlete_id': 'count'
            }).reset_index()
            finish_rates['finish_rate'] = (finish_rates['event_number_of_finishers'] / finish_rates['athlete_id']) * 100
            
            fig3 = px.scatter(
                finish_rates.head(20),
                x='athlete_id',
                y='finish_rate',
                size='event_number_of_finishers',
                title='<b>Event Finish Rate Analysis</b>',
                hover_name='event_name'
            )
        else:
            fig3 = self._create_empty_chart("Finish rate data unavailable")
        
        return fig1, fig2, fig3
    
    def create_geographic_intelligence(self, data):
        """Create comprehensive geographic analysis"""
        # Country participation heatmap
        if 'event_country' in data.columns:
            country_data = data.dropna(subset=['event_country'])
            if not country_data.empty:
                country_participation = country_data['event_country'].value_counts().reset_index()
                country_participation.columns = ['Country', 'Participants']
                
                fig1 = px.choropleth(
                    country_participation,
                    locations='Country',
                    locationmode='country names',
                    color='Participants',
                    title='<b>Global Participation Heatmap</b>',
                    color_continuous_scale='Viridis',
                    height=500
                )
            else:
                fig1 = self._create_empty_chart("Geographic data unavailable")
        else:
            fig1 = self._create_empty_chart("Geographic data unavailable")
        
        # Top countries bar chart
        if 'event_country' in data.columns:
            country_data = data.dropna(subset=['event_country'])
            if not country_data.empty:
                top_countries = country_data['event_country'].value_counts().head(15).reset_index()
                top_countries.columns = ['Country', 'Participants']
                
                fig2 = px.bar(
                    top_countries,
                    x='Participants',
                    y='Country',
                    orientation='h',
                    title='<b>Top 15 Countries by Participation</b>',
                    color='Participants',
                    color_continuous_scale='Blues'
                )
                fig2.update_layout(showlegend=False, height=500)
            else:
                fig2 = self._create_empty_chart("Geographic data unavailable")
        else:
            fig2 = self._create_empty_chart("Geographic data unavailable")
        
        # Geographic performance analysis
        if 'event_country' in data.columns and 'athlete_average_speed' in data.columns:
            geo_performance = data.groupby('event_country')['athlete_average_speed'].mean().reset_index()
            geo_performance = geo_performance.dropna()
            if not geo_performance.empty:
                top_performance = geo_performance.nlargest(10, 'athlete_average_speed')
                
                fig3 = px.bar(
                    top_performance,
                    x='athlete_average_speed',
                    y='event_country',
                    orientation='h',
                    title='<b>Top 10 Countries by Average Speed</b>',
                    color='athlete_average_speed',
                    color_continuous_scale='Greens'
                )
                fig3.update_layout(showlegend=False, height=400)
            else:
                fig3 = self._create_empty_chart("Performance by country data unavailable")
        else:
            fig3 = self._create_empty_chart("Performance by country data unavailable")
        
        return fig1, fig2, fig3
    
    def create_advanced_analytics(self, data):
        """Create advanced analytics visualizations"""
        # Performance by age group
        if 'age_group' in data.columns and 'athlete_average_speed' in data.columns:
            plot_data = data.dropna(subset=['age_group', 'athlete_average_speed'])
            if not plot_data.empty:
                fig1 = px.box(plot_data, x='age_group', y='athlete_average_speed',
                             title="<b>Performance by Age Group</b>")
            else:
                fig1 = self._create_empty_chart("Performance data unavailable")
        else:
            fig1 = self._create_empty_chart("Performance data unavailable")
        
        # Event difficulty analysis
        if 'event_difficulty' in data.columns:
            difficulty_data = data.dropna(subset=['event_difficulty'])
            if not difficulty_data.empty:
                difficulty_stats = difficulty_data['event_difficulty'].value_counts()
                fig2 = px.bar(x=difficulty_stats.index, y=difficulty_stats.values,
                             title="<b>Event Difficulty Distribution</b>",
                             color=difficulty_stats.values,
                             color_continuous_scale='Viridis')
            else:
                fig2 = self._create_empty_chart("Event data unavailable")
        else:
            fig2 = self._create_empty_chart("Event data unavailable")
        
        return fig1, fig2
    
    def _create_empty_chart(self, message):
        """Create placeholder for missing data"""
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16, color="gray"))
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False},
                         height=300)
        return fig

# ================================
# ENTERPRISE DASHBOARD LAYOUT
# ================================

def apply_enterprise_styling():
    """Apply professional enterprise styling"""
    st.markdown("""
        <style>
        .enterprise-header {
            font-size: 3.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 800;
        }
        .section-header {
            font-size: 1.8rem;
            color: #343a40;
            margin: 2.5rem 0 1.2rem 0;
            font-weight: 700;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.7rem;
        }
        .subsection-header {
            font-size: 1.4rem;
            color: #495057;
            margin: 2rem 0 1rem 0;
            font-weight: 600;
            border-left: 4px solid #667eea;
            padding-left: 1rem;
        }
        .kpi-card-enterprise {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .kpi-card-enterprise:hover {
            transform: translateY(-5px);
        }
        .kpi-title-enterprise {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .kpi-value-enterprise {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .kpi-subtitle-enterprise {
            font-size: 0.85rem;
            opacity: 0.8;
            font-weight: 500;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid #e9ecef;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin: 0.5rem 0;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-size: 1.5rem;
            color: #343a40;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

def create_enterprise_sidebar():
    """Create professional enterprise sidebar"""
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #667eea; font-weight: 800;'>🚀 ENTERPRISE CONTROL PANEL</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 NAVIGATION")
        page = st.radio("Select Dashboard", 
                       ["Executive Overview", "Advanced Analytics", "Performance Metrics", 
                        "Geographic Intelligence", "ML Insights", "Data Management"])
        
        st.markdown("---")
        
        # Analysis Controls
        st.markdown("### ⚙️ ANALYSIS SETTINGS")
        
        year_range = st.slider(
            "📅 Year Range Analysis",
            min_value=1980,
            max_value=2023,
            value=(2000, 2022)
        )
        
        analysis_focus = st.selectbox(
            "🎯 Primary Focus",
            ["Overall Performance", "Demographic Trends", "Geographic Distribution", 
             "Event Analysis", "Athlete Segmentation"]
        )
        
        st.markdown("---")
        
        # Enterprise Features
        st.markdown("### 🚀 ENTERPRISE FEATURES")
        
        col1, col2 = st.columns(2)
        with col1:
            real_time = st.checkbox("Live Updates", value=True)
        with col2:
            ml_insights = st.checkbox("AI Insights", value=True)
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 SYSTEM STATUS")
        st.success("✅ All Systems Operational")
        st.info(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return page, year_range, analysis_focus, real_time, ml_insights

def calculate_enterprise_kpis(data):
    """Calculate comprehensive enterprise KPIs"""
    kpis = {}
    
    if data is None:
        return kpis
    
    # Basic metrics
    kpis['total_athletes'] = len(data)
    kpis['total_events'] = data['year_of_event'].nunique() if 'year_of_event' in data.columns else 0
    kpis['date_range'] = f"{data['year_of_event'].min()}-{data['year_of_event'].max()}" if 'year_of_event' in data.columns else "N/A"
    
    # Performance metrics
    if 'athlete_average_speed' in data.columns:
        speed_data = data['athlete_average_speed'].dropna()
        if not speed_data.empty:
            kpis['avg_speed'] = speed_data.mean()
            kpis['speed_std'] = speed_data.std()
            kpis['max_speed'] = speed_data.max()
            kpis['min_speed'] = speed_data.min()
    
    # Demographic metrics
    if 'age' in data.columns:
        age_data = data['age'].dropna()
        if not age_data.empty:
            kpis['avg_age'] = age_data.mean()
            kpis['age_diversity'] = age_data.std()
            kpis['youngest'] = age_data.min()
            kpis['oldest'] = age_data.max()
    
    if 'athlete_gender' in data.columns:
        gender_data = data['athlete_gender'].dropna()
        if not gender_data.empty:
            gender_ratio = gender_data.mean()  # Assuming 1=Male, 0=Female
            kpis['male_percentage'] = gender_ratio * 100
            kpis['female_percentage'] = (1 - gender_ratio) * 100
    
    # Event metrics
    if 'event_distance/length' in data.columns:
        distance_data = data['event_distance/length'].dropna()
        if not distance_data.empty:
            kpis['avg_distance'] = distance_data.mean()
            kpis['max_distance'] = distance_data.max()
            kpis['min_distance'] = distance_data.min()
    
    # Geographic metrics
    if 'event_country' in data.columns:
        kpis['countries'] = data['event_country'].nunique()
    
    return kpis

def main():
    """Main enterprise dashboard application"""
    
    # Initialize enterprise components
    config = EnterpriseConfig()
    viz_engine = VisualizationEngine(config.THEMES['light'])
    
    # Page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Apply enterprise styling
    apply_enterprise_styling()
    
    # Enterprise Header
    st.markdown(f'<h1 class="enterprise-header">🏃‍♂️ {config.PAGE_TITLE}</h1>', 
                unsafe_allow_html=True)
    
    # Create sidebar and get controls
    page, year_range, analysis_focus, real_time, ml_insights = create_enterprise_sidebar()
    
    # Load data using static method
    with st.spinner('🚀 Initializing Enterprise Data Engine...'):
        file_paths = [
            'data analysis project/small_file.csv',
            'Data Analysis Project/small_file.csv',
            'small_file.csv'
        ]
        data = DataEngine.load_enterprise_data(file_paths)
    
    if data is None:
        st.error("""
        ❌ Enterprise data initialization failed. 
        Please ensure your data file is available in one of these locations:
        - `data analysis project/small_file.csv`
        - `Data Analysis Project/small_file.csv` 
        - `small_file.csv` (current directory)
        """)
        return
    
    # Calculate KPIs
    kpis = calculate_enterprise_kpis(data)
    
    # Filter data based on year range
    filtered_data = data[
        (data['year_of_event'] >= year_range[0]) & 
        (data['year_of_event'] <= year_range[1])
    ] if 'year_of_event' in data.columns else data
    
    # EXECUTIVE OVERVIEW PAGE
    if page == "Executive Overview":
        st.markdown('<div class="section-header">📊 EXECUTIVE DASHBOARD</div>', unsafe_allow_html=True)
        
        # Enterprise KPI Grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">👥 TOTAL ATHLETES</div>
                    <div class="kpi-value-enterprise">{kpis.get('total_athletes', 0):,}</div>
                    <div class="kpi-subtitle-enterprise">Enterprise Database</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">📅 TOTAL EVENTS</div>
                    <div class="kpi-value-enterprise">{kpis.get('total_events', 0):,}</div>
                    <div class="kpi-subtitle-enterprise">Global Coverage</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_speed = kpis.get('avg_speed', 0)
            speed_display = f"{avg_speed:.1f} km/h" if avg_speed > 0 else "N/A"
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">⚡ AVG SPEED</div>
                    <div class="kpi-value-enterprise">{speed_display}</div>
                    <div class="kpi-subtitle-enterprise">Performance Metric</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            male_pct = kpis.get('male_percentage', 0)
            gender_display = f"{male_pct:.1f}% Male" if male_pct > 0 else "N/A"
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">♂️ GENDER RATIO</div>
                    <div class="kpi-value-enterprise">{gender_display}</div>
                    <div class="kpi-subtitle-enterprise">Demographic Insight</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Second KPI Row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            avg_age = kpis.get('avg_age', 0)
            age_display = f"{avg_age:.1f} yrs" if avg_age > 0 else "N/A"
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">🎂 AVERAGE AGE</div>
                    <div class="kpi-value-enterprise">{age_display}</div>
                    <div class="kpi-subtitle-enterprise">Athlete Profile</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            avg_dist = kpis.get('avg_distance', 0)
            dist_display = f"{avg_dist:.1f} km" if avg_dist > 0 else "N/A"
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">📏 AVG DISTANCE</div>
                    <div class="kpi-value-enterprise">{dist_display}</div>
                    <div class="kpi-subtitle-enterprise">Event Difficulty</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col7:
            date_range = kpis.get('date_range', 'N/A')
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">📆 DATA RANGE</div>
                    <div class="kpi-value-enterprise">{date_range}</div>
                    <div class="kpi-subtitle-enterprise">Historical Coverage</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col8:
            countries = kpis.get('countries', 0)
            countries_display = f"{countries:,}" if countries > 0 else "N/A"
            st.markdown(f"""
                <div class="kpi-card-enterprise">
                    <div class="kpi-title-enterprise">🌍 COUNTRIES</div>
                    <div class="kpi-value-enterprise">{countries_display}</div>
                    <div class="kpi-subtitle-enterprise">Global Reach</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Executive Summary Dashboard
        st.markdown('<div class="section-header">📈 EXECUTIVE SUMMARY DASHBOARD</div>', unsafe_allow_html=True)
        exec_fig = viz_engine.create_executive_summary(filtered_data, kpis)
        st.plotly_chart(exec_fig, use_container_width=True, key="executive_summary")
    
    # PERFORMANCE METRICS PAGE
    elif page == "Performance Metrics":
        st.markdown('<div class="section-header">⚡ PERFORMANCE METRICS DASHBOARD</div>', unsafe_allow_html=True)
        
        # Performance KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Speed</div>
                    <div class="metric-value">{kpis.get('avg_speed', 0):.1f} km/h</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Performance Consistency</div>
                    <div class="metric-value">{kpis.get('speed_std', 0):.2f} std</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Max Speed</div>
                    <div class="metric-value">{kpis.get('max_speed', 0):.1f} km/h</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Min Speed</div>
                    <div class="metric-value">{kpis.get('min_speed', 0):.1f} km/h</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Performance Charts
        st.markdown('<div class="subsection-header">📊 Performance Analysis</div>', unsafe_allow_html=True)
        fig1, fig2, fig3 = viz_engine.create_performance_metrics(filtered_data)
        
        st.plotly_chart(fig1, use_container_width=True, key="performance_evolution")
        
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(fig2, use_container_width=True, key="performance_tiers")
        with col6:
            st.plotly_chart(fig3, use_container_width=True, key="finish_rates")
    
    # GEOGRAPHIC INTELLIGENCE PAGE
    elif page == "Geographic Intelligence":
        st.markdown('<div class="section-header">🌍 GEOGRAPHIC INTELLIGENCE DASHBOARD</div>', unsafe_allow_html=True)
        
        # Geographic KPIs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Countries Covered</div>
                    <div class="metric-value">{kpis.get('countries', 0):,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'event_country' in filtered_data.columns:
                top_country = filtered_data['event_country'].value_counts().index[0] if not filtered_data['event_country'].value_counts().empty else "N/A"
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Top Country</div>
                        <div class="metric-value">{top_country}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'event_country' in filtered_data.columns:
                country_diversity = filtered_data['event_country'].nunique()
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Geographic Diversity</div>
                        <div class="metric-value">{country_diversity}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Geographic Charts
        st.markdown('<div class="subsection-header">🗺️ Global Participation Analysis</div>', unsafe_allow_html=True)
        fig1, fig2, fig3 = viz_engine.create_geographic_intelligence(filtered_data)
        
        st.plotly_chart(fig1, use_container_width=True, key="global_heatmap")
        
        col4, col5 = st.columns(2)
        with col4:
            st.plotly_chart(fig2, use_container_width=True, key="top_countries")
        with col5:
            st.plotly_chart(fig3, use_container_width=True, key="country_performance")
    
    # ADVANCED ANALYTICS PAGE
    elif page == "Advanced Analytics":
        st.markdown('<div class="section-header">🔬 ADVANCED ANALYTICS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, fig2 = viz_engine.create_advanced_analytics(filtered_data)
            st.plotly_chart(fig1, use_container_width=True, key="age_performance")
        with col2:
            st.plotly_chart(fig2, use_container_width=True, key="event_difficulty")
    
    # ML INSIGHTS PAGE
    elif page == "ML Insights" and ml_insights:
        st.markdown('<div class="section-header">🤖 MACHINE LEARNING INSIGHTS</div>', unsafe_allow_html=True)
        
        insights = DataEngine.get_ml_insights(filtered_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Performance Analytics")
            for insight in insights.get('performance_trend', []):
                st.info(f"🔍 {insight}")
        
        with col2:
            st.markdown("### 🔮 Predictive Insights")
            for insight in insights.get('participation_forecast', []):
                st.success(f"🎯 {insight}")
        
        st.markdown("### 🎯 Demographic Intelligence")
        for insight in insights.get('demographic_shifts', []):
            st.warning(f"📊 {insight}")
    
    # DATA MANAGEMENT PAGE
    elif page == "Data Management":
        st.markdown('<div class="section-header">💾 ENTERPRISE DATA MANAGEMENT</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Export")
            if st.button("🚀 Generate Comprehensive Report", use_container_width=True, key="export_btn"):
                csv_data = filtered_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Enterprise Data",
                    data=csv_data,
                    file_name=f"ultra_marathons_enterprise_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_btn"
                )
        
        with col2:
            st.markdown("### 🔍 Data Quality Report")
            if st.button("📋 Generate Quality Report", use_container_width=True, key="quality_btn"):
                with st.expander("Data Quality Metrics", expanded=True):
                    st.write(f"**Dataset Overview:**")
                    st.write(f"- Total Records: {len(filtered_data):,}")
                    
                    complete_records = len(filtered_data.dropna())
                    completeness = complete_records/len(filtered_data)*100
                    st.write(f"- Data Completeness: {completeness:.1f}%")
                    
                    if 'year_of_event' in filtered_data.columns:
                        st.write(f"- Date Range: {filtered_data['year_of_event'].min()}-{filtered_data['year_of_event'].max()}")
                    
                    missing_data = filtered_data.isnull().sum()
                    if missing_data.any():
                        st.write("**Data Quality Issues:**")
                        for col, count in missing_data[missing_data > 0].items():
                            st.write(f"- {col}: {count:,} missing ({count/len(filtered_data)*100:.1f}%)")

    # Real-time updates simulation
    if real_time:
        st.markdown("---")
        st.markdown(f"*🔄 Live updates enabled • Last refresh: {datetime.now().strftime('%H:%M:%S')}*")

if __name__ == "__main__":
    main()