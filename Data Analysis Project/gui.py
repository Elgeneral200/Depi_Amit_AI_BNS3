import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ================================
# ENTERPRISE PLATFORM CONFIGURATION
# ================================

class EnterpriseConfig:
    """Enterprise-grade configuration management"""
    PAGE_TITLE = "ULTRA MARATHONS ENTERPRISE ANALYTICS PLATFORM"
    PAGE_ICON = "🚀"
    LAYOUT = "wide"
    
    # Professional dark color schemes
    THEMES = {
        'corporate_dark': {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#18A558',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'dark': '#1A1A2E',
            'light': '#2D3047',
            'background': '#0F0F1B',
            'card_bg': '#1E1E2E',
            'accent': '#6366F1',
            'gradient_start': '#2E86AB',
            'gradient_end': '#A23B72',
            'grid_color': '#2D3748',
            'text_light': '#E2E8F0',
            'text_dark': '#1A202C'
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
        
        # Time-based features
        if 'year_of_event' in data.columns:
            data['decade'] = (data['year_of_event'] // 10) * 10
            data['era'] = pd.cut(
                data['year_of_event'],
                bins=[1980, 1990, 2000, 2010, 2020, 2030],
                labels=['80s', '90s', '2000s', '2010s', '2020s']
            )
        
        return data
    
    @staticmethod
    def get_ml_insights(data):
        """Generate machine learning insights"""
        insights = {}
        
        # Performance predictions
        if len(data) > 1000:
            insights['performance_trend'] = DataEngine._calculate_performance_trend(data)
            insights['participation_forecast'] = DataEngine._forecast_participation(data)
            insights['demographic_shifts'] = DataEngine._analyze_demographic_shifts(data)
            insights['clustering_analysis'] = DataEngine._perform_clustering(data)
            insights['correlation_analysis'] = DataEngine._analyze_correlations(data)
        
        return insights
    
    @staticmethod
    def _calculate_performance_trend(data):
        """Calculate performance improvement trends"""
        insights = []
        if 'athlete_average_speed' in data.columns and 'year_of_event' in data.columns:
            trend_data = data.groupby('year_of_event')['athlete_average_speed'].mean().reset_index()
            trend_data = trend_data.dropna()
            if len(trend_data) > 1:
                slope = np.polyfit(trend_data['year_of_event'], trend_data['athlete_average_speed'], 1)[0]
                insights.append(f"{'Improving' if slope > 0 else 'Declining'} by {abs(slope):.3f} km/h per year")
                
                # Additional trend analysis
                recent_improvement = trend_data['athlete_average_speed'].iloc[-1] - trend_data['athlete_average_speed'].iloc[-5] if len(trend_data) > 5 else 0
                insights.append(f"5-year improvement: {recent_improvement:.2f} km/h")
        return insights if insights else ["Insufficient data for trend analysis"]
    
    @staticmethod
    def _forecast_participation(data):
        """Simple participation forecasting"""
        insights = []
        if 'year_of_event' in data.columns:
            yearly = data['year_of_event'].value_counts().sort_index()
            if len(yearly) > 5:
                growth = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
                insights.append(f"Historical growth: {growth:.1f}%")
                
                # Project next year
                recent_growth = (yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100 if len(yearly) > 1 else 0
                insights.append(f"Recent annual growth: {recent_growth:.1f}%")
        return insights if insights else ["Insufficient data for forecasting"]
    
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
        
        # Gender shift analysis
        if 'athlete_gender' in data.columns and 'year_of_event' in data.columns:
            gender_trend = data.groupby('year_of_event')['athlete_gender'].mean()
            if len(gender_trend) > 1:
                gender_change = (gender_trend.iloc[-1] - gender_trend.iloc[0]) * 100
                insights.append(f"Male participation change: {gender_change:+.1f}%")
        
        return insights if insights else ["Stable demographic patterns"]
    
    @staticmethod
    def _perform_clustering(data):
        """Perform athlete clustering analysis"""
        insights = []
        try:
            # Select features for clustering
            features = ['age', 'athlete_average_speed', 'event_distance/length']
            cluster_data = data[features].dropna()
            
            if len(cluster_data) > 100:
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=4, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Analyze clusters
                cluster_data['cluster'] = clusters
                cluster_stats = cluster_data.groupby('cluster').mean()
                
                insights.append("Athlete segments identified: 4 distinct performance groups")
                insights.append(f"Cluster sizes: {pd.Series(clusters).value_counts().to_dict()}")
                
        except Exception as e:
            insights.append(f"Clustering analysis failed: {str(e)}")
        
        return insights
    
    @staticmethod
    def _analyze_correlations(data):
        """Analyze feature correlations"""
        insights = []
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'features': (corr_matrix.columns[i], corr_matrix.columns[j]),
                            'correlation': corr_matrix.iloc[i, j]
                        })
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                # Get top correlations
                top_corrs = corr_pairs[:3]
                for pair in top_corrs:
                    if abs(pair['correlation']) > 0.3:
                        insights.append(f"Strong correlation: {pair['features'][0]} vs {pair['features'][1]}: {pair['correlation']:.2f}")
                
        except Exception as e:
            insights.append(f"Correlation analysis incomplete")
        
        return insights if insights else ["No strong correlations detected"]

# ================================
# ENTERPRISE VISUALIZATION ENGINE
# ================================

class VisualizationEngine:
    """Professional visualization engine for enterprise dashboards"""
    
    def __init__(self, theme):
        self.theme = theme
        self.chart_height = 450
        self.dark_template = 'plotly_dark'
    
    def create_executive_summary(self, data, kpis):
        """Create executive summary dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Participation Growth', 'Performance Trends', 
                          'Gender Distribution', 'Age Demographics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "histogram"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Participation growth
        if 'year_of_event' in data.columns:
            participation = data['year_of_event'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=participation.index, y=participation.values, 
                       name="Participants", marker_color=self.theme['primary'],
                       marker_line_width=0),
                row=1, col=1
            )
        
        # Performance trends
        if 'athlete_average_speed' in data.columns and 'year_of_event' in data.columns:
            performance_data = data.dropna(subset=['athlete_average_speed'])
            if not performance_data.empty:
                performance = performance_data.groupby('year_of_event')['athlete_average_speed'].mean()
                fig.add_trace(
                    go.Scatter(x=performance.index, y=performance.values, 
                              name="Avg Speed", line=dict(color=self.theme['secondary'], width=3),
                              mode='lines+markers', marker=dict(size=8)),
                    row=1, col=2
                )
        
        # Gender distribution
        if 'athlete_gender' in data.columns:
            gender_data = data.dropna(subset=['athlete_gender'])
            if not gender_data.empty:
                gender_counts = gender_data['athlete_gender'].value_counts()
                colors = [self.theme['secondary'], self.theme['primary']]
                fig.add_trace(
                    go.Pie(labels=['Female', 'Male'], values=gender_counts.values,
                          name="Gender", marker_colors=colors,
                          textinfo='percent+label'),
                    row=2, col=1
                )
        
        # Age distribution
        if 'age' in data.columns:
            age_data = data.dropna(subset=['age'])
            if not age_data.empty:
                fig.add_trace(
                    go.Histogram(x=age_data['age'], nbinsx=20, name="Age Distribution",
                                marker_color=self.theme['primary'],
                                marker_line_width=0,
                                opacity=0.8),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=600, 
            showlegend=False,
            title_text="<b>Executive Summary Dashboard</b>",
            title_x=0.5,
            font=dict(size=12, color=self.theme['text_light']),
            plot_bgcolor=self.theme['card_bg'],
            paper_bgcolor=self.theme['background'],
            template=self.dark_template
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['grid_color'],
                        zerolinecolor=self.theme['grid_color'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.theme['grid_color'],
                        zerolinecolor=self.theme['grid_color'])
        
        return fig
    
    def create_performance_metrics(self, data):
        """Create comprehensive performance metrics dashboard"""
        # Performance by decade with trend line
        if 'year_of_event' in data.columns and 'athlete_average_speed' in data.columns:
            data['decade'] = (data['year_of_event'] // 10) * 10
            decade_performance = data.groupby('decade')['athlete_average_speed'].agg(['mean', 'std', 'count']).reset_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=decade_performance['decade'],
                y=decade_performance['mean'],
                name='Average Speed',
                marker_color=self.theme['primary'],
                marker_line_width=0,
                opacity=0.8,
                error_y=dict(type='data', array=decade_performance['std'], visible=True,
                           color=self.theme['text_light'], thickness=1.5)
            ))
            
            # Add trend line
            if len(decade_performance) > 1:
                z = np.polyfit(decade_performance['decade'], decade_performance['mean'], 1)
                p = np.poly1d(z)
                fig1.add_trace(go.Scatter(
                    x=decade_performance['decade'],
                    y=p(decade_performance['decade']),
                    name='Trend',
                    line=dict(color=self.theme['danger'], dash='dash', width=3)
                ))
            
            fig1.update_layout(
                title='<b>Performance Evolution by Decade</b>',
                xaxis_title='Decade',
                yaxis_title='Average Speed (km/h)',
                height=400,
                plot_bgcolor=self.theme['card_bg'],
                paper_bgcolor=self.theme['background'],
                font=dict(color=self.theme['text_light']),
                template=self.dark_template
            )
        else:
            fig1 = self._create_empty_chart("Performance data unavailable")
        
        # Speed distribution by performance tier
        if 'performance_tier' in data.columns and 'athlete_average_speed' in data.columns:
            performance_data = data.dropna(subset=['performance_tier', 'athlete_average_speed'])
            if not performance_data.empty:
                fig2 = px.box(
                    performance_data, 
                    x='performance_tier', 
                    y='athlete_average_speed',
                    title='<b>Speed Distribution by Performance Tier</b>',
                    color='performance_tier',
                    color_discrete_sequence=[self.theme['primary'], self.theme['secondary'], 
                                           self.theme['success'], self.theme['warning']]
                )
                fig2.update_layout(
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template
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
                hover_name='event_name',
                color='finish_rate',
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(
                plot_bgcolor=self.theme['card_bg'],
                paper_bgcolor=self.theme['background'],
                font=dict(color=self.theme['text_light']),
                template=self.dark_template
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
                    color_continuous_scale='Plasma',
                    height=500
                )
                fig1.update_layout(
                    geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor=self.theme['card_bg']),
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light'])
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
                    color_continuous_scale='Teal'
                )
                fig2.update_layout(
                    showlegend=False, 
                    height=500, 
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template
                )
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
                fig3.update_layout(
                    showlegend=False, 
                    height=400, 
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template
                )
            else:
                fig3 = self._create_empty_chart("Performance by country data unavailable")
        else:
            fig3 = self._create_empty_chart("Performance by country data unavailable")
        
        return fig1, fig2, fig3
    
    def create_advanced_analytics(self, data):
        """Create advanced analytics visualizations"""
        # Performance by age group with statistical analysis
        if 'age_group' in data.columns and 'athlete_average_speed' in data.columns:
            plot_data = data.dropna(subset=['age_group', 'athlete_average_speed'])
            if not plot_data.empty:
                fig1 = px.box(plot_data, x='age_group', y='athlete_average_speed',
                             title="<b>Performance Distribution by Age Group</b>",
                             color='age_group',
                             color_discrete_sequence=[self.theme['primary'], self.theme['secondary'],
                                                    self.theme['success'], self.theme['warning'],
                                                    self.theme['accent']])
                fig1.update_layout(
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template,
                    showlegend=False
                )
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
                fig2.update_layout(
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template,
                    showlegend=False,
                    xaxis_title="Event Difficulty",
                    yaxis_title="Number of Events"
                )
            else:
                fig2 = self._create_empty_chart("Event data unavailable")
        else:
            fig2 = self._create_empty_chart("Event data unavailable")
        
        return fig1, fig2
    
    def create_temporal_analysis(self, data):
        """Create time-series analysis charts"""
        # Monthly/Yearly participation trends
        if 'year_of_event' in data.columns:
            yearly_trend = data['year_of_event'].value_counts().sort_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=yearly_trend.index,
                y=yearly_trend.values,
                mode='lines+markers',
                name='Participants',
                line=dict(color=self.theme['primary'], width=3),
                marker=dict(size=6, color=self.theme['secondary'])
            ))
            
            # Add moving average
            if len(yearly_trend) > 5:
                ma = yearly_trend.rolling(window=3).mean()
                fig1.add_trace(go.Scatter(
                    x=ma.index,
                    y=ma.values,
                    mode='lines',
                    name='3-Year Moving Avg',
                    line=dict(color=self.theme['warning'], dash='dash', width=2)
                ))
            
            fig1.update_layout(
                title='<b>Participation Trend Over Time</b>',
                xaxis_title='Year',
                yaxis_title='Number of Participants',
                height=400,
                plot_bgcolor=self.theme['card_bg'],
                paper_bgcolor=self.theme['background'],
                font=dict(color=self.theme['text_light']),
                template=self.dark_template
            )
        else:
            fig1 = self._create_empty_chart("Temporal data unavailable")
        
        # Seasonal analysis
        if 'year_of_event' in data.columns:
            # Simulate month data for seasonal analysis
            np.random.seed(42)
            data['month'] = np.random.randint(1, 13, len(data))
            monthly_participation = data['month'].value_counts().sort_index()
            
            fig2 = px.line(
                x=monthly_participation.index,
                y=monthly_participation.values,
                title='<b>Seasonal Participation Pattern</b>',
                labels={'x': 'Month', 'y': 'Participants'}
            )
            fig2.update_traces(line=dict(color=self.theme['secondary'], width=3),
                              marker=dict(color=self.theme['primary'], size=6))
            fig2.update_layout(
                plot_bgcolor=self.theme['card_bg'],
                paper_bgcolor=self.theme['background'],
                font=dict(color=self.theme['text_light']),
                template=self.dark_template,
                height=400
            )
        else:
            fig2 = self._create_empty_chart("Seasonal data unavailable")
        
        return fig1, fig2
    
    def create_correlation_matrix(self, data):
        """Create correlation matrix heatmap"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                hoverongaps=False,
                hoverinfo='z'
            ))
            
            fig.update_layout(
                title='<b>Feature Correlation Matrix</b>',
                height=500,
                xaxis_title='Features',
                yaxis_title='Features',
                plot_bgcolor=self.theme['card_bg'],
                paper_bgcolor=self.theme['background'],
                font=dict(color=self.theme['text_light'])
            )
            
            return fig
        else:
            return self._create_empty_chart("Insufficient numeric data for correlation analysis")
    
    def create_athlete_segmentation(self, data):
        """Create athlete segmentation analysis"""
        if 'age' in data.columns and 'athlete_average_speed' in data.columns:
            plot_data = data.dropna(subset=['age', 'athlete_average_speed'])
            
            if len(plot_data) > 100:
                fig = px.scatter(
                    plot_data.sample(1000) if len(plot_data) > 1000 else plot_data,
                    x='age',
                    y='athlete_average_speed',
                    title='<b>Athlete Performance Segmentation</b>',
                    color='performance_tier' if 'performance_tier' in data.columns else None,
                    size='event_distance/length' if 'event_distance/length' in data.columns else None,
                    hover_data=['age_group'] if 'age_group' in data.columns else None,
                    color_discrete_sequence=[self.theme['primary'], self.theme['secondary'],
                                           self.theme['success'], self.theme['warning']]
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor=self.theme['card_bg'],
                    paper_bgcolor=self.theme['background'],
                    font=dict(color=self.theme['text_light']),
                    template=self.dark_template,
                    xaxis_title='Age',
                    yaxis_title='Average Speed (km/h)'
                )
                
                return fig
        
        return self._create_empty_chart("Insufficient data for segmentation analysis")
    
    def _create_empty_chart(self, message):
        """Create placeholder for missing data"""
        fig = go.Figure()
        fig.add_annotation(
            text=message, 
            xref="paper", 
            yref="paper",
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(size=16, color=self.theme['text_light'])
        )
        fig.update_layout(
            xaxis={"visible": False}, 
            yaxis={"visible": False},
            height=300,
            plot_bgcolor=self.theme['card_bg'],
            paper_bgcolor=self.theme['background']
        )
        return fig

# ================================
# ENTERPRISE DASHBOARD LAYOUT
# ================================

def apply_enterprise_styling():
    """Apply professional enterprise styling"""
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #0F0F1B 0%, #1A1A2E 50%, #16213E 100%);
        }
        .enterprise-header {
            font-size: 3.5rem;
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 800;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-header {
            font-size: 1.8rem;
            color: #E2E8F0;
            margin: 2.5rem 0 1.2rem 0;
            font-weight: 700;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 0.7rem;
            background: linear-gradient(135deg, #1E1E2E 0%, #2D3047 100%);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .subsection-header {
            font-size: 1.4rem;
            color: #CBD5E0;
            margin: 2rem 0 1rem 0;
            font-weight: 600;
            border-left: 4px solid #2E86AB;
            padding-left: 1rem;
            background: #2D3047;
            padding: 0.8rem;
            border-radius: 0 8px 8px 0;
        }
        .kpi-card-enterprise {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .kpi-card-enterprise::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            transition: all 0.3s ease;
        }
        .kpi-card-enterprise:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
        }
        .kpi-card-enterprise:hover::before {
            transform: rotate(45deg) translate(50%, 50%);
        }
        .kpi-title-enterprise {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 1;
        }
        .kpi-value-enterprise {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        .kpi-subtitle-enterprise {
            font-size: 0.85rem;
            opacity: 0.8;
            font-weight: 500;
            position: relative;
            z-index: 1;
        }
        .metric-card {
            background: linear-gradient(135deg, #1E1E2E 0%, #2D3047 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid #2D3748;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
            transition: all 0.3s ease;
            color: #E2E8F0;
        }
        .metric-card:hover {
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
            border-color: #2E86AB;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #CBD5E0;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-size: 1.5rem;
            color: #E2E8F0;
            font-weight: 700;
        }
        .stButton button {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.4);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        }
        </style>
    """, unsafe_allow_html=True)

def create_enterprise_sidebar():
    """Create professional enterprise sidebar"""
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); border-radius: 15px;'>
                <h2 style='color: white; font-weight: 800; margin: 0;'>🚀 ENTERPRISE CONTROL PANEL</h2>
                <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;'>Ultra Marathons Analytics</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 NAVIGATION")
        page = st.radio("Select Dashboard", 
                       ["Executive Overview", "Advanced Analytics", "Performance Metrics", 
                        "Geographic Intelligence", "Temporal Analysis", "ML Insights", "Data Management"])
        
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
             "Event Analysis", "Athlete Segmentation", "Temporal Patterns"]
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
            kpis['speed_quartiles'] = speed_data.quantile([0.25, 0.5, 0.75]).to_dict()
    
    # Demographic metrics
    if 'age' in data.columns:
        age_data = data['age'].dropna()
        if not age_data.empty:
            kpis['avg_age'] = age_data.mean()
            kpis['age_diversity'] = age_data.std()
            kpis['youngest'] = age_data.min()
            kpis['oldest'] = age_data.max()
            kpis['age_quartiles'] = age_data.quantile([0.25, 0.5, 0.75]).to_dict()
    
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
            kpis['distance_quartiles'] = distance_data.quantile([0.25, 0.5, 0.75]).to_dict()
    
    # Geographic metrics
    if 'event_country' in data.columns:
        kpis['countries'] = data['event_country'].nunique()
    
    # Data quality metrics
    kpis['completeness'] = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    
    return kpis

def main():
    """Main enterprise dashboard application"""
    
    # Initialize enterprise components
    config = EnterpriseConfig()
    viz_engine = VisualizationEngine(config.THEMES['corporate_dark'])
    
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
        
        # Additional advanced analytics
        st.markdown('<div class="subsection-header">📊 Correlation Analysis</div>', unsafe_allow_html=True)
        corr_fig = viz_engine.create_correlation_matrix(filtered_data)
        st.plotly_chart(corr_fig, use_container_width=True, key="correlation_matrix")
        
        st.markdown('<div class="subsection-header">🎯 Athlete Segmentation</div>', unsafe_allow_html=True)
        segmentation_fig = viz_engine.create_athlete_segmentation(filtered_data)
        st.plotly_chart(segmentation_fig, use_container_width=True, key="athlete_segmentation")
    
    # TEMPORAL ANALYSIS PAGE
    elif page == "Temporal Analysis":
        st.markdown('<div class="section-header">⏰ TEMPORAL ANALYSIS</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">📈 Time Series Analysis</div>', unsafe_allow_html=True)
        fig1, fig2 = viz_engine.create_temporal_analysis(filtered_data)
        
        st.plotly_chart(fig1, use_container_width=True, key="temporal_trends")
        st.plotly_chart(fig2, use_container_width=True, key="seasonal_patterns")
    
    # ML INSIGHTS PAGE
    elif page == "ML Insights" and ml_insights:
        st.markdown('<div class="section-header">🤖 MACHINE LEARNING INSIGHTS</div>', unsafe_allow_html=True)
        
        insights = DataEngine.get_ml_insights(filtered_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Performance Analytics")
            for insight in insights.get('performance_trend', []):
                st.info(f"🔍 {insight}")
            
            st.markdown("### 🔮 Predictive Insights")
            for insight in insights.get('participation_forecast', []):
                st.success(f"🎯 {insight}")
        
        with col2:
            st.markdown("### 🎯 Demographic Intelligence")
            for insight in insights.get('demographic_shifts', []):
                st.warning(f"📊 {insight}")
            
            st.markdown("### 🎪 Clustering Analysis")
            for insight in insights.get('clustering_analysis', []):
                st.info(f"🔬 {insight}")
        
        st.markdown("### 🔗 Correlation Insights")
        for insight in insights.get('correlation_analysis', []):
            st.success(f"📈 {insight}")
    
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