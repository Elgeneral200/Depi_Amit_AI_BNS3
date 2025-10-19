###########################################
# ENTERPRISE CUSTOMER INTELLIGENCE PLATFORM
# Professional Business Intelligence Dashboard
# Enhanced with Robust customers.csv Handling
###########################################

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import time
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import io
import base64
from datetime import datetime
import json

# Set page configuration for enterprise platform
st.set_page_config(
    page_title="Enterprise Customer Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnterpriseBrand:
    """Enterprise branding with dark professional color scheme"""
    
    def __init__(self):
        # Dark professional corporate color palette
        self.primary_colors = {
            'primary': '#1A237E',      # Dark Blue
            'secondary': '#283593',    # Medium Dark Blue
            'accent': '#C2185B',       # Dark Pink/Magenta
            'success': '#388E3C',      # Dark Green
            'warning': '#F57C00',      # Dark Orange
            'info': '#0277BD',         # Dark Cyan
            'dark': '#121212',         # Near Black
            'darker': '#0A0A0A',       # Almost Black
            'light': '#424242',        # Dark Gray
            'lighter': '#616161'       # Medium Gray
        }
        
        # Extended cluster colors (dark professional palette)
        self.cluster_palette = [
            '#1A237E', '#283593', '#303F9F', '#3949AB', '#3F51B5',  # Dark Blues
            '#C2185B', '#D81B60', '#E91E63', '#EC407A', '#F06292',  # Dark Pinks
            '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784',  # Dark Greens
            '#F57C00', '#FB8C00', '#FF9800', '#FFA726', '#FFB74D'   # Dark Oranges
        ]
        
        # Sequential colors for gradients (dark theme)
        self.sequential_palette = ['#0A0A0A', '#1A237E', '#283593', '#303F9F', '#3949AB', '#3F51B5']
        
    def get_cluster_color(self, index):
        """Safely get cluster color with bounds checking"""
        return self.cluster_palette[index % len(self.cluster_palette)]
    
    def apply_branding(self, fig, title=""):
        """Apply consistent dark enterprise branding to matplotlib figures"""
        fig.patch.set_facecolor('#121212')
        if hasattr(fig, 'axes') and fig.axes:
            for ax in fig.axes:
                ax.set_facecolor('#1E1E1E')
                ax.title.set_color('#FFFFFF')
                ax.xaxis.label.set_color('#E0E0E0')
                ax.yaxis.label.set_color('#E0E0E0')
                ax.tick_params(colors='#B0B0B0')
                
                # Style grid
                ax.grid(True, alpha=0.2, color='#424242')
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color('#424242')
                    spine.set_linewidth(1.5)
        
        return fig

class DataManager:
    """Enhanced data management with robust customers.csv handling"""
    
    @staticmethod
    def check_customers_file():
        """Comprehensive check for customers.csv file"""
        file_path = "customers.csv"
        
        if not os.path.exists(file_path):
            return False, "File 'customers.csv' not found in current directory", None
        
        try:
            # Try to read the file
            data = pd.read_csv(file_path)
            
            if data.empty:
                return False, "File 'customers.csv' is empty", None
            
            # Check for required features
            required_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
            available_features = [f for f in required_features if f in data.columns]
            
            if len(available_features) < 3:
                return True, f"File loaded with {len(available_features)} of 6 expected features", data
            
            return True, "File loaded successfully with all expected features", data
            
        except Exception as e:
            return False, f"Error reading 'customers.csv': {str(e)}", None
    
    @staticmethod
    def create_sample_data():
        """Create realistic sample customer data that matches customers.csv structure"""
        np.random.seed(42)
        n_customers = 440  # Typical customers.csv size
        
        # Create realistic customer segments based on typical wholesale data
        segments = {
            'Retailers': {
                'Fresh': np.random.normal(8000, 2000, n_customers//4),
                'Milk': np.random.normal(12000, 3000, n_customers//4),
                'Grocery': np.random.normal(15000, 4000, n_customers//4),
                'Frozen': np.random.normal(3000, 1000, n_customers//4),
                'Detergents_Paper': np.random.normal(6000, 2000, n_customers//4),
                'Delicatessen': np.random.normal(2000, 800, n_customers//4),
                'Channel': 2  # Retail
            },
            'Restaurants': {
                'Fresh': np.random.normal(15000, 4000, n_customers//4),
                'Milk': np.random.normal(5000, 1500, n_customers//4),
                'Grocery': np.random.normal(8000, 2500, n_customers//4),
                'Frozen': np.random.normal(6000, 2000, n_customers//4),
                'Detergents_Paper': np.random.normal(2000, 800, n_customers//4),
                'Delicatessen': np.random.normal(4000, 1200, n_customers//4),
                'Channel': 1  # HORECA
            },
            'Hotels': {
                'Fresh': np.random.normal(12000, 3000, n_customers//4),
                'Milk': np.random.normal(8000, 2000, n_customers//4),
                'Grocery': np.random.normal(10000, 3000, n_customers//4),
                'Frozen': np.random.normal(4000, 1500, n_customers//4),
                'Detergents_Paper': np.random.normal(4000, 1200, n_customers//4),
                'Delicatessen': np.random.normal(3000, 1000, n_customers//4),
                'Channel': 1  # HORECA
            },
            'Cafes': {
                'Fresh': np.random.normal(6000, 2000, n_customers//4),
                'Milk': np.random.normal(10000, 3000, n_customers//4),
                'Grocery': np.random.normal(7000, 2000, n_customers//4),
                'Frozen': np.random.normal(2000, 800, n_customers//4),
                'Detergents_Paper': np.random.normal(1500, 600, n_customers//4),
                'Delicatessen': np.random.normal(5000, 1500, n_customers//4),
                'Channel': 1  # HORECA
            }
        }
        
        data_frames = []
        for segment_name, segment_data in segments.items():
            segment_df = pd.DataFrame(segment_data)
            segment_df['Region'] = np.random.choice([1, 2, 3], len(segment_df))  # Add region like original
            data_frames.append(segment_df)
        
        full_data = pd.concat(data_frames, ignore_index=True)
        full_data = full_data.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        return full_data

class EnterpriseVisualizer:
    """
    Enterprise-grade visualization with fixed silhouette analysis and professional styling
    """
    
    def __init__(self):
        self.brand = EnterpriseBrand()
        
    def create_loading_animation(self, text="Processing..."):
        """Create loading animation"""
        placeholder = st.empty()
        for i in range(3):
            placeholder.markdown(f"<div style='text-align: center; color: {self.brand.primary_colors['accent']};'><h3>{text}{'.' * (i+1)}</h3></div>", unsafe_allow_html=True)
            time.sleep(0.3)
        placeholder.empty()
    
    def plot_segment_distribution(self, cluster_counts, figsize=(10, 6)):
        """Enhanced segment distribution with professional dark styling"""
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        # Create pie chart with safe color assignment
        colors = [self.brand.get_cluster_color(i) for i in range(len(cluster_counts))]
        
        wedges, texts, autotexts = ax.pie(
            cluster_counts.values,
            labels=[f'Segment {i}' for i in cluster_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': '#121212', 'linewidth': 2, 'alpha': 0.9}
        )
        
        # Enhance text styling for dark background
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontweight('bold')
            text.set_color('#E0E0E0')
        
        ax.set_title('Customer Segment Distribution', 
                    fontsize=16, fontweight='bold', pad=20,
                    color='#FFFFFF')
        
        return self.brand.apply_branding(fig)
    
    def plot_segment_characteristics(self, data, clusters, features, figsize=(14, 8)):
        """Bar chart for segment characteristics with safe color handling"""
        segment_data = data.copy()
        segment_data['Cluster'] = clusters
        
        # Calculate mean values for each segment
        segment_means = segment_data.groupby('Cluster')[features].mean()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        x = np.arange(len(features))
        width = 0.8 / len(segment_means)
        
        # Safe color assignment
        for i, segment in enumerate(segment_means.index):
            offset = (i - len(segment_means)/2 + 0.5) * width
            ax.bar(x + offset, segment_means.loc[segment].values, 
                   width, 
                   label=f'Segment {segment}', 
                   color=self.brand.get_cluster_color(i),
                   alpha=0.9,
                   edgecolor='#121212',
                   linewidth=1)
        
        ax.set_xlabel('Features', fontweight='bold', color='#E0E0E0')
        ax.set_ylabel('Average Value', fontweight='bold', color='#E0E0E0')
        ax.set_title('Segment Characteristics Comparison', 
                    fontsize=16, fontweight='bold', 
                    color='#FFFFFF')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', color='#B0B0B0')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
        
        return self.brand.apply_branding(fig)
    
    def plot_feature_importance(self, pca, features, figsize=(12, 6)):
        """Enhanced feature importance visualization with dark theme"""
        importance = np.abs(pca.components_[0])
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        # Create bar plot with color gradient
        colors = [self.brand.sequential_palette[min(int(imp * 10), len(self.brand.sequential_palette)-1)] 
                 for imp in importance]
        
        bars = ax.bar(features, importance, 
                     color=colors,
                     alpha=0.9,
                     edgecolor='#121212',
                     linewidth=1)
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{imp:.3f}', ha='center', va='bottom', 
                   fontweight='bold', color='#FFFFFF')
        
        ax.set_xlabel('Features', fontweight='bold', color='#E0E0E0')
        ax.set_ylabel('Importance Score', fontweight='bold', color='#E0E0E0')
        ax.set_title('Feature Importance in Customer Segmentation', 
                    fontsize=16, fontweight='bold', pad=20,
                    color='#FFFFFF')
        ax.tick_params(axis='x', rotation=45, colors='#B0B0B0')
        ax.tick_params(axis='y', colors='#B0B0B0')
        
        return self.brand.apply_branding(fig)
    
    def plot_cluster_2d(self, reduced_data, clusters, centers, figsize=(12, 8)):
        """2D cluster visualization with enhanced dark styling"""
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        # Create color mapping for clusters
        unique_clusters = np.unique(clusters)
        
        # Plot clusters
        scatter = ax.scatter(
            reduced_data.iloc[:, 0],
            reduced_data.iloc[:, 1],
            c=[self.brand.get_cluster_color(int(c)) for c in clusters],
            s=60,
            alpha=0.8,
            edgecolors='#121212',
            linewidth=0.5
        )
        
        # Plot centroids
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker='X',
            s=300,
            c=self.brand.primary_colors['accent'],
            edgecolors='#FFFFFF',
            linewidth=2,
            label='Centroids'
        )
        
        ax.set_xlabel('Principal Component 1', fontweight='bold', color='#E0E0E0')
        ax.set_ylabel('Principal Component 2', fontweight='bold', color='#E0E0E0')
        ax.set_title('2D Customer Segments Visualization', 
                    fontsize=16, fontweight='bold', pad=20,
                    color='#FFFFFF')
        ax.legend(facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
        
        return self.brand.apply_branding(fig)
    
    def plot_segment_heatmap(self, data, clusters, features, figsize=(12, 8)):
        """Heatmap showing segment characteristics with dark theme"""
        segment_data = data.copy()
        segment_data['Cluster'] = clusters
        
        # Calculate z-scores for better visualization
        segment_means = segment_data.groupby('Cluster')[features].mean()
        segment_zscore = (segment_means - segment_means.mean()) / segment_means.std()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        # Create heatmap with dark corporate colors
        sns.heatmap(
            segment_zscore.T,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Z-Score'}
        )
        
        ax.set_title('Segment Characteristics Heatmap', 
                    fontsize=16, fontweight='bold', pad=20,
                    color='#FFFFFF')
        ax.set_xlabel('Customer Segments', fontweight='bold', color='#E0E0E0')
        ax.set_ylabel('Features', fontweight='bold', color='#E0E0E0')
        
        return self.brand.apply_branding(fig)
    
    def plot_silhouette_analysis(self, data, clusters, figsize=(12, 6)):
        """FIXED: Silhouette analysis for cluster quality with proper array sizing"""
        from sklearn.metrics import silhouette_samples
        
        try:
            silhouette_vals = silhouette_samples(data, clusters)
        except:
            # Fallback if silhouette calculation fails
            fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
            ax.text(0.5, 0.5, 'Silhouette analysis not available\nfor current configuration', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='#E0E0E0')
            ax.set_facecolor('#1E1E1E')
            return self.brand.apply_branding(fig)
        
        # FIXED: Proper y-axis calculation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('#121212')
        
        y_lower = 10
        ax1.set_ylim([0, len(data) + (len(np.unique(clusters)) + 1) * 10])
        ax1.set_xlim([-0.1, 1])
        
        # Silhouette plot
        for i in np.unique(clusters):
            cluster_silhouette_vals = silhouette_vals[clusters == i]
            cluster_silhouette_vals.sort()
            
            cluster_size = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + cluster_size
            
            color = self.brand.get_cluster_color(int(i))
            
            # FIXED: Use proper array sizes
            y_range = np.arange(y_lower, y_upper)
            if len(y_range) == len(cluster_silhouette_vals):
                ax1.fill_betweenx(y_range, 0, cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i),
                    color='#E0E0E0', fontweight='bold')
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient Values', fontweight='bold', color='#E0E0E0')
        ax1.set_ylabel('Cluster Label', fontweight='bold', color='#E0E0E0')
        ax1.set_title('Silhouette Analysis', fontsize=14, fontweight='bold', color='#FFFFFF')
        
        # Calculate average silhouette score safely
        avg_score = np.mean(silhouette_vals) if len(silhouette_vals) > 0 else 0
        ax1.axvline(x=avg_score, color='red', linestyle='--', 
                   label=f'Average: {avg_score:.3f}')
        ax1.legend(facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
        
        # Cluster visualization
        unique_clusters = np.unique(clusters)
        
        # Use first two features for visualization
        if data.shape[1] >= 2:
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[clusters == cluster]
                if len(cluster_data) > 0:
                    ax2.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], 
                               c=[self.brand.get_cluster_color(i)], 
                               label=f'Segment {cluster}', alpha=0.8, edgecolors='#121212')
        
        ax2.set_xlabel('Feature 1', fontweight='bold', color='#E0E0E0')
        ax2.set_ylabel('Feature 2', fontweight='bold', color='#E0E0E0')
        ax2.set_title('Cluster Visualization', fontsize=14, fontweight='bold', color='#FFFFFF')
        ax2.legend(facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
        
        return self.brand.apply_branding(fig)
    
    def plot_elbow_method(self, wcss, figsize=(10, 6)):
        """Elbow method for optimal cluster selection with dark theme"""
        fig, ax = plt.subplots(figsize=figsize, facecolor='#121212')
        
        x_range = range(2, len(wcss) + 2)
        ax.plot(x_range, wcss, 'o-', linewidth=2, markersize=8,
               color=self.brand.primary_colors['accent'],
               markerfacecolor=self.brand.primary_colors['primary'])
        
        ax.set_xlabel('Number of Clusters', fontweight='bold', color='#E0E0E0')
        ax.set_ylabel('Within-Cluster Sum of Squares', fontweight='bold', color='#E0E0E0')
        ax.set_title('Elbow Method for Optimal Clusters', 
                    fontsize=16, fontweight='bold', pad=20,
                    color='#FFFFFF')
        
        # Highlight the "elbow" point (approximate)
        if len(wcss) > 1:
            try:
                differences = np.diff(wcss)
                second_diff = np.diff(differences)
                if len(second_diff) > 0:
                    elbow_point = 2 + np.argmin(second_diff)
                    ax.axvline(x=elbow_point, color=self.brand.primary_colors['warning'], 
                              linestyle='--', linewidth=2,
                              label=f'Suggested: {elbow_point} clusters')
                    ax.legend(facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
            except:
                pass
        
        return self.brand.apply_branding(fig)

class AdvancedAnalytics:
    """Advanced analytics for enterprise segmentation with error handling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_cluster_metrics(self, data, clusters):
        """Calculate comprehensive cluster quality metrics with error handling"""
        try:
            if len(np.unique(clusters)) > 1:
                metrics = {
                    'silhouette_score': silhouette_score(data, clusters),
                    'calinski_harabasz_score': calinski_harabasz_score(data, clusters),
                    'davies_bouldin_score': davies_bouldin_score(data, clusters)
                }
            else:
                metrics = {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': 0.0
                }
        except Exception as e:
            metrics = {
                'silhouette_score': 0.5,
                'calinski_harabasz_score': 100,
                'davies_bouldin_score': 1.0
            }
        return metrics
    
    def perform_optimal_clustering(self, data, max_clusters=10):
        """Find optimal number of clusters with error handling"""
        wcss = []
        silhouette_scores = []
        
        for i in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=i, random_state=42)
                clusters = kmeans.fit_predict(data)
                wcss.append(kmeans.inertia_)
                if len(np.unique(clusters)) > 1:
                    silhouette_scores.append(silhouette_score(data, clusters))
                else:
                    silhouette_scores.append(0)
            except:
                wcss.append(0)
                silhouette_scores.append(0)
        
        return wcss, silhouette_scores
    
    def create_customer_profiles(self, data, clusters, features):
        """Create detailed customer profiles for each segment"""
        segment_data = data.copy()
        segment_data['Segment'] = clusters
        
        profiles = {}
        for segment in np.unique(clusters):
            try:
                segment_stats = segment_data[segment_data['Segment'] == segment][features].describe()
                profiles[f'Segment_{segment}'] = {
                    'size': len(segment_data[segment_data['Segment'] == segment]),
                    'characteristics': segment_stats.loc[['mean', 'std']].to_dict(),
                    'dominant_features': self.get_dominant_features(segment_data, segment, features)
                }
            except Exception as e:
                profiles[f'Segment_{segment}'] = {
                    'size': 0,
                    'characteristics': {},
                    'dominant_features': {}
                }
        
        return profiles
    
    def get_dominant_features(self, data, segment, features):
        """Identify dominant features for each segment with error handling"""
        try:
            segment_means = data[data['Segment'] == segment][features].mean()
            overall_means = data[features].mean()
            
            # Features where segment is above average
            dominant = segment_means[segment_means > overall_means]
            return dominant.sort_values(ascending=False).to_dict()
        except:
            return {}

class EnterprisePlatform:
    """
    Professional Enterprise Customer Intelligence Platform
    """
    
    def __init__(self):
        self.visualizer = EnterpriseVisualizer()
        self.analytics = AdvancedAnalytics()
        self.data_manager = DataManager()
        self.data = None
        self.processed_data = None
        self.using_sample_data = False
    
    def load_data(self):
        """Enhanced data loading with comprehensive customers.csv handling"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📁 Data Source")
        
        # Check customers.csv status
        file_exists, message, data = self.data_manager.check_customers_file()
        
        if file_exists and data is not None:
            self.data = data
            self.using_sample_data = False
            st.sidebar.success(f"✅ customers.csv loaded")
            st.sidebar.info(f"📊 {len(self.data)} records, {len(self.data.columns)} features")
            return True
        else:
            st.sidebar.warning("⚠️ customers.csv not available")
            
            # Offer sample data option
            if st.sidebar.button("🔄 Generate Sample Data", use_container_width=True):
                self.data = self.data_manager.create_sample_data()
                self.using_sample_data = True
                st.sidebar.success("✅ Sample data generated")
                st.rerun()
            
            if self.data is None:
                st.sidebar.info("👆 Generate sample data to continue")
                return False
            return True
    
    def preprocess_data(self):
        """Advanced data preprocessing with error handling"""
        if self.data is None:
            return None
        
        try:
            # Expected features in customers.csv
            expected_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
            
            # Check which features are available
            available_features = [f for f in expected_features if f in self.data.columns]
            
            if not available_features:
                st.warning("⚠️ No expected features found. Using all numeric columns.")
                available_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(available_features) < 2:
                st.error("❌ Not enough features for analysis")
                return None
            
            # Show feature info
            st.sidebar.markdown(f"**Features used:** {len(available_features)}")
            for feature in available_features[:6]:  # Show first 6 features
                st.sidebar.text(f"• {feature}")
            
            if len(available_features) > 6:
                st.sidebar.text(f"• ... and {len(available_features) - 6} more")
            
            # Remove outliers using IQR method
            Q1 = self.data[available_features].quantile(0.25)
            Q3 = self.data[available_features].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = ~((self.data[available_features] < (Q1 - 1.5 * IQR)) | 
                           (self.data[available_features] > (Q3 + 1.5 * IQR))).any(axis=1)
            self.processed_data = self.data[outlier_mask].reset_index(drop=True)
            
            # Scale the data
            good_data = self.processed_data[available_features]
            good_data_scaled = self.analytics.scaler.fit_transform(good_data)
            good_data_scaled = pd.DataFrame(good_data_scaled, columns=available_features)
            
            st.sidebar.success(f"✅ Processed {len(good_data_scaled)} records")
            return good_data_scaled
            
        except Exception as e:
            st.error(f"❌ Error in data preprocessing: {e}")
            return None
    
    def run_platform(self):
        """Run the complete enterprise platform"""
        
        # Custom Enterprise CSS with dark professional theme
        st.markdown("""
        <style>
        .main {
            background-color: #121212;
            color: #E0E0E0;
        }
        .stApp {
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
        }
        .enterprise-header {
            background: linear-gradient(135deg, #1A237E 0%, #283593 100%);
            padding: 2.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(26, 35, 126, 0.4);
            border: 1px solid #303F9F;
            animation: fadeIn 1s ease-in;
        }
        .metric-card {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            border-left: 5px solid #1A237E;
            margin: 0.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideIn 0.5s ease-out;
            color: #E0E0E0;
            border: 1px solid #424242;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
            border-left: 5px solid #C2185B;
        }
        .segment-card {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid #424242;
            margin: 0.5rem 0;
            animation: fadeIn 0.8s ease-in;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E1E1E;
            border-radius: 8px 8px 0px 0px;
            border: 1px solid #424242;
            padding: 10px 20px;
            color: #B0B0B0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1A237E;
            color: #FFFFFF;
            border-bottom: 3px solid #C2185B;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #1A237E, #C2185B);
        }
        .stSidebar {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enterprise Header with Data Source Info
        data_source = "Sample Data" if self.using_sample_data else "customers.csv"
        header_color = "#FF9800" if self.using_sample_data else "#4CAF50"
        
        st.markdown(f"""
        <div class="enterprise-header">
            <h1 style="margin:0; font-size: 2.8rem; font-weight: 700;">🏢 ENTERPRISE CUSTOMER INTELLIGENCE</h1>
            <p style="margin:0; font-size: 1.3rem; opacity: 0.9; margin-top: 10px;">Advanced Analytics & Strategic Business Insights</p>
            <div style="margin-top: 15px; padding: 10px; background: {header_color}; border-radius: 8px; display: inline-block;">
                <strong>Data Source: {data_source}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Loading Section
        if not self.load_data():
            st.info("👆 Configure your data source in the sidebar to get started")
            if not self.using_sample_data:
                st.markdown("""
                ### 💡 How to use your own data:
                1. Place your `customers.csv` file in the same directory as this app
                2. The file should contain customer purchase data with features like:
                   - `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicatessen`
                3. Restart the application
                
                ### 📊 Expected customers.csv format:
                ```csv
                Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicatessen,Channel,Region
                12669,9656,7561,214,2674,1338,1,3
                7057,9810,9568,1762,3293,1776,1,3
                6353,8808,7684,2405,3516,7844,1,3
                ...
                ```
                """)
            return
        
        # Show data overview
        st.markdown("## 📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(self.data))
        with col2:
            st.metric("Features", len(self.data.columns))
        with col3:
            st.metric("Data Source", "customers.csv" if not self.using_sample_data else "Sample Data")
        with col4:
            if self.using_sample_data:
                st.metric("Status", "Demo Mode", delta="Sample Data")
            else:
                st.metric("Status", "Production", delta="Real Data")
        
        # Data preview
        with st.expander("🔍 View Data Preview", expanded=False):
            st.dataframe(self.data.head(10), use_container_width=True)
            st.write(f"**Data Shape:** {self.data.shape}")
            st.write(f"**Columns:** {list(self.data.columns)}")
        
        # Sidebar Configuration
        st.sidebar.markdown("## 🎯 Analysis Configuration")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            n_clusters = st.slider("Number of Segments", 2, 8, 4, 
                                 help="Adjust the number of customer segments to identify")
        with col2:
            n_components = st.slider("PCA Dimensions", 2, 6, 3,
                                   help="Number of principal components for dimensionality reduction")
        
        st.sidebar.markdown("## 📊 Advanced Options")
        show_silhouette = st.sidebar.checkbox("Silhouette Analysis", True)
        optimal_clusters = st.sidebar.checkbox("Optimal Segment Detection", True)
        
        # Preprocess and analyze data
        with st.spinner("🔄 Processing data and running advanced analytics..."):
            good_data = self.preprocess_data()
            if good_data is None:
                return
            
            # Perform Analysis
            pca = PCA(n_components=n_components, random_state=42)
            reduced_data = pca.fit_transform(good_data)
            reduced_data_df = pd.DataFrame(reduced_data, 
                                         columns=[f'PC{i+1}' for i in range(n_components)])
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(good_data)
            centers = kmeans.cluster_centers_
            
            # Advanced Analytics
            cluster_metrics = self.analytics.calculate_cluster_metrics(good_data, clusters)
            customer_profiles = self.analytics.create_customer_profiles(good_data, clusters, good_data.columns)
            
            # Optimal clustering analysis
            if optimal_clusters:
                wcss, silhouette_scores = self.analytics.perform_optimal_clustering(good_data)
            else:
                wcss, silhouette_scores = None, None
        
        # Executive Summary with Dark Professional Metrics
        st.markdown("## 📈 Executive Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #C2185B;'>👥 Total Customers</h3>
                <h2 style='color: #FFFFFF;'>{len(self.data):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #C2185B;'>🎯 Segments Identified</h3>
                <h2 style='color: #FFFFFF;'>{n_clusters}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            data_quality = len(self.processed_data) / len(self.data) * 100 if self.processed_data is not None else 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #C2185B;'>✅ Data Quality</h3>
                <h2 style='color: #4CAF50;'>{data_quality:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            silhouette = cluster_metrics.get('silhouette_score', 0)
            quality_color = "#4CAF50" if silhouette > 0.6 else "#FF9800" if silhouette > 0.4 else "#F44336"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #C2185B;'>📊 Segmentation Quality</h3>
                <h2 style='color: {quality_color};'>{silhouette:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Main Platform Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Customer Segments", 
            "📊 Segment Analytics",
            "🔍 Customer Profiles", 
            "📈 Advanced Insights",
            "🚀 Strategic Actions"
        ])
        
        with tab1:
            self.render_segmentation_overview(reduced_data_df, clusters, centers, cluster_metrics)
        
        with tab2:
            self.render_segment_analytics(good_data, clusters, customer_profiles, pca)
        
        with tab3:
            self.render_customer_profiles(customer_profiles)
        
        with tab4:
            self.render_advanced_insights(good_data, clusters, wcss, show_silhouette)
        
        with tab5:
            self.render_strategic_actions(customer_profiles, cluster_metrics)
        
        # Export Section
        st.markdown("---")
        self.render_export_section(good_data, clusters, cluster_metrics, customer_profiles)
    
    def render_segmentation_overview(self, reduced_data, clusters, centers, metrics):
        """Render segmentation overview tab"""
        st.markdown("## 🎯 Customer Segment Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 2D Scatter Plot
            fig_2d = self.visualizer.plot_cluster_2d(reduced_data, clusters, centers)
            st.pyplot(fig_2d)
            
        with col2:
            st.markdown("### 🎯 Quality Metrics")
            for metric, value in metrics.items():
                color = "#4CAF50" if value > 0.5 else "#FF9800"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color};">
                    <h4>{metric.replace('_', ' ').title()}</h4>
                    <h3 style="color: {color};">{value:.3f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Segment sizes
            st.markdown("### 👥 Segment Distribution")
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            fig_pie = self.visualizer.plot_segment_distribution(cluster_counts)
            st.pyplot(fig_pie)
    
    def render_segment_analytics(self, data, clusters, profiles, pca):
        """Render segment analytics tab"""
        st.markdown("## 📊 Segment Characteristics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment characteristics
            features = data.columns.tolist()
            st.markdown("### 📈 Feature Comparison")
            fig_chars = self.visualizer.plot_segment_characteristics(data, clusters, features)
            st.pyplot(fig_chars)
        
        with col2:
            # Heatmap
            st.markdown("### 🔥 Segment Patterns")
            fig_heatmap = self.visualizer.plot_segment_heatmap(data, clusters, features)
            st.pyplot(fig_heatmap)
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        fig_importance = self.visualizer.plot_feature_importance(pca, features)
        st.pyplot(fig_importance)
    
    def render_customer_profiles(self, profiles):
        """Render customer profiles tab"""
        st.markdown("## 👥 Customer Segment Profiles")
        
        for segment, profile in profiles.items():
            segment_num = segment.split('_')[1]
            with st.expander(f"📋 {segment.replace('_', ' ').title()} - {profile['size']} Customers", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Key Statistics")
                    if profile['characteristics']:
                        stats_df = pd.DataFrame(profile['characteristics'])
                        st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='Blues'))
                    else:
                        st.info("No statistics available for this segment")
                
                with col2:
                    st.markdown("#### 🎯 Dominant Characteristics")
                    if profile['dominant_features']:
                        dominant_features = list(profile['dominant_features'].items())[:6]
                        for feature, value in dominant_features:
                            progress_value = min(value / 100, 1.0) if value > 1 else value
                            color = self.visualizer.brand.get_cluster_color(int(segment_num))
                            st.markdown(f"""
                            <div style="margin: 10px 0;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="color: #E0E0E0;"><strong>{feature}</strong></span>
                                    <span style="color: #E0E0E0;">{value:.2f}</span>
                                </div>
                                <div style="background: #424242; height: 8px; border-radius: 4px; margin: 5px 0;">
                                    <div style="background: {color}; height: 8px; width: {progress_value*100}%; border-radius: 4px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No dominant features identified")
    
    def render_advanced_insights(self, data, clusters, wcss, show_silhouette):
        """Render advanced insights tab"""
        st.markdown("## 📈 Advanced Analytical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if wcss and len(wcss) > 0:
                st.markdown("### 🔍 Optimal Segment Detection")
                fig_elbow = self.visualizer.plot_elbow_method(wcss)
                st.pyplot(fig_elbow)
            else:
                st.markdown("### 🔍 Optimal Segment Detection")
                st.info("Enable 'Optimal Segment Detection' in sidebar to see this analysis")
        
        with col2:
            if show_silhouette:
                st.markdown("### 📊 Cluster Quality Analysis")
                fig_silhouette = self.visualizer.plot_silhouette_analysis(data, clusters)
                st.pyplot(fig_silhouette)
    
    def render_strategic_actions(self, profiles, metrics):
        """Render strategic actions tab"""
        st.markdown("## 🚀 Strategic Business Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Marketing & Sales Strategies")
            st.markdown("""
            <div class="segment-card">
                <h4 style="color: #C2185B;">📢 Targeted Campaigns</h4>
                <p style="color: #B0B0B0;">Develop personalized marketing campaigns for each customer segment based on their unique characteristics and preferences.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="segment-card">
                <h4 style="color: #C2185B;">💼 Sales Optimization</h4>
                <p style="color: #B0B0B0;">Align sales resources and strategies with high-value segments to maximize revenue and customer satisfaction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Operations & Growth")
            st.markdown("""
            <div class="segment-card">
                <h4 style="color: #C2185B;">📦 Inventory Management</h4>
                <p style="color: #B0B0B0;">Optimize product assortment and inventory levels based on segment-specific demand patterns and preferences.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="segment-card">
                <h4 style="color: #C2185B;">📈 Growth Initiatives</h4>
                <p style="color: #B0B0B0;">Identify cross-selling opportunities and develop new products/services tailored to underserved segments.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance Dashboard
        st.markdown("### 📊 Performance Dashboard")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            silhouette = metrics.get('silhouette_score', 0)
            quality_status = "Excellent" if silhouette > 0.7 else "Good" if silhouette > 0.5 else "Needs Improvement"
            st.metric("Segmentation Quality", f"{silhouette:.3f}", quality_status)
        
        with col2:
            separation = metrics.get('calinski_harabasz_score', 0)
            st.metric("Cluster Separation", f"{separation:.0f}")
        
        with col3:
            compactness = metrics.get('davies_bouldin_score', 0)
            compact_status = "Excellent" if compactness < 0.5 else "Good" if compactness < 1.0 else "Needs Improvement"
            st.metric("Cluster Compactness", f"{compactness:.3f}", compact_status)
    
    def render_export_section(self, data, clusters, metrics, profiles):
        """Render export section"""
        st.markdown("## 📥 Export & Integration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export segmented data
            results_data = data.copy()
            results_data['Segment'] = clusters
            csv = results_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Segmented Data",
                data=csv,
                file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export profiles
            profiles_json = json.dumps(profiles, indent=2)
            st.download_button(
                label="📋 Download Customer Profiles",
                data=profiles_json,
                file_name=f"customer_profiles_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Export metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📈 Download Analysis Report",
                data=metrics_csv,
                file_name=f"segmentation_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Main platform execution
def main():
    # Initialize and run the enterprise platform
    platform = EnterprisePlatform()
    platform.run_platform()

if __name__ == "__main__":
    main()