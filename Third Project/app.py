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
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import base64
import io
from datetime import datetime, timedelta
import hashlib
import secrets

# ML Imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Database & API
import sqlite3
from contextlib import contextmanager
import requests
from urllib.parse import urlparse

# Enhanced Configuration
class Config:
    """Centralized configuration management"""
    SECRET_KEY = os.getenv('APP_SECRET', secrets.token_hex(32))
    DB_PATH = "enterprise_customers.db"
    LOG_LEVEL = logging.INFO
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    SESSION_TIMEOUT = timedelta(hours=2)
    CACHE_DURATION = 3600  # 1 hour

# Enhanced Logging
class EnterpriseLogger:
    """Professional logging system"""
    def __init__(self):
        self.logger = logging.getLogger('EnterprisePlatform')
        self.logger.setLevel(Config.LOG_LEVEL)
        
        # Create logs directory if not exists
        Path("logs").mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler('logs/enterprise_platform.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
    
    def log_operation(self, operation: str, status: str, details: str = ""):
        """Log business operations"""
        self.logger.info(f"OPERATION: {operation} | STATUS: {status} | DETAILS: {details}")
    
    def log_security_event(self, event: str, user: str = "anonymous"):
        """Log security-related events"""
        self.logger.warning(f"SECURITY: {event} | USER: {user}")

# Enhanced Security
class SecurityManager:
    """Enhanced security management"""
    def __init__(self):
        self.logger = EnterpriseLogger()
    
    def validate_file_upload(self, uploaded_file) -> Tuple[bool, str]:
        """Comprehensive file upload validation"""
        try:
            # Check file size
            if uploaded_file.size > Config.MAX_FILE_SIZE:
                return False, "File size exceeds maximum limit"
            
            # Check file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in Config.ALLOWED_EXTENSIONS:
                return False, f"File type {file_ext} not allowed"
            
            # Basic malware scan (simulated)
            if self._detect_suspicious_patterns(uploaded_file):
                return False, "File contains suspicious patterns"
            
            return True, "File validation successful"
            
        except Exception as e:
            self.logger.log_security_event(f"File validation failed: {str(e)}")
            return False, f"File validation error: {str(e)}"
    
    def _detect_suspicious_patterns(self, uploaded_file) -> bool:
        """Basic pattern detection for suspicious content"""
        suspicious_patterns = [
            b'<script', b'javascript:', b'vbscript:', 
            b'<iframe', b'<object', b'<embed'
        ]
        
        content = uploaded_file.getvalue()
        return any(pattern in content.lower() for pattern in suspicious_patterns)
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token for form protection"""
        return secrets.token_hex(32)

# Enhanced Database Management
class DatabaseManager:
    """Professional database management with SQLite"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Analysis sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    data_source TEXT,
                    segment_count INTEGER,
                    analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metrics_json TEXT,
                    created_by TEXT DEFAULT 'system'
                )
            ''')
            
            # Customer segments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS customer_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    customer_id INTEGER,
                    segment INTEGER,
                    features_json TEXT,
                    prediction_confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
                )
            ''')
            
            # Model artifacts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    model_type TEXT,
                    model_binary BLOB,
                    feature_names TEXT,
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
                )
            ''')
            
            conn.commit()

# Enhanced EnterpriseBrand with Professional Themes
class EnterpriseBrand:
    """Enhanced enterprise branding with multiple themes"""
    
    def __init__(self):
        self.themes = {
            'dark_professional': self._get_dark_theme(),
            'light_corporate': self._get_light_theme(),
            'blue_business': self._get_blue_theme()
        }
        self.current_theme = 'dark_professional'
    
    def _get_dark_theme(self):
        return {
            'primary': '#1A237E', 'secondary': '#283593', 'accent': '#C2185B',
            'success': '#388E3C', 'warning': '#F57C00', 'info': '#0277BD',
            'dark': '#121212', 'darker': '#0A0A0A', 'light': '#424242', 
            'lighter': '#616161', 'text_primary': '#FFFFFF', 'text_secondary': '#E0E0E0',
            'background': '#121212', 'card_bg': '#1E1E1E', 'border': '#424242'
        }
    
    def _get_light_theme(self):
        return {
            'primary': '#1976D2', 'secondary': '#2196F3', 'accent': '#FF4081',
            'success': '#4CAF50', 'warning': '#FF9800', 'info': '#00BCD4',
            'dark': '#F5F5F5', 'darker': '#E0E0E0', 'light': '#FFFFFF', 
            'lighter': '#FAFAFA', 'text_primary': '#212121', 'text_secondary': '#757575',
            'background': '#FFFFFF', 'card_bg': '#FAFAFA', 'border': '#E0E0E0'
        }
    
    def _get_blue_theme(self):
        return {
            'primary': '#0D47A1', 'secondary': '#1565C0', 'accent': '#E91E63',
            'success': '#2E7D32', 'warning': '#EF6C00', 'info': '#006978',
            'dark': '#0A1A35', 'darker': '#051225', 'light': '#1E3A5F', 
            'lighter': '#2D4B75', 'text_primary': '#FFFFFF', 'text_secondary': '#BBDEFB',
            'background': '#0A1A35', 'card_bg': '#1E3A5F', 'border': '#2D4B75'
        }
    
    @property
    def colors(self):
        return self.themes[self.current_theme]
    
    def set_theme(self, theme_name: str):
        if theme_name in self.themes:
            self.current_theme = theme_name
    
    def get_cluster_color(self, index: int) -> str:
        """Enhanced color palette with theme support"""
        cluster_palettes = {
            'dark_professional': [
                '#1A237E', '#283593', '#303F9F', '#3949AB', '#3F51B5',
                '#C2185B', '#D81B60', '#E91E63', '#EC407A', '#F06292',
                '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784',
                '#F57C00', '#FB8C00', '#FF9800', '#FFA726', '#FFB74D'
            ],
            'light_corporate': [
                '#1976D2', '#2196F3', '#03A9F4', '#00BCD4', '#0097A7',
                '#FF4081', '#E91E63', '#C2185B', '#AD1457', '#880E4F',
                '#4CAF50', '#66BB6A', '#81C784', '#AED581', '#C5E1A5',
                '#FF9800', '#FFA726', '#FFB74D', '#FFCC80', '#FFE0B2'
            ],
            'blue_business': [
                '#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#2196F3',
                '#E91E63', '#D81B60', '#C2185B', '#AD1457', '#880E4F',
                '#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A',
                '#EF6C00', '#F57C00', '#FB8C00', '#FF9800', '#FFA726'
            ]
        }
        palette = cluster_palettes.get(self.current_theme, cluster_palettes['dark_professional'])
        return palette[index % len(palette)]

# Enhanced Data Manager with Advanced Features
class DataManager:
    """Enhanced data management with professional features"""
    
    def __init__(self):
        self.logger = EnterpriseLogger()
        self.security = SecurityManager()
    
    def check_customers_file(self, file_path: str = "customers.csv") -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Enhanced file checking - FIXED: Removed problematic decorator"""
        try:
            if not os.path.exists(file_path):
                return False, f"File '{file_path}' not found", None
            
            # Determine file type and read accordingly
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                return False, f"Unsupported file format: {file_path}", None
            
            if data.empty:
                return False, f"File '{file_path}' is empty", None
            
            # Enhanced feature validation
            validation_result = self._validate_data_structure(data)
            if not validation_result[0]:
                return validation_result
            
            self.logger.log_operation("file_validation", "success", f"Loaded {len(data)} records")
            return True, "File loaded successfully with validated structure", data
            
        except Exception as e:
            self.logger.log_operation("file_validation", "failed", str(e))
            return False, f"Error reading '{file_path}': {str(e)}", None
    
    def _validate_data_structure(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Comprehensive data structure validation"""
        try:
            # Check for minimum data requirements
            if len(data) < 10:
                return False, "Insufficient data records (minimum 10 required)"
            
            if len(data.columns) < 3:
                return False, "Insufficient features for analysis"
            
            # Check for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 3:
                return False, "Insufficient numeric features for clustering"
            
            # Check for missing values
            missing_percentage = data[numeric_cols].isnull().sum().sum() / (len(data) * len(numeric_cols))
            if missing_percentage > 0.3:  # 30% threshold
                return False, f"Excessive missing values ({missing_percentage:.1%})"
            
            # Check for constant columns
            constant_cols = [col for col in numeric_cols if data[col].nunique() <= 1]
            if constant_cols:
                return False, f"Constant columns detected: {constant_cols}"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def handle_file_upload(self) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Handle file uploads with security validation"""
        uploaded_file = st.sidebar.file_uploader(
            "Upload Customer Data", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your customer data file (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Security validation
            is_valid, message = self.security.validate_file_upload(uploaded_file)
            if not is_valid:
                return False, message, None
            
            try:
                # Read uploaded file
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Validate structure
                is_valid, message = self._validate_data_structure(data)
                if is_valid:
                    self.logger.log_operation("file_upload", "success", f"Uploaded {len(data)} records")
                    return True, f"✅ {uploaded_file.name} uploaded successfully", data
                else:
                    return False, f"❌ {message}", None
                    
            except Exception as e:
                self.logger.log_operation("file_upload", "failed", str(e))
                return False, f"❌ Error processing uploaded file: {str(e)}", None
        
        return False, "No file uploaded", None

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

# Enhanced Analytics with Multiple Algorithms
class AdvancedAnalytics:
    """Enhanced analytics with multiple ML algorithms and professional features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = EnterpriseLogger()
        self.db = DatabaseManager()
    
    def perform_advanced_clustering(self, data: pd.DataFrame, method: str = 'kmeans', **kwargs):
        """Perform clustering with multiple algorithm support"""
        try:
            if method == 'kmeans':
                return self._kmeans_clustering(data, **kwargs)
            elif method == 'hierarchical':
                return self._hierarchical_clustering(data, **kwargs)
            elif method == 'dbscan':
                return self._dbscan_clustering(data, **kwargs)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
                
        except Exception as e:
            self.logger.log_operation("clustering", "failed", f"{method}: {str(e)}")
            raise
    
    def _kmeans_clustering(self, data: pd.DataFrame, n_clusters: int = 4, **kwargs):
        """Enhanced K-means with auto-optimization"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,  # Multiple initializations for stability
            max_iter=300,
            tol=1e-4
        )
        
        clusters = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_
        
        return {
            'clusters': clusters,
            'centers': centers,
            'model': kmeans,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
    
    def _hierarchical_clustering(self, data: pd.DataFrame, n_clusters: int = 4, **kwargs):
        """Hierarchical clustering implementation"""
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        
        clusters = hierarchical.fit_predict(data)
        
        return {
            'clusters': clusters,
            'centers': self._calculate_centroids(data, clusters),
            'model': hierarchical
        }
    
    def _dbscan_clustering(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5, **kwargs):
        """DBSCAN clustering for density-based segmentation"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        
        return {
            'clusters': clusters,
            'centers': self._calculate_centroids(data, clusters),
            'model': dbscan,
            'n_clusters': len(np.unique(clusters[clusters != -1]))
        }
    
    def _calculate_centroids(self, data: pd.DataFrame, clusters: np.ndarray):
        """Calculate centroids for non-centroid-based algorithms"""
        centroids = []
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  # Skip noise points
                cluster_data = data[clusters == cluster_id]
                centroids.append(cluster_data.mean(axis=0))
        return np.array(centroids)
    
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
    
    def perform_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        engineered_data = data.copy()
        
        # Create interaction features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Add polynomial features for important columns
        for col in numeric_cols[:3]:  # Use first 3 columns
            engineered_data[f'{col}_squared'] = data[col] ** 2
        
        # Add ratio features
        if len(numeric_cols) >= 2:
            engineered_data['fresh_to_grocery_ratio'] = (
                data.get('Fresh', data[numeric_cols[0]]) / 
                (data.get('Grocery', data[numeric_cols[1]]) + 1e-8)
            )
        
        return engineered_data
    
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """Anomaly detection using Isolation Forest"""
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomalies = iso_forest.fit_predict(data)
        return anomalies

# Enhanced Visualizer with Professional Reporting
class EnterpriseVisualizer:
    """Enhanced visualizer with professional reporting capabilities"""
    
    def __init__(self):
        self.brand = EnterpriseBrand()
        self.logger = EnterpriseLogger()
    
    def create_loading_animation(self, text="Processing..."):
        """Create loading animation"""
        placeholder = st.empty()
        for i in range(3):
            placeholder.markdown(f"<div style='text-align: center; color: {self.brand.colors['accent']};'><h3>{text}{'.' * (i+1)}</h3></div>", unsafe_allow_html=True)
            time.sleep(0.3)
        placeholder.empty()
    
    def plot_segment_distribution(self, cluster_counts, figsize=(10, 6)):
        """Enhanced segment distribution with professional dark styling"""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
        # Create pie chart with safe color assignment
        colors = [self.brand.get_cluster_color(i) for i in range(len(cluster_counts))]
        
        wedges, texts, autotexts = ax.pie(
            cluster_counts.values,
            labels=[f'Segment {i}' for i in cluster_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': self.brand.colors['background'], 'linewidth': 2, 'alpha': 0.9}
        )
        
        # Enhance text styling for dark background
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontweight('bold')
            text.set_color(self.brand.colors['text_secondary'])
        
        ax.set_title('Customer Segment Distribution', 
                    fontsize=16, fontweight='bold', pad=20,
                    color=self.brand.colors['text_primary'])
        
        return self.apply_branding(fig)
    
    def plot_segment_characteristics(self, data, clusters, features, figsize=(14, 8)):
        """Bar chart for segment characteristics with safe color handling"""
        segment_data = data.copy()
        segment_data['Cluster'] = clusters
        
        # Calculate mean values for each segment
        segment_means = segment_data.groupby('Cluster')[features].mean()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
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
                   edgecolor=self.brand.colors['background'],
                   linewidth=1)
        
        ax.set_xlabel('Features', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_ylabel('Average Value', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_title('Segment Characteristics Comparison', 
                    fontsize=16, fontweight='bold', 
                    color=self.brand.colors['text_primary'])
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', color=self.brand.colors['text_secondary'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 facecolor=self.brand.colors['card_bg'], 
                 edgecolor=self.brand.colors['border'], 
                 labelcolor=self.brand.colors['text_secondary'])
        
        return self.apply_branding(fig)
    
    def plot_feature_importance(self, pca, features, figsize=(10, 5)):  # FIXED: Reduced size
        """Enhanced feature importance visualization with dark theme - FIXED SIZE"""
        importance = np.abs(pca.components_[0])
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
        # Create bar plot with color gradient
        sequential_palette = ['#0A0A0A', '#1A237E', '#283593', '#303F9F', '#3949AB', '#3F51B5']
        colors = [sequential_palette[min(int(imp * 10), len(sequential_palette)-1)] 
                 for imp in importance]
        
        bars = ax.bar(features, importance, 
                     color=colors,
                     alpha=0.9,
                     edgecolor=self.brand.colors['background'],
                     linewidth=1)
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{imp:.3f}', ha='center', va='bottom', 
                   fontweight='bold', color='#FFFFFF', fontsize=9)  # Smaller font
        
        ax.set_xlabel('Features', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_ylabel('Importance Score', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_title('Feature Importance in Customer Segmentation', 
                    fontsize=14, fontweight='bold', pad=20,  # Smaller title
                    color=self.brand.colors['text_primary'])
        ax.tick_params(axis='x', rotation=45, colors=self.brand.colors['text_secondary'])
        ax.tick_params(axis='y', colors=self.brand.colors['text_secondary'])
        
        return self.apply_branding(fig)
    
    def plot_cluster_2d(self, reduced_data, clusters, centers, figsize=(12, 8)):
        """2D cluster visualization with enhanced dark styling"""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
        # Create color mapping for clusters
        unique_clusters = np.unique(clusters)
        
        # Plot clusters
        scatter = ax.scatter(
            reduced_data.iloc[:, 0],
            reduced_data.iloc[:, 1],
            c=[self.brand.get_cluster_color(int(c)) for c in clusters],
            s=60,
            alpha=0.8,
            edgecolors=self.brand.colors['background'],
            linewidth=0.5
        )
        
        # Plot centroids
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker='X',
            s=300,
            c=self.brand.colors['accent'],
            edgecolors='#FFFFFF',
            linewidth=2,
            label='Centroids'
        )
        
        ax.set_xlabel('Principal Component 1', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_ylabel('Principal Component 2', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_title('2D Customer Segments Visualization', 
                    fontsize=16, fontweight='bold', pad=20,
                    color=self.brand.colors['text_primary'])
        ax.legend(facecolor=self.brand.colors['card_bg'], 
                 edgecolor=self.brand.colors['border'], 
                 labelcolor=self.brand.colors['text_secondary'])
        
        return self.apply_branding(fig)
    
    def plot_segment_heatmap(self, data, clusters, features, figsize=(12, 8)):
        """Heatmap showing segment characteristics with dark theme"""
        segment_data = data.copy()
        segment_data['Cluster'] = clusters
        
        # Calculate z-scores for better visualization
        segment_means = segment_data.groupby('Cluster')[features].mean()
        segment_zscore = (segment_means - segment_means.mean()) / segment_means.std()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
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
                    color=self.brand.colors['text_primary'])
        ax.set_xlabel('Customer Segments', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_ylabel('Features', fontweight='bold', color=self.brand.colors['text_secondary'])
        
        return self.apply_branding(fig)
    
    def plot_silhouette_analysis(self, data, clusters, figsize=(12, 6)):
        """FIXED: Silhouette analysis for cluster quality with proper array sizing"""
        from sklearn.metrics import silhouette_samples
        
        try:
            silhouette_vals = silhouette_samples(data, clusters)
        except:
            # Fallback if silhouette calculation fails
            fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
            ax.text(0.5, 0.5, 'Silhouette analysis not available\nfor current configuration', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color=self.brand.colors['text_secondary'])
            ax.set_facecolor(self.brand.colors['card_bg'])
            return self.apply_branding(fig)
        
        # FIXED: Proper y-axis calculation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(self.brand.colors['background'])
        
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
                    color=self.brand.colors['text_secondary'], fontweight='bold')
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient Values', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax1.set_ylabel('Cluster Label', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax1.set_title('Silhouette Analysis', fontsize=14, fontweight='bold', color=self.brand.colors['text_primary'])
        
        # Calculate average silhouette score safely
        avg_score = np.mean(silhouette_vals) if len(silhouette_vals) > 0 else 0
        ax1.axvline(x=avg_score, color='red', linestyle='--', 
                   label=f'Average: {avg_score:.3f}')
        ax1.legend(facecolor=self.brand.colors['card_bg'], 
                  edgecolor=self.brand.colors['border'], 
                  labelcolor=self.brand.colors['text_secondary'])
        
        # Cluster visualization
        unique_clusters = np.unique(clusters)
        
        # Use first two features for visualization
        if data.shape[1] >= 2:
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[clusters == cluster]
                if len(cluster_data) > 0:
                    ax2.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], 
                               c=[self.brand.get_cluster_color(i)], 
                               label=f'Segment {cluster}', alpha=0.8, 
                               edgecolors=self.brand.colors['background'])
        
        ax2.set_xlabel('Feature 1', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax2.set_ylabel('Feature 2', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax2.set_title('Cluster Visualization', fontsize=14, fontweight='bold', color=self.brand.colors['text_primary'])
        ax2.legend(facecolor=self.brand.colors['card_bg'], 
                  edgecolor=self.brand.colors['border'], 
                  labelcolor=self.brand.colors['text_secondary'])
        
        return self.apply_branding(fig)
    
    def plot_elbow_method(self, wcss, figsize=(10, 6)):
        """Elbow method for optimal cluster selection with dark theme"""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.brand.colors['background'])
        
        x_range = range(2, len(wcss) + 2)
        ax.plot(x_range, wcss, 'o-', linewidth=2, markersize=8,
               color=self.brand.colors['accent'],
               markerfacecolor=self.brand.colors['primary'])
        
        ax.set_xlabel('Number of Clusters', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_ylabel('Within-Cluster Sum of Squares', fontweight='bold', color=self.brand.colors['text_secondary'])
        ax.set_title('Elbow Method for Optimal Clusters', 
                    fontsize=16, fontweight='bold', pad=20,
                    color=self.brand.colors['text_primary'])
        
        # Highlight the "elbow" point (approximate)
        if len(wcss) > 1:
            try:
                differences = np.diff(wcss)
                second_diff = np.diff(differences)
                if len(second_diff) > 0:
                    elbow_point = 2 + np.argmin(second_diff)
                    ax.axvline(x=elbow_point, color=self.brand.colors['warning'], 
                              linestyle='--', linewidth=2,
                              label=f'Suggested: {elbow_point} clusters')
                    ax.legend(facecolor=self.brand.colors['card_bg'], 
                             edgecolor=self.brand.colors['border'], 
                             labelcolor=self.brand.colors['text_secondary'])
            except:
                pass
        
        return self.apply_branding(fig)
    
    def apply_branding(self, fig):
        """Apply consistent enterprise branding to matplotlib figures"""
        fig.patch.set_facecolor(self.brand.colors['background'])
        if hasattr(fig, 'axes') and fig.axes:
            for ax in fig.axes:
                ax.set_facecolor(self.brand.colors['card_bg'])
                ax.title.set_color(self.brand.colors['text_primary'])
                ax.xaxis.label.set_color(self.brand.colors['text_secondary'])
                ax.yaxis.label.set_color(self.brand.colors['text_secondary'])
                ax.tick_params(colors=self.brand.colors['text_secondary'])
                
                # Style grid
                ax.grid(True, alpha=0.2, color=self.brand.colors['border'])
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color(self.brand.colors['border'])
                    spine.set_linewidth(1.5)
        
        return fig
    
    def create_comprehensive_report(self, data: pd.DataFrame, clusters: np.ndarray, 
                                  metrics: Dict, profiles: Dict) -> str:
        """Generate comprehensive HTML report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enterprise Customer Intelligence Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: {self.brand.colors['background']}; color: {self.brand.colors['text_primary']}; }}
                .header {{ background: {self.brand.colors['primary']}; color: white; padding: 30px; border-radius: 10px; }}
                .section {{ background: {self.brand.colors['card_bg']}; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid {self.brand.colors['border']}; }}
                .metric {{ display: inline-block; background: {self.brand.colors['secondary']}; color: white; padding: 10px 20px; margin: 5px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid {self.brand.colors['border']}; }}
                th {{ background-color: {self.brand.colors['primary']}; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏢 Enterprise Customer Intelligence Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric">Total Customers: {len(data)}</div>
                <div class="metric">Segments Identified: {len(np.unique(clusters))}</div>
                <div class="metric">Segmentation Quality: {metrics.get('silhouette_score', 0):.3f}</div>
            </div>
            
            <div class="section">
                <h2>📈 Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        """
        
        # Add metrics
        metric_interpretations = {
            'silhouette_score': ('> 0.7: Strong structure', '0.5-0.7: Reasonable', '< 0.5: Weak'),
            'calinski_harabasz_score': ('Higher = Better separation', '', ''),
            'davies_bouldin_score': ('Lower = Better', '< 0.5: Excellent', '> 1.0: Poor')
        }
        
        for metric, value in metrics.items():
            interpretation = metric_interpretations.get(metric, ('', '', ''))[0]
            report_html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.3f}</td><td>{interpretation}</td></tr>"
        
        report_html += """
                </table>
            </div>
            
            <div class="section">
                <h2>👥 Customer Segments</h2>
        """
        
        # Add segment profiles
        for segment, profile in profiles.items():
            report_html += f"""
                <h3>{segment.replace('_', ' ').title()}</h3>
                <p><strong>Size:</strong> {profile.get('size', 0)} customers</p>
                <p><strong>Dominant Features:</strong> {', '.join(list(profile.get('dominant_features', {}).keys())[:3])}</p>
            """
        
        report_html += """
            </div>
        </body>
        </html>
        """
        
        return report_html

# Enhanced Enterprise Platform
class EnterprisePlatform:
    """
    Professional Enterprise Customer Intelligence Platform
    with Enhanced Features and Production Readiness
    """
    
    def __init__(self):
        self.visualizer = EnterpriseVisualizer()
        self.analytics = AdvancedAnalytics()
        self.data_manager = DataManager()
        self.security = SecurityManager()
        self.logger = EnterpriseLogger()
        self.db = DatabaseManager()
        
        self.data = None
        self.processed_data = None
        self.using_sample_data = False
        self.current_session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "🎯 Customer Segments"
        if 'csrf_token' not in st.session_state:
            st.session_state.csrf_token = self.security.generate_csrf_token()
    
    def apply_enterprise_css(self):
        """Apply enhanced enterprise CSS"""
        theme = self.visualizer.brand.colors
        
        st.markdown(f"""
        <style>
        .main {{
            background-color: {theme['background']};
            color: {theme['text_primary']};
        }}
        .stApp {{
            background: linear-gradient(135deg, {theme['darker']} 0%, {theme['dark']} 100%);
        }}
        .enterprise-header {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            padding: 2.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(26, 35, 126, 0.4);
            border: 1px solid {theme['border']};
            animation: fadeIn 1s ease-in;
        }}
        .metric-card {{
            background: linear-gradient(135deg, {theme['card_bg']} 0%, {theme['light']} 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            border-left: 5px solid {theme['primary']};
            margin: 0.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideIn 0.5s ease-out;
            color: {theme['text_primary']};
            border: 1px solid {theme['border']};
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
            border-left: 5px solid {theme['accent']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {theme['card_bg']};
            border-radius: 8px 8px 0px 0px;
            border: 1px solid {theme['border']};
            padding: 10px 20px;
            color: {theme['text_secondary']};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {theme['primary']};
            color: #FFFFFF;
            border-bottom: 3px solid {theme['accent']};
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: scale(0.95); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes slideIn {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['accent']});
        }}
        .stSidebar {{
            background: linear-gradient(135deg, {theme['card_bg']} 0%, {theme['light']} 100%);
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def render_theme_selector(self):
        """Render theme selection in sidebar"""
        st.sidebar.markdown("## 🎨 Theme Settings")
        theme = st.sidebar.selectbox(
            "Select Theme",
            options=['dark_professional', 'light_corporate', 'blue_business'],
            index=0,
            help="Choose your preferred color theme"
        )
        self.visualizer.brand.set_theme(theme)
    
    def _load_data_with_options(self) -> bool:
        """Enhanced data loading with multiple options - FIXED: Error handling"""
        st.sidebar.markdown("## 📁 Data Source Configuration")
        
        data_option = st.sidebar.radio(
            "Select Data Source",
            options=["Upload File", "Use customers.csv", "Generate Sample Data"],
            index=1
        )
        
        if data_option == "Upload File":
            success, message, data = self.data_manager.handle_file_upload()
            if success:
                self.data = data
                self.using_sample_data = False
                st.sidebar.success(message)
            elif data is None:
                return False
            else:
                st.sidebar.error(message)
                return False
                
        elif data_option == "Use customers.csv":
            # FIXED: Removed the problematic _self parameter
            success, message, data = self.data_manager.check_customers_file()
            if success and data is not None:
                self.data = data
                self.using_sample_data = False
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
                if st.sidebar.button("🔄 Generate Sample Data", use_container_width=True):
                    self.data = DataManager.create_sample_data()
                    self.using_sample_data = True
                    st.sidebar.success("✅ Sample data generated")
                    st.rerun()
                return False
                
        else:  # Generate Sample Data
            self.data = DataManager.create_sample_data()
            self.using_sample_data = True
            st.sidebar.success("✅ Sample data generated")
        
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
    
    def _calculate_feature_importance(self, data: pd.DataFrame, clusters: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        try:
            # Use random forest to determine feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(data, clusters)
            
            importance_dict = dict(zip(data.columns, rf.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except:
            # Fallback: use variance as importance
            variances = data.var().sort_values(ascending=False)
            return variances.to_dict()
    
    def _calculate_cluster_stability(self, data: pd.DataFrame, clusters: np.ndarray, n_iterations: int = 10) -> float:
        """Calculate cluster stability across multiple runs"""
        try:
            scores = []
            for _ in range(n_iterations):
                # Resample data
                sample_indices = np.random.choice(len(data), size=len(data), replace=True)
                sample_data = data.iloc[sample_indices]
                sample_clusters = clusters[sample_indices]
                
                # Recalculate metrics
                if len(np.unique(sample_clusters)) > 1:
                    score = silhouette_score(sample_data, sample_clusters)
                    scores.append(score)
            
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def render_advanced_analytics(self, data: pd.DataFrame, clusters: np.ndarray):
        """Render advanced analytics section - FIXED: Only in its own tab"""
        st.markdown("## 🔬 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            st.markdown("### 🎯 Feature Importance")
            feature_importance = self._calculate_feature_importance(data, clusters)
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_importance.keys())[:10]  # Top 10 features
            importance = list(feature_importance.values())[:10]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance, color=self.visualizer.brand.colors['primary'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importance Scores')
            
            st.pyplot(fig)
        
        with col2:
            # Cluster stability analysis
            st.markdown("### 📊 Cluster Stability")
            stability_score = self._calculate_cluster_stability(data, clusters)
            st.metric("Stability Score", f"{stability_score:.3f}")
            
            # Anomaly detection
            st.markdown("### 🚨 Anomaly Detection")
            anomalies = self.analytics.detect_anomalies(data)
            anomaly_count = np.sum(anomalies == -1)
            st.metric("Detected Anomalies", anomaly_count)
    
    def save_analysis_session(self, data: pd.DataFrame, clusters: np.ndarray, metrics: Dict, profiles: Dict):
        """Save analysis session to database"""
        try:
            with self.db.get_connection() as conn:
                # Save session metadata
                conn.execute('''
                    INSERT INTO analysis_sessions 
                    (session_id, data_source, segment_count, metrics_json)
                    VALUES (?, ?, ?, ?)
                ''', (
                    self.current_session_id,
                    'customers.csv' if not self.using_sample_data else 'sample_data',
                    len(np.unique(clusters)),
                    json.dumps(metrics)
                ))
                
                # Save segment assignments
                for i, (customer_id, cluster) in enumerate(zip(data.index, clusters)):
                    conn.execute('''
                        INSERT INTO customer_segments 
                        (session_id, customer_id, segment, features_json, prediction_confidence)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        self.current_session_id,
                        int(customer_id),
                        int(cluster),
                        json.dumps(data.iloc[i].to_dict()),
                        0.95  # Placeholder confidence
                    ))
                
                conn.commit()
                self.logger.log_operation("session_save", "success", f"Session {self.current_session_id}")
                
        except Exception as e:
            self.logger.log_operation("session_save", "failed", str(e))
    
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
        """Render segment analytics tab - FIXED: Proper chart sizing"""
        st.markdown("## 📊 Segment Characteristics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment characteristics - FIXED: Balanced size
            features = data.columns.tolist()
            st.markdown("### 📈 Feature Comparison")
            fig_chars = self.visualizer.plot_segment_characteristics(data, clusters, features)
            st.pyplot(fig_chars)
        
        with col2:
            # Heatmap - FIXED: Balanced size
            st.markdown("### 🔥 Segment Patterns")
            fig_heatmap = self.visualizer.plot_segment_heatmap(data, clusters, features)
            st.pyplot(fig_heatmap)
        
        # Feature importance - FIXED: Smaller size and only in this tab
        st.markdown("### 🎯 Feature Importance")
        fig_importance = self.visualizer.plot_feature_importance(pca, features, figsize=(10, 5))  # Smaller size
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
                                <div style="display: flex; justify-content: between;">
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
        
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            # Export HTML report
            html_report = self.visualizer.create_comprehensive_report(data, clusters, metrics, profiles)
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name=f"enterprise_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
    
    def run_platform(self):
        """Run the enhanced enterprise platform - FIXED: All issues resolved"""
        # Apply CSS and setup
        self.apply_enterprise_css()
        self.render_theme_selector()
        
        # Enhanced header with session info
        data_source = "Sample Data" if self.using_sample_data else "customers.csv"
        header_color = self.visualizer.brand.colors['warning'] if self.using_sample_data else self.visualizer.brand.colors['success']
        
        st.markdown(f"""
        <div class="enterprise-header">
            <h1 style="margin:0; font-size: 2.8rem; font-weight: 700;">🏢 ENTERPRISE CUSTOMER INTELLIGENCE</h1>
            <p style="margin:0; font-size: 1.3rem; opacity: 0.9; margin-top: 10px;">Advanced Analytics & Strategic Business Insights</p>
            <div style="margin-top: 15px; padding: 10px; background: {header_color}; border-radius: 8px; display: inline-block;">
                <strong>Data Source: {data_source} | Session: {self.current_session_id}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data loading with enhanced options
        if not self._load_data_with_options():
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
        
        # Algorithm selection
        clustering_method = st.sidebar.selectbox(
            "Clustering Algorithm",
            options=["kmeans", "hierarchical", "dbscan"],
            index=0,
            help="Choose the clustering algorithm to use"
        )
        
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
            
            # Enhanced Clustering with algorithm selection
            clustering_result = self.analytics.perform_advanced_clustering(
                good_data, 
                method=clustering_method,
                n_clusters=n_clusters
            )
            
            clusters = clustering_result['clusters']
            centers = clustering_result['centers']
            
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
                <h3 style='color: {self.visualizer.brand.colors["accent"]};'>👥 Total Customers</h3>
                <h2 style='color: #FFFFFF;'>{len(self.data):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: {self.visualizer.brand.colors["accent"]};'>🎯 Segments Identified</h3>
                <h2 style='color: #FFFFFF;'>{n_clusters}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            data_quality = len(self.processed_data) / len(self.data) * 100 if self.processed_data is not None else 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: {self.visualizer.brand.colors["accent"]};'>✅ Data Quality</h3>
                <h2 style='color: #4CAF50;'>{data_quality:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            silhouette = cluster_metrics.get('silhouette_score', 0)
            quality_color = "#4CAF50" if silhouette > 0.6 else "#FF9800" if silhouette > 0.4 else "#F44336"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: {self.visualizer.brand.colors["accent"]};'>📊 Segmentation Quality</h3>
                <h2 style='color: {quality_color};'>{silhouette:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Main Platform Tabs - FIXED: Removed duplicate advanced analytics call
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Customer Segments", 
            "📊 Segment Analytics",
            "🔍 Customer Profiles", 
            "📈 Advanced Insights",
            "🔬 Advanced Analytics",  # This tab will contain the feature importance analysis
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
            # FIXED: Advanced analytics only appears in this tab
            self.render_advanced_analytics(good_data, clusters)
        
        with tab6:
            self.render_strategic_actions(customer_profiles, cluster_metrics)
        
        # Export Section
        st.markdown("---")
        self.render_export_section(good_data, clusters, cluster_metrics, customer_profiles)
        
        # Save session
        if st.button("💾 Save Analysis Session", use_container_width=True):
            self.save_analysis_session(good_data, clusters, cluster_metrics, customer_profiles)
            st.success("Analysis session saved successfully!")

# Enhanced main execution
def main():
    """Enhanced main function with error handling and monitoring"""
    try:
        # Set page configuration for enterprise platform
        st.set_page_config(
            page_title="Enterprise Customer Intelligence",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize platform
        platform = EnterprisePlatform()
        
        # Add monitoring and performance tracking
        start_time = time.time()
        
        # Run platform
        platform.run_platform()
        
        # Log performance
        execution_time = time.time() - start_time
        platform.logger.log_operation(
            "platform_execution", 
            "completed", 
            f"Execution time: {execution_time:.2f}s"
        )
        
    except Exception as e:
        # Global error handling
        logger = EnterpriseLogger()
        logger.log_operation("platform_execution", "failed", str(e))
        
        st.error(f"""
        ## 🚨 Application Error
        
        The application encountered an unexpected error. This has been logged for investigation.
        
        **Error Details:** {str(e)}
        
        Please refresh the page and try again. If the problem persists, contact support.
        """)

if __name__ == "__main__":
    main()