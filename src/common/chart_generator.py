import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime
import numpy as np

class MLSNPChartGenerator:
    """
    Enhanced chart generator for MLS Next Pro predictions supporting both Monte Carlo 
    and Machine Learning methods with comparison capabilities.
    
    Charts generated vary by prediction method:
    - Monte Carlo: Playoff probabilities, rank distributions, points analysis, dashboard
    - ML: Playoff probabilities, feature importance, confidence analysis, ML dashboard  
    - Both: Method comparison, confidence comparison, feature importance, comparison dashboard
    """
    
    def __init__(self, output_dir: str = "output/charts"):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save chart HTML files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # MLS Next Pro team colors (you can customize these)
        self.team_colors = {
            'default': '#1f77b4',
            'clinched': '#2ca02c',  # Green for clinched
            'eliminated': '#d62728',  # Red for eliminated
            'bubble': '#ff7f0e'  # Orange for bubble teams
        }
    
    def generate_all_charts(self, summary_df: pd.DataFrame, simulation_results: Dict, 
                            qualification_data: Dict, conference: str, 
                            n_simulations: int, prediction_method: str = "monte_carlo",
                            feature_importance: Dict = None, 
                            comparison_data: Dict = None,
                            variability_stats: Dict = None) -> Dict[str, str]:
        """
        Generate charts based on prediction method.
        
        Args:
            summary_df: Summary DataFrame from simulation/prediction
            simulation_results: Raw simulation results (rank distributions for MC)
            qualification_data: Qualification analysis data
            conference: Conference name
            n_simulations: Number of simulations run (ignored for ML)
            prediction_method: "monte_carlo", "machine_learning", or "both"
            feature_importance: Feature importance dict for ML method
            comparison_data: Data for comparing both methods
            variability_stats: Feature importance variability statistics
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_files = {}
        
        if prediction_method == "monte_carlo":
            chart_files = self._generate_monte_carlo_charts(
                summary_df, simulation_results, qualification_data, 
                conference, n_simulations, timestamp
            )
            
        elif prediction_method == "ml":
            chart_files = self._generate_ml_charts(
                summary_df, feature_importance, conference, timestamp, variability_stats
            )
            
        elif prediction_method == "both" and comparison_data:
            chart_files = {}
            
            # Generate individual charts for Monte Carlo
            mc_data = comparison_data.get('monte_carlo')
            if mc_data:
                mc_charts = self._generate_monte_carlo_charts(
                    mc_data['summary_df'], 
                    mc_data['simulation_results'], 
                    mc_data['qualification_data'], 
                    conference, 
                    n_simulations, 
                    timestamp
                )
                for key, path in mc_charts.items():
                    chart_files[f'mc_{key}'] = path
            
            # Generate individual charts for ML
            ml_data = comparison_data.get('machine_learning')
            if ml_data:
                ml_charts = self._generate_ml_charts(
                    ml_data['summary_df'], 
                    ml_data.get('feature_importance', {}), 
                    conference, 
                    timestamp,
                    ml_data.get('variability_stats', {})
                )
                for key, path in ml_charts.items():
                    chart_files[f'ml_{key}'] = path
            
            # Generate comparison charts
            ml_data = comparison_data.get('machine_learning', {})
            ml_variability_stats = ml_data.get('variability_stats', {})

            comparison_charts = self._generate_comparison_charts(
                comparison_data, conference, n_simulations, 
                feature_importance, timestamp, ml_variability_stats
            )
            for key, path in comparison_charts.items():
                chart_files[f'comparison_{key}'] = path
        
        return chart_files
    
    def _generate_monte_carlo_charts(self, summary_df, simulation_results, 
                                   qualification_data, conference, n_simulations, timestamp):
        """Generate charts for Monte Carlo predictions."""
        chart_files = {}
        
        # 1. Playoff Probability Chart
        playoff_chart = self.create_playoff_probability_chart(
            summary_df, conference, n_simulations, "monte_carlo"
        )
        playoff_file = f"{self.output_dir}/playoff_probabilities_{conference}_{timestamp}.html"
        playoff_chart.write_html(playoff_file)
        chart_files['playoff_probabilities'] = playoff_file
        
        # 2. Rank Distribution Chart
        rank_chart = self.create_rank_distribution_chart(
            summary_df, simulation_results, conference
        )
        rank_file = f"{self.output_dir}/rank_distributions_{conference}_{timestamp}.html"
        rank_chart.write_html(rank_file)
        chart_files['rank_distributions'] = rank_file
        
        # 3. Points Analysis Chart
        points_chart = self.create_points_analysis_chart(summary_df, conference)
        points_file = f"{self.output_dir}/points_analysis_{conference}_{timestamp}.html"
        points_chart.write_html(points_file)
        chart_files['points_analysis'] = points_file
        
        # 4. Combined Dashboard
        dashboard = self.create_monte_carlo_dashboard(
            summary_df, simulation_results, conference, n_simulations
        )
        dashboard_file = f"{self.output_dir}/mc_dashboard_{conference}_{timestamp}.html"
        dashboard.write_html(dashboard_file)
        chart_files['dashboard'] = dashboard_file
        
        return chart_files
    
    def _generate_ml_charts(self, summary_df, feature_importance, conference, timestamp, variability_stats=None):
        """Generate charts for Machine Learning predictions."""
        chart_files = {}
        
        # 1. Playoff Probability Chart (ML version)
        playoff_chart = self.create_playoff_probability_chart(
            summary_df, conference, 0, "machine_learning"
        )
        playoff_file = f"{self.output_dir}/ml_playoff_probabilities_{conference}_{timestamp}.html"
        playoff_chart.write_html(playoff_file)
        chart_files['playoff_probabilities'] = playoff_file
        
        # 2. Standard Feature Importance Chart
        if feature_importance:
            importance_chart = self.create_feature_importance_chart(
                feature_importance, conference
            )
            importance_file = f"{self.output_dir}/ml_feature_importance_{conference}_{timestamp}.html"
            importance_chart.write_html(importance_file)
            chart_files['feature_importance'] = importance_file
        
        # 3. Feature Importance Variability Chart (if available)
        if variability_stats:
            variability_chart = self.create_feature_importance_variability_chart(
                variability_stats, conference
            )
            variability_file = f"{self.output_dir}/ml_feature_variability_{conference}_{timestamp}.html"
            variability_chart.write_html(variability_file)
            chart_files['feature_variability'] = variability_file
            
            # 4. Stability Heatmap
            stability_chart = self.create_stability_heatmap(
                variability_stats, conference
            )
            stability_file = f"{self.output_dir}/ml_stability_heatmap_{conference}_{timestamp}.html"
            stability_chart.write_html(stability_file)
            chart_files['stability_heatmap'] = stability_file
        
        # 5. ML Confidence Analysis
        confidence_chart = self.create_ml_confidence_chart(summary_df, conference)
        confidence_file = f"{self.output_dir}/ml_confidence_{conference}_{timestamp}.html"
        confidence_chart.write_html(confidence_file)
        chart_files['ml_confidence'] = confidence_file
        
        # 6. ML Dashboard
        dashboard = self.create_ml_dashboard(
            summary_df, feature_importance, conference, variability_stats
        )
        dashboard_file = f"{self.output_dir}/ml_dashboard_{conference}_{timestamp}.html"
        dashboard.write_html(dashboard_file)
        chart_files['dashboard'] = dashboard_file
        
        return chart_files
    
    def _generate_comparison_charts(self, comparison_data, conference, n_simulations, feature_importance, timestamp, variability_stats=None):
        """Generate comparison charts when both methods are run."""
        chart_files = {}
        
        # 1. Method Comparison Chart
        comparison_chart = self.create_method_comparison_chart(
            comparison_data, conference
        )
        comparison_file = f"{self.output_dir}/method_comparison_{conference}_{timestamp}.html"
        comparison_chart.write_html(comparison_file)
        chart_files['method_comparison'] = comparison_file
        
        # 2. Prediction Confidence Comparison
        confidence_comparison = self.create_confidence_comparison_chart(
            comparison_data, conference
        )
        confidence_file = f"{self.output_dir}/confidence_comparison_{conference}_{timestamp}.html"
        confidence_comparison.write_html(confidence_file)
        chart_files['confidence_comparison'] = confidence_file
        
        # 3. Feature Importance (if available)
        if feature_importance:
            importance_chart = self.create_feature_importance_chart(
                feature_importance, conference
            )
            importance_file = f"{self.output_dir}/feature_importance_{conference}_{timestamp}.html"
            importance_chart.write_html(importance_file)
            chart_files['feature_importance'] = importance_file
        
        # 4. Combined Comparison Dashboard
        dashboard = self.create_comparison_dashboard(
            comparison_data, feature_importance, conference, n_simulations, variability_stats
        )
        dashboard_file = f"{self.output_dir}/comparison_dashboard_{conference}_{timestamp}.html"
        dashboard.write_html(dashboard_file)
        chart_files['dashboard'] = dashboard_file
        
        return chart_files
    
    def create_playoff_probability_chart(self, summary_df: pd.DataFrame, 
                                       conference: str, n_simulations: int,
                                       prediction_method: str = "monte_carlo") -> go.Figure:
        """
        Create horizontal bar chart of playoff probabilities with method-appropriate subtitle.
        """
        # Sort by playoff probability
        df_sorted = summary_df.sort_values('Playoff Qualification %', ascending=True)
        
        # Assign colors based on playoff status
        colors = []
        for _, row in df_sorted.iterrows():
            prob = row['Playoff Qualification %']
            if prob >= 99.9:
                colors.append(self.team_colors['clinched'])
            elif prob <= 0.1:
                colors.append(self.team_colors['eliminated'])
            elif 25 <= prob <= 75:
                colors.append(self.team_colors['bubble'])
            else:
                colors.append(self.team_colors['default'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_sorted['Team'],
            x=df_sorted['Playoff Qualification %'],
            orientation='h',
            marker_color=colors,
            text=[f"{prob:.1f}%" for prob in df_sorted['Playoff Qualification %']],
            textposition='inside',
            textfont_color='white',
            hovertemplate='<b>%{y}</b><br>' +
                         'Playoff Probability: %{x:.1f}%<br>' +
                         'Current Points: %{customdata[0]}<br>' +
                         'Current Rank: %{customdata[1]}<extra></extra>',
            customdata=list(zip(df_sorted['Current Points'], df_sorted['Current Rank']))
        ))
        
        # Dynamic subtitle based on method
        if prediction_method == "machine_learning":
            subtitle = "Based on Machine Learning model predictions"
        else:
            subtitle = f"Based on {n_simulations:,} Monte Carlo simulations"
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Playoff Probabilities<br>' +
                       f'<sub>{subtitle}</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Playoff Probability (%)',
            yaxis_title='Teams',
            height=600,
            margin=dict(l=150),
            template='plotly_white'
        )
        
        # Add vertical lines for reference
        fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                     annotation_text="50% chance")
        fig.add_vline(x=75, line_dash="dot", line_color="green", 
                     annotation_text="75% chance")
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict[str, float], 
                                      conference: str) -> go.Figure:
        """Create feature importance chart for ML predictions."""
        if not feature_importance:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Feature importance data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Feature Importance - Data Not Available",
                height=400
            )
            return fig
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        # Color gradient
        colors = px.colors.sequential.Viridis_r[:len(features)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color=colors,
            text=[f"{imp:.3f}" for imp in importance],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Importance: %{x:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Key Prediction Factors<br>' +
                       '<sub>Machine Learning Model Feature Importance</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Feature Importance Score',
            yaxis_title='Prediction Factors',
            height=600,
            margin=dict(l=200),
            template='plotly_white'
        )
        
        return fig
    
    def create_ml_confidence_chart(self, summary_df: pd.DataFrame, 
                                 conference: str) -> go.Figure:
        """Create ML model confidence analysis chart."""
        # Calculate confidence metrics
        summary_df = summary_df.copy()
        
        # Confidence based on how close to 0% or 100% the playoff probability is
        summary_df['confidence'] = summary_df['Playoff Qualification %'].apply(
            lambda x: max(x, 100-x) if x > 50 else max(100-x, x)
        )
        
        # Expected points spread as uncertainty indicator
        summary_df['points_uncertainty'] = summary_df['Best Points'] - summary_df['Worst Points']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=summary_df['Average Points'],
            y=summary_df['Playoff Qualification %'],
            mode='markers',
            textposition='top center',
            marker=dict(
                size=summary_df['confidence']/4 + 5,  # Scale bubble size
                color=summary_df['points_uncertainty'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Points<br>Uncertainty"),
                line=dict(width=1, color='black')
            ),
            text=summary_df['Team'].str[:3],  # Use team abbreviations
            hovertemplate='<b>%{text}</b><br>' +
                         'Projected Points: %{x:.1f}<br>' +
                         'Playoff Probability: %{y:.1f}%<br>' +
                         'Model Confidence: %{customdata[0]:.1f}<br>' +
                         'Points Range: %{customdata[1]} - %{customdata[2]}<extra></extra>',
            customdata=list(zip(
                summary_df['confidence'], 
                summary_df['Worst Points'], 
                summary_df['Best Points']
            ))
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: ML Model Confidence<br>' +
                    '<sub>Bubble size = Model confidence | Color = Points uncertainty</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Projected Final Season Points<br><sub>Current Points + Expected Points from Remaining Games</sub>',
            yaxis_title='Playoff Qualification Probability (%)<br><sub>Likelihood of Finishing in Top 8 Regular Season Positions</sub>',
            height=600,
            template='plotly_white'
        )
        
        # Add playoff threshold line
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                     annotation_text="50% playoff chance")
        
        return fig
    
    def create_method_comparison_chart(self, comparison_data: Dict, 
                                     conference: str) -> go.Figure:
        """Create chart comparing ML vs Monte Carlo predictions."""
        mc_df = comparison_data['monte_carlo']['summary_df']
        ml_df = comparison_data['machine_learning']['summary_df']
        
        # Merge dataframes
        comparison = pd.merge(
            mc_df[['_team_id', 'Team', 'Current Points', 'Playoff Qualification %', 'Average Points']],
            ml_df[['_team_id', 'Playoff Qualification %', 'Average Points']],
            on='_team_id',
            suffixes=('_MC', '_ML')
        )
        
        # Calculate differences
        comparison['Playoff_Diff'] = comparison['Playoff Qualification %_ML'] - comparison['Playoff Qualification %_MC']
        comparison['Points_Diff'] = comparison['Average Points_ML'] - comparison['Average Points_MC']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Playoff Probability Comparison', 'Points Projection Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Playoff probability comparison
        fig.add_trace(
            go.Scatter(
                x=comparison['Playoff Qualification %_MC'],
                y=comparison['Playoff Qualification %_ML'],
                mode='markers+text',
                marker=dict(size=10, color='blue', opacity=0.7),
                text=comparison['Team'].str[:3],  # Team abbreviations
                textposition='top center',
                name='Teams',
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Monte Carlo: %{x:.1f}%<br>' +
                             'ML Prediction: %{y:.1f}%<br>' +
                             'Difference: %{customdata[1]:+.1f}%<extra></extra>',
                customdata=list(zip(comparison['Team'], comparison['Playoff_Diff']))
            ),
            row=1, col=1
        )
        
        # Add diagonal line (perfect agreement)
        fig.add_trace(
            go.Scatter(
                x=[0, 100], y=[0, 100],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Agreement',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Points comparison
        fig.add_trace(
            go.Scatter(
                x=comparison['Average Points_MC'],
                y=comparison['Average Points_ML'],
                mode='markers+text',
                marker=dict(size=10, color='red', opacity=0.7),
                text=comparison['Team'].str[:3],
                textposition='top center',
                name='Teams',
                showlegend=False,
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             'Monte Carlo: %{x:.1f} pts<br>' +
                             'ML Prediction: %{y:.1f} pts<br>' +
                             'Difference: %{customdata[1]:+.1f} pts<extra></extra>',
                customdata=list(zip(comparison['Team'], comparison['Points_Diff']))
            ),
            row=1, col=2
        )
        
        # Diagonal line for points
        min_pts = min(comparison['Average Points_MC'].min(), comparison['Average Points_ML'].min())
        max_pts = max(comparison['Average Points_MC'].max(), comparison['Average Points_ML'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_pts, max_pts], y=[min_pts, max_pts],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Monte Carlo vs Machine Learning<br>' +
                       '<sub>Points above diagonal line = ML predicts higher</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=500,
            template='plotly_white'
        )
        
        fig.update_xaxes(
            title_text="Monte Carlo Playoff Probability (%)<br><sub>Simulation-Based Predictions (Expected Value)</sub>", 
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Machine Learning Playoff Probability (%)<br><sub>Model-Based Predictions (AutoGluon)</sub>", 
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Monte Carlo Projected Final Points<br><sub>Average Points from Simulation Results</sub>", 
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Machine Learning Projected Final Points<br><sub>Model Predicted Final Season Total</sub>", 
            row=1, col=2
        )
        
        return fig
    
    def create_confidence_comparison_chart(self, comparison_data: Dict, 
                                         conference: str) -> go.Figure:
        """Compare prediction confidence between methods."""
        mc_df = comparison_data['monte_carlo']['summary_df']
        ml_df = comparison_data['machine_learning']['summary_df']
        
        # Merge dataframes
        comparison = pd.merge(
            mc_df[['_team_id', 'Team', 'Playoff Qualification %', 'Best Rank', 'Worst Rank']],
            ml_df[['_team_id', 'Playoff Qualification %', 'Best Points', 'Worst Points']],
            on='_team_id',
            suffixes=('_MC', '_ML')
        )
        
        # Calculate uncertainty metrics
        comparison['MC_Rank_Spread'] = comparison['Worst Rank'] - comparison['Best Rank']
        comparison['ML_Points_Spread'] = comparison['Worst Points'] - comparison['Best Points']
        
        # Calculate playoff probability confidence (distance from 50%)
        comparison['MC_Confidence'] = abs(comparison['Playoff Qualification %_MC'] - 50)
        comparison['ML_Confidence'] = abs(comparison['Playoff Qualification %_ML'] - 50)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=comparison['MC_Confidence'],
            y=comparison['ML_Confidence'],
            mode='markers+text',
            marker=dict(
                size=comparison['MC_Rank_Spread'],
                color=comparison['ML_Points_Spread'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="ML Points<br>Uncertainty"),
                opacity=0.7
            ),
            text=comparison['Team'].str[:3],
            textposition='middle center',
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'MC Confidence: %{x:.1f}<br>' +
                         'ML Confidence: %{y:.1f}<br>' +
                         'MC Rank Range: %{customdata[1]} positions<br>' +
                         'ML Points Range: %{customdata[2]} points<extra></extra>',
            customdata=list(zip(
                comparison['Team'],
                comparison['MC_Rank_Spread'],
                comparison['ML_Points_Spread']
            ))
        ))
        
        # Add diagonal line
        max_conf = max(comparison['MC_Confidence'].max(), comparison['ML_Confidence'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_conf], y=[0, max_conf],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Equal Confidence',
            showlegend=False
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Prediction Confidence Comparison<br>' +
                    '<sub>Size = MC rank uncertainty | Color = ML points uncertainty</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Monte Carlo Prediction Confidence<br><sub>Distance from 50% Playoff Probability (Higher = More Certain)</sub>',
            yaxis_title='Machine Learning Prediction Confidence<br><sub>Distance from 50% Playoff Probability (Higher = More Certain)</sub>',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_rank_distribution_chart(self, summary_df: pd.DataFrame, 
                                     simulation_results: Dict, 
                                     conference: str) -> go.Figure:
        """
        Create box plot showing rank distribution for each team.
        """
        fig = go.Figure()
        
        # Sort teams by average rank
        df_sorted = summary_df.sort_values('Average Final Rank')
        
        for _, row in df_sorted.iterrows():
            team_id = row['_team_id']
            team_name = row['Team']
            
            # Get rank distribution from simulation results
            rank_data = simulation_results.get(team_id, [])
            
            if rank_data:
                fig.add_trace(go.Box(
                    y=rank_data,
                    name=team_name,
                    boxmean='sd',  # Show mean and standard deviation
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Rank: %{y}<br>' +
                                 'Median: %{customdata[0]:.1f}<br>' +
                                 'Mean: %{customdata[1]:.1f}<extra></extra>',
                    customdata=[[row['Median Final Rank'], row['Average Final Rank']]] * len(rank_data)
                ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Final Rank Distributions',
                'x': 0.5,
                'font': {'size': 20}
            },
            yaxis_title='Final Regular Season Rank',
            xaxis_title='Teams',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        # Reverse y-axis so rank 1 is at the top
        fig.update_yaxes(autorange="reversed")
        
        # Add playoff line
        fig.add_hline(y=8.5, line_dash="dash", line_color="green", 
                     annotation_text="Playoff Line (Top 8)")
        
        return fig
    
    def create_points_analysis_chart(self, summary_df: pd.DataFrame, 
                                   conference: str) -> go.Figure:
        """
        Create scatter plot comparing current vs projected points.
        """
        fig = go.Figure()
        
        # Create bubble chart
        fig.add_trace(go.Scatter(
            x=summary_df['Current Points'],
            y=summary_df['Average Points'],
            mode='markers+text',
            text=summary_df['Team'].str[:3],  # team abbreviations
            textposition='top center',
            marker=dict(
                size=summary_df['Playoff Qualification %'] / 3,  # Size based on playoff prob
                color=summary_df['Playoff Qualification %'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Playoff<br>Probability (%)")
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Current Points: %{x}<br>' +
                         'Projected Points: %{y:.1f}<br>' +
                         'Playoff Probability: %{marker.color:.1f}%<br>' +
                         'Current Rank: %{customdata[1]}<extra></extra>',
            customdata=list(zip(summary_df['Team'], summary_df['Current Rank']))
        ))
        
        # Add diagonal line (y = x)
        min_points = min(summary_df['Current Points'].min(), summary_df['Average Points'].min())
        max_points = max(summary_df['Current Points'].max(), summary_df['Average Points'].max())
        
        fig.add_trace(go.Scatter(
            x=[min_points, max_points],
            y=[min_points, max_points],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Current = Projected',
            showlegend=True
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Current vs Projected Points<br>' +
                        f'<sub>Bubble size = Playoff Probability | Points above diagonal = Improving teams</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Current Season Points Earned<br><sub>Points from Games Already Played This Season</sub>',
            yaxis_title='Projected Final Season Points<br><sub>Current Points + Expected Points from Remaining Games</sub>',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_monte_carlo_dashboard(self, summary_df: pd.DataFrame, simulation_results: Dict,
                        conference: str, n_simulations: int) -> go.Figure:
        """
        Create a combined dashboard with multiple subplots for Monte Carlo.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Playoff Probabilities', 
                'Points: Current vs Projected',
                'Top 8 Teams - Rank Ranges',
                'Bottom 8 Teams - Rank Ranges'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "box"}]]
        )
        
        # Sort for consistent ordering
        df_sorted_by_playoff = summary_df.sort_values('Playoff Qualification %', ascending=False)
        df_sorted_by_rank = summary_df.sort_values('Average Final Rank', ascending=True)
        
        # 1. Playoff Probabilities (top-left)
        fig.add_trace(
            go.Bar(
                x=df_sorted_by_playoff['Team'][:8],
                y=df_sorted_by_playoff['Playoff Qualification %'][:8],
                name='Playoff %',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Points Analysis (top-right)
        fig.add_trace(
            go.Scatter(
                x=summary_df['Current Points'],
                y=summary_df['Average Points'],
                mode='markers',
                marker=dict(size=8, color='orange'),
                name='Teams',
                hovertemplate='<b>%{customdata}</b><br>' +
                     'Current Points: %{x}<br>' +
                     'Projected Points: %{y:.1f}<extra></extra>',
                customdata=summary_df['Team']
            ),
            row=1, col=2
        )
        
        # 3. Top 8 Teams Rank Ranges (bottom-left)
        top_8_teams = df_sorted_by_rank.head(8)
        for _, row in top_8_teams.iterrows():
            team_id = row['_team_id']
            rank_data = simulation_results.get(team_id, [])
            if rank_data:
                fig.add_trace(
                    go.Box(
                        y=rank_data,
                        name=row['Team'],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Bottom 8 Teams Rank Ranges (bottom-right)
        bottom_8_teams = df_sorted_by_rank.tail(8)
        for _, row in bottom_8_teams.iterrows():
            team_id = row['_team_id']
            rank_data = simulation_results.get(team_id, [])
            if rank_data:
                fig.add_trace(
                    go.Box(
                        y=rank_data,
                        name=row['Team'],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Monte Carlo Dashboard<br>' +
                       f'<sub>{n_simulations:,} simulations</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white'
        )
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=2)

        # Panel 1 (Top Left): Playoff Probabilities
        fig.update_xaxes(title_text="Teams<br><sub>Top 8 by Playoff Probability</sub>", row=1, col=1)
        fig.update_yaxes(title_text="Playoff Probability (%)<br><sub>Simulation-Based Likelihood</sub>", row=1, col=1)
        
        # Panel 2 (Top Right): Points Analysis Scatter
        fig.update_xaxes(title_text="Current Season Points<br><sub>Points from Completed Games</sub>", row=1, col=2)
        fig.update_yaxes(title_text="Projected Final Points<br><sub>Average from Simulations</sub>", row=1, col=2)
        
        # Panel 3 (Bottom Left): Top 8 Teams Rank Ranges
        fig.update_xaxes(title_text="Teams<br><sub>Likely Playoff Contenders</sub>", row=2, col=1)
        fig.update_yaxes(title_text="Final Regular Season Rank<br><sub>Distribution from Simulations</sub>", row=2, col=1)
        
        # Panel 4 (Bottom Right): Bottom 8 Teams Rank Ranges  
        fig.update_xaxes(title_text="Teams<br><sub>Unlikely Playoff Contenders</sub>", row=2, col=2)
        fig.update_yaxes(title_text="Final Regular Season Rank<br><sub>Distribution from Simulations</sub>", row=2, col=2)
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Monte Carlo Dashboard<br>' +
                    f'<sub>{n_simulations:,} simulations</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def create_ml_dashboard(self, summary_df, feature_importance, conference, variability_stats=None):
        """Enhanced ML-specific dashboard with variability if available."""
        if variability_stats:
            # 4-panel dashboard with variability
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Playoff Probabilities',
                    'Feature Importance Variability',
                    'Model Confidence Analysis',
                    'Points Projection Analysis'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # Panel 2: Feature importance with error bars
            if variability_stats:
                top_features = list(variability_stats.items())[:8]
                features = [self._format_single_feature_name(f) for f, _ in top_features]
                means = [stats['mean'] for _, stats in top_features]
                ci_lowers = [stats['ci_lower'] for _, stats in top_features]
                ci_uppers = [stats['ci_upper'] for _, stats in top_features]
                
                fig.add_trace(
                    go.Bar(
                        y=features,
                        x=means,
                        orientation='h',
                        name='Feature Importance',
                        marker_color='lightgreen',
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[upper - mean for upper, mean in zip(ci_uppers, means)],
                            arrayminus=[mean - lower for mean, lower in zip(means, ci_lowers)],
                            color='darkgreen'
                        )
                    ),
                    row=1, col=2
                )
        else:
            # Standard 3-panel dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Playoff Probabilities',
                    'Feature Importance',
                    'Model Confidence Analysis',
                    'Points Projection Analysis'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # Panel 2: Standard feature importance
            if feature_importance:
                top_features = list(feature_importance.items())[:8]
                features, importance = zip(*top_features)
                formatted_features = [self._format_single_feature_name(f) for f in features]
                
                fig.add_trace(
                    go.Bar(
                        y=formatted_features,
                        x=importance,
                        orientation='h',
                        name='Feature Importance',
                        marker_color='lightgreen'
                    ),
                    row=1, col=2
                )
        
        # Panel 1: Top 8 teams playoff probabilities
        top_teams = summary_df.head(8)
        fig.add_trace(
            go.Bar(
                x=top_teams['Team'],
                y=top_teams['Playoff Qualification %'],
                name='Playoff %',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Panel 3: ML Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=summary_df['Average Points'],
                y=summary_df['Playoff Qualification %'],
                mode='markers+text',
                text=summary_df['Team'].str[:3],  # team abbreviations
                textposition='top center',
                marker=dict(size=8, color='orange'),
                name='Confidence',
                hovertemplate='%{customdata}<br>Points: %{x:.1f}<br>Playoff %: %{y:.1f}%<extra></extra>',
                customdata=summary_df['Team']
            ),
            row=2, col=1
        )
        
        # Panel 4: Points analysis
        fig.add_trace(
            go.Scatter(
                x=summary_df['Current Points'],
                y=summary_df['Average Points'],
                mode='markers+text',
                text=summary_df['Team'].str[:3], # team abbreviations
                textposition='top center',
                marker=dict(size=8, color='red'),
                name='Points Projection',
                hovertemplate='%{customdata}<br>Current: %{x}<br>Projected: %{y:.1f}<extra></extra>',
                customdata=summary_df['Team']
            ),
            row=2, col=2
        )
        
        title_suffix = " with Variability Analysis" if variability_stats else ""
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference ML Dashboard{title_suffix}',
                'x': 0.5,
                'font': {'size': 20}
            },
            height=800,
            template='plotly_white',
            showlegend=False
        )

        fig.update_xaxes(title_text="Projected Final Season Points<br><sub>Current + Expected</sub>", row=2, col=1)
        fig.update_yaxes(title_text="Playoff Probability (%)<br><sub>Top 8 Likelihood</sub>", row=2, col=1)
        fig.update_xaxes(title_text="Current Points<br><sub>Games Played</sub>", row=2, col=2)
        fig.update_yaxes(title_text="Projected Points<br><sub>Season Total</sub>", row=2, col=2)
        
        return fig
    
    def create_feature_importance_variability_chart(self, variability_stats: Dict[str, Dict], 
                                               conference: str) -> go.Figure:
        """
        Create feature importance chart with confidence intervals and stability metrics.
        """
        if not variability_stats:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Feature importance variability data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Feature Importance Variability - Data Not Available",
                height=400
            )
            return fig
        
        # Get top 15 features by mean importance
        top_features = list(variability_stats.items())[:15]
        
        features = []
        means = []
        ci_lowers = []
        ci_uppers = []
        stabilities = []
        cvs = []
        
        for feature, stats in top_features:
            readable_name = self._format_single_feature_name(feature)
            features.append(readable_name)
            means.append(stats['mean'])
            ci_lowers.append(stats['ci_lower'])
            ci_uppers.append(stats['ci_upper'])
            stabilities.append(stats['stability_score'])
            cvs.append(stats['coefficient_of_variation'])
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=['Feature Importance with Confidence Intervals']
        )
        
        # Main bars showing mean importance
        fig.add_trace(
            go.Bar(
                y=features,
                x=means,
                orientation='h',
                name='Mean Importance',
                marker_color='lightblue',
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[upper - mean for upper, mean in zip(ci_uppers, means)],
                    arrayminus=[mean - lower for mean, lower in zip(means, ci_lowers)],
                    color='darkblue',
                    thickness=2
                ),
                hovertemplate='<b>%{y}</b><br>' +
                            'Mean Importance: %{x:.4f}<br>' +
                            '95% CI: %{customdata[0]:.4f} - %{customdata[1]:.4f}<br>' +
                            'Stability Score: %{customdata[2]:.1f}/100<br>' +
                            'Coef. of Variation: %{customdata[3]:.1f}%<extra></extra>',
                customdata=list(zip(ci_lowers, ci_uppers, stabilities, cvs))
            ),
            secondary_y=False
        )
        
        # Stability scores as scatter plot
        fig.add_trace(
            go.Scatter(
                y=features,
                x=stabilities,
                mode='markers',
                name='Stability Score',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond'
                ),
                hovertemplate='<b>%{y}</b><br>' +
                            'Stability Score: %{x:.1f}/100<br>' +
                            'Higher = More Consistent<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Mean Feature Importance Score<br><sub>Average Across Multiple Model Training Runs</sub>"
        )
        fig.update_yaxes(
            title_text="Prediction Factors<br><sub>Key Variables Used by ML Model</sub>"
        )
        
        # Secondary y-axis
        fig.update_yaxes(
            title_text="Stability Score (0-100)<br><sub>Consistency Across Training Runs</sub>",
            secondary_y=True,
            overlaying='y',
            side='right'
        )
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Feature Importance Variability Analysis<br>' +
                    '<sub>Error bars = 95% confidence intervals | Red diamonds = Stability scores</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=700,
            margin=dict(l=200, r=100),
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def create_stability_heatmap(self, variability_stats: Dict[str, Dict], 
                            conference: str) -> go.Figure:
        """
        Create heatmap showing feature importance stability metrics.
        """
        if not variability_stats:
            fig = go.Figure()
            fig.add_annotation(
                text="Stability data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get top 12 features
        top_features = list(variability_stats.items())[:12]
        
        features = [self._format_single_feature_name(f) for f, _ in top_features]
        metrics = ['Mean Importance', 'Std Deviation', 'Stability Score', 'Coef. of Variation']
        
        # Create data matrix
        data_matrix = []
        for _, stats in top_features:
            row = [
                stats['mean'],
                stats['std'], 
                stats['stability_score'],
                stats['coefficient_of_variation']
            ]
            data_matrix.append(row)
        
        # Normalize each column for better visualization
        data_matrix = np.array(data_matrix)
        normalized_matrix = np.zeros_like(data_matrix)
        
        for col in range(data_matrix.shape[1]):
            col_data = data_matrix[:, col]
            if col_data.max() != col_data.min():
                normalized_matrix[:, col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            else:
                normalized_matrix[:, col] = 0.5
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_matrix,
            x=metrics,
            y=features,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                        'Metric: %{x}<br>' +
                        'Normalized Value: %{z:.3f}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Feature Stability Heatmap<br>' +
                    '<sub>Normalized values (0-1) | Red = High, Blue = Low</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Stability Metrics<br><sub>Various Measures of Feature Consistency</sub>',
            yaxis_title='Prediction Factors<br><sub>Top Features by Mean Importance</sub>',
            height=600,
            template='plotly_white'
        )
        
        return fig

    def _format_single_feature_name(self, feature_name: str) -> str:
        """Format a single feature name for display."""
        name_mapping = {
            'is_home': 'Home Field Advantage',
            'team_xgf_per_game': 'Team Attack Strength (xG)',
            'team_xga_per_game': 'Team Defense Strength (xG)',
            'opp_xgf_per_game': 'Opponent Attack Strength',
            'opp_xga_per_game': 'Opponent Defense Strength', 
            'xg_diff': 'Team Goal Difference (xG)',
            'opp_xg_diff': 'Opponent Goal Difference (xG)',
            'team_form_points': 'Team Recent Form (Points)',
            'team_form_gf': 'Team Recent Goals For',
            'team_form_ga': 'Team Recent Goals Against',
            'opp_form_points': 'Opponent Recent Form',
            'opp_form_gf': 'Opponent Recent Goals For',
            'opp_form_ga': 'Opponent Recent Goals Against',
            'h2h_win_rate': 'Head-to-Head Record',
            'h2h_goals_for_avg': 'H2H Goals For Average',
            'h2h_goals_against_avg': 'H2H Goals Against Average',
            'h2h_games_played': 'H2H Games Played',
            'team_rest_days': 'Team Rest Days',
            'opp_rest_days': 'Opponent Rest Days',
            'month': 'Season Month',
            'day_of_week': 'Day of Week',
            'is_weekend': 'Weekend Game'
        }
        
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())
    
    def create_comparison_dashboard(self, comparison_data, feature_importance, conference, n_simulations, variability_stats=None):
        """Dashboard comparing both methods."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Playoff Probability Comparison',
                'Method Agreement Analysis',
                'Feature Importance (ML)',
                'Prediction Confidence'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Get comparison dataframes
        mc_df = comparison_data['monte_carlo']['summary_df']
        ml_df = comparison_data['machine_learning']['summary_df']
        
        comparison = pd.merge(
            mc_df[['_team_id', 'Team', 'Playoff Qualification %']],
            ml_df[['_team_id', 'Playoff Qualification %']],
            on='_team_id',
            suffixes=('_MC', '_ML')
        )
        
        # Playoff comparison
        if feature_importance:
            if variability_stats:
                # Use variability stats with confidence intervals
                top_features = list(variability_stats.items())[:8]
                features = [self._format_single_feature_name(f) for f, _ in top_features]
                means = [stats['mean'] for _, stats in top_features]
                ci_lowers = [stats['ci_lower'] for _, stats in top_features]
                ci_uppers = [stats['ci_upper'] for _, stats in top_features]
                
                fig.add_trace(
                    go.Bar(
                        y=features,
                        x=means,
                        orientation='h',
                        name='Feature Importance',
                        marker_color='lightgreen',
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[upper - mean for upper, mean in zip(ci_uppers, means)],
                            arrayminus=[mean - lower for mean, lower in zip(means, ci_lowers)],
                            color='darkgreen',
                            thickness=2
                        ),
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Mean: %{x:.4f}<br>' +
                                    '95% CI: %{customdata[0]:.4f} - %{customdata[1]:.4f}<extra></extra>',
                        customdata=list(zip(ci_lowers, ci_uppers))
                    ),
                    row=2, col=1
                )
            else:
                # Fallback to basic feature importance
                top_features = list(feature_importance.items())[:8]
                features, importance = zip(*top_features)
                
                fig.add_trace(
                    go.Bar(
                        y=features,
                        x=importance,
                        orientation='h',
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
        
        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 100], y=[0, 100],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Method agreement (difference analysis)
        comparison['diff'] = abs(comparison['Playoff Qualification %_ML'] - comparison['Playoff Qualification %_MC'])
        comparison_sorted = comparison.sort_values('diff', ascending=False).head(8)
        
        fig.add_trace(
            go.Scatter(
                x=comparison_sorted['diff'],
                y=list(range(len(comparison_sorted))),
                mode='markers',
                marker=dict(size=10, color='red'),
                text=comparison_sorted['Team'],
                textposition='middle right'
            ),
            row=1, col=2
        )
        
        # Feature importance
        if feature_importance:
            top_features = list(feature_importance.items())[:8]
            features, importance = zip(*top_features)
            
            fig.add_trace(
                go.Bar(
                    y=features,
                    x=importance,
                    orientation='h',
                    marker_color='green'
                ),
                row=2, col=1
            )
        
        # Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=comparison['Playoff Qualification %_MC'],
                y=comparison['diff'],
                mode='markers',
                marker=dict(size=8, color='purple'),
                text=comparison['Team'],
                hovertemplate='<b>%{text}</b><br>' +
                             'MC Playoff %: %{x:.1f}%<br>' +
                             'Difference: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Method Comparison Dashboard<br>' +
                       f'<sub>MC: {n_simulations:,} simulations | ML: Model predictions</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=800,
            template='plotly_white',
            showlegend=False
        )

        # Panel 1 (Top Left): Playoff Probability Comparison
        fig.update_xaxes(title_text="Monte Carlo Playoff Probability (%)<br><sub>Simulation-Based Predictions</sub>", row=1, col=1)
        fig.update_yaxes(title_text="Machine Learning Playoff Probability (%)<br><sub>Model-Based Predictions</sub>", row=1, col=1)
        
        # Panel 2 (Top Right): Method Agreement Analysis
        fig.update_xaxes(title_text="Prediction Disagreement<br><sub>Absolute Difference in Playoff %</sub>", row=1, col=2)
        fig.update_yaxes(title_text="Teams<br><sub>Ranked by Method Disagreement</sub>", row=1, col=2)
        
        # Panel 3 (Bottom Left): Feature Importance  
        fig.update_xaxes(title_text="Feature Importance Score<br><sub>ML Model Prediction Factors</sub>", row=2, col=1)
        fig.update_yaxes(title_text="Prediction Factors<br><sub>Key Variables</sub>", row=2, col=1)
        
        # Panel 4 (Bottom Right): Prediction Confidence
        fig.update_xaxes(title_text="Monte Carlo Playoff Probability (%)<br><sub>Baseline Method</sub>", row=2, col=2)
        fig.update_yaxes(title_text="Method Disagreement<br><sub>Difference in Predictions</sub>", row=2, col=2)
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Method Comparison Dashboard<br>' +
                    f'<sub>MC: {n_simulations:,} simulations | ML: Model predictions</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=800,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    # Keeping the original dashboard method for backward compatibility
    def create_dashboard(self, summary_df: pd.DataFrame, simulation_results: Dict,
                        conference: str, n_simulations: int) -> go.Figure:
        """
        Create a combined dashboard with multiple subplots (backward compatibility).
        """
        return self.create_monte_carlo_dashboard(summary_df, simulation_results, conference, n_simulations)
    
    def show_charts_summary(self, chart_files: Dict[str, str], conference: str):
        """
        Print a summary of generated charts.
        """
        print(f"\n{'='*60}")
        print(f"Generated Charts for {conference.title()} Conference")
        print(f"{'='*60}")
        
        for chart_name, file_path in chart_files.items():
            print(f" {chart_name.replace('_', ' ').title()}")
            print(f"      → {file_path}")
        
        print(f"\n Open any HTML file in your browser to view interactive charts!")