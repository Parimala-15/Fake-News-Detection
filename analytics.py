import json
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

class PredictionTracker:
    """Tracks and analyzes prediction history"""
    
    def __init__(self, data_file="prediction_history.json"):
        self.data_file = data_file
        self.predictions = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load prediction history from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_data(self):
        """Save prediction history to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.predictions, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Failed to save prediction history: {e}")
    
    def add_prediction(self, text: str, prediction: str, confidence: float, 
                      model_type: str = "user_model"):
        """Add a new prediction to history"""
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "text": text[:500],  # Limit text length for storage
            "text_length": len(text),
            "prediction": prediction,
            "confidence": float(confidence),
            "model_type": model_type,
            "session_id": st.session_state.get('session_id', 'unknown')
        }
        
        self.predictions.append(prediction_data)
        
        # Keep only last 1000 predictions to manage file size
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
        
        self._save_data()
    
    def get_recent_predictions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get predictions from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = []
        
        for pred in self.predictions:
            try:
                pred_date = datetime.fromisoformat(pred['timestamp'])
                if pred_date >= cutoff_date:
                    recent.append(pred)
            except (ValueError, KeyError):
                continue
        
        return recent
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Generate analytics data for the dashboard"""
        recent_predictions = self.get_recent_predictions(days)
        
        if not recent_predictions:
            return {
                "total_predictions": 0,
                "fake_percentage": 0,
                "real_percentage": 0,
                "avg_confidence": 0,
                "daily_counts": {},
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
                "text_length_stats": {"avg": 0, "min": 0, "max": 0}
            }
        
        # Basic statistics
        total = len(recent_predictions)
        fake_count = sum(1 for p in recent_predictions if p['prediction'] == 'FAKE')
        real_count = total - fake_count
        
        confidences = [p['confidence'] for p in recent_predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Daily counts
        daily_counts = {}
        for pred in recent_predictions:
            try:
                date = datetime.fromisoformat(pred['timestamp']).date().isoformat()
                daily_counts[date] = daily_counts.get(date, 0) + 1
            except (ValueError, KeyError):
                continue
        
        # Confidence distribution
        high_conf = sum(1 for c in confidences if c >= 0.8)
        medium_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        # Text length statistics
        text_lengths = [p.get('text_length', 0) for p in recent_predictions]
        text_length_stats = {
            "avg": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0
        }
        
        return {
            "total_predictions": total,
            "fake_percentage": (fake_count / total * 100) if total > 0 else 0,
            "real_percentage": (real_count / total * 100) if total > 0 else 0,
            "avg_confidence": avg_confidence,
            "daily_counts": daily_counts,
            "confidence_distribution": {
                "high": high_conf,
                "medium": medium_conf,
                "low": low_conf
            },
            "text_length_stats": text_length_stats
        }
    
    def get_prediction_trends(self, days: int = 30) -> pd.DataFrame:
        """Get prediction trends as a DataFrame for plotting"""
        recent_predictions = self.get_recent_predictions(days)
        
        if not recent_predictions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Group by date and prediction type
        daily_stats = df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
        daily_stats = daily_stats.reset_index()
        
        # Add percentage columns
        daily_stats['total'] = daily_stats.get('FAKE', 0) + daily_stats.get('REAL', 0)
        daily_stats['fake_percentage'] = (daily_stats.get('FAKE', 0) / daily_stats['total'] * 100).fillna(0)
        daily_stats['real_percentage'] = (daily_stats.get('REAL', 0) / daily_stats['total'] * 100).fillna(0)
        
        return daily_stats
    
    def clear_history(self):
        """Clear all prediction history"""
        self.predictions = []
        self._save_data()

def display_analytics_dashboard(tracker: PredictionTracker):
    """Display the analytics dashboard in Streamlit"""
    st.markdown("## 📊 Prediction Analytics")
    
    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col1:
        time_period = st.selectbox(
            "Time Period",
            [7, 14, 30, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
    with col2:
        if st.button("Clear History", type="secondary"):
            tracker.clear_history()
            st.success("History cleared!")
            st.rerun()
    
    # Get analytics data
    analytics = tracker.get_analytics_data(time_period)
    
    if analytics["total_predictions"] == 0:
        st.info("No predictions recorded yet. Start analyzing some news to see analytics!")
        return
    
    # Key metrics
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predictions",
            analytics["total_predictions"]
        )
    
    with col2:
        st.metric(
            "Average Confidence",
            f"{analytics['avg_confidence']:.1%}"
        )
    
    with col3:
        st.metric(
            "Fake News Detected",
            f"{analytics['fake_percentage']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Real News Detected", 
            f"{analytics['real_percentage']:.1f}%"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Prediction Distribution")
        if analytics["total_predictions"] > 0:
            chart_data = pd.DataFrame({
                'Type': ['Fake News', 'Real News'],
                'Count': [
                    analytics["fake_percentage"],
                    analytics["real_percentage"]
                ]
            })
            st.bar_chart(chart_data.set_index('Type'))
    
    with col2:
        st.markdown("### Confidence Distribution")
        conf_data = analytics["confidence_distribution"]
        if sum(conf_data.values()) > 0:
            conf_chart = pd.DataFrame({
                'Confidence Level': ['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)'],
                'Count': [conf_data['high'], conf_data['medium'], conf_data['low']]
            })
            st.bar_chart(conf_chart.set_index('Confidence Level'))
    
    # Trends over time
    if analytics["total_predictions"] > 5:  # Only show if enough data
        st.markdown("### Prediction Trends")
        trends_df = tracker.get_prediction_trends(time_period)
        
        if not trends_df.empty:
            # Line chart for daily predictions
            chart_data = trends_df[['date', 'fake_percentage', 'real_percentage']].set_index('date')
            chart_data.columns = ['Fake News %', 'Real News %']
            st.line_chart(chart_data)
    
    # Recent predictions table
    st.markdown("### Recent Predictions")
    recent = tracker.get_recent_predictions(7)  # Last 7 days
    
    if recent:
        # Prepare data for table
        table_data = []
        for pred in recent[-10:]:  # Show last 10
            table_data.append({
                "Time": datetime.fromisoformat(pred['timestamp']).strftime("%m/%d %H:%M"),
                "Prediction": pred['prediction'],
                "Confidence": f"{pred['confidence']:.1%}",
                "Text Preview": pred['text'][:100] + "..." if len(pred['text']) > 100 else pred['text']
            })
        
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent predictions to display.")
    
    # Text analysis insights
    if analytics["total_predictions"] > 0:
        st.markdown("### Text Analysis Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Text Length",
                f"{analytics['text_length_stats']['avg']:.0f} chars"
            )
        
        with col2:
            st.metric(
                "Shortest Text",
                f"{analytics['text_length_stats']['min']} chars"
            )
        
        with col3:
            st.metric(
                "Longest Text",
                f"{analytics['text_length_stats']['max']} chars"
            )