from flask import Flask, render_template, jsonify, request
from dashboard_generator import DashboardDataGenerator
import os

app = Flask(__name__, template_folder=os.path.abspath("templates"))
generator = DashboardDataGenerator()

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/api/engine-data")
def engine_data():
    """Get all engine data with enhanced filtering"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        # Get filter parameters
        risk_filter = request.args.get('risk_category', 'all')
        search_term = request.args.get('search', '')
        rul_min = request.args.get('rul_min', type=int)
        rul_max = request.args.get('rul_max', type=int)
        
        data = generator.dashboard_data.copy()
        original_count = len(data['engines'])
        
        # Apply filters
        filtered_engines = data['engines'].copy()
        
        # Risk category filter
        if risk_filter and risk_filter.lower() != 'all':
            filtered_engines = [
                engine for engine in filtered_engines 
                if engine.get('Risk_Category', '').lower() == risk_filter.lower()
            ]
        
        # Search filter
        if search_term:
            search_lower = search_term.lower()
            filtered_engines = [
                engine for engine in filtered_engines
                if (search_lower in str(engine.get('id', '')).lower() or
                    search_lower in str(engine.get('Health_Status', '')).lower() or
                    search_lower in str(engine.get('Risk_Category', '')).lower() or
                    search_lower in str(engine.get('Action_Required', '')).lower() or
                    search_lower in str(engine.get('Failure_Reason', '')).lower())
            ]
        
        # RUL range filter
        if rul_min is not None:
            filtered_engines = [
                engine for engine in filtered_engines
                if engine.get('Predicted_RUL', 0) >= rul_min
            ]
        
        if rul_max is not None:
            filtered_engines = [
                engine for engine in filtered_engines
                if engine.get('Predicted_RUL', 0) <= rul_max
            ]
        
        data['engines'] = filtered_engines
        data['filtered'] = len(filtered_engines) != original_count
        data['filter_info'] = {
            'risk_category': risk_filter,
            'search_term': search_term,
            'rul_min': rul_min,
            'rul_max': rul_max,
            'showing': len(filtered_engines),
            'total': original_count
        }
        
        # Update summary for filtered data
        if data['filtered']:
            data['filtered_summary'] = {
                'totalEngines': len(filtered_engines),
                'safeEngines': sum(1 for e in filtered_engines if e.get('Risk_Category') == 'Safe'),
                'warningEngines': sum(1 for e in filtered_engines if e.get('Risk_Category') == 'Warning'),
                'criticalEngines': sum(1 for e in filtered_engines if e.get('Risk_Category') == 'Critical'),
                'averageRUL': round(sum(e.get('Predicted_RUL', 0) for e in filtered_engines) / len(filtered_engines), 2) if filtered_engines else 0
            }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/engine/<engine_id>")
def engine_details(engine_id):
    """Get detailed information for a specific engine"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        # Find the specific engine
        engine = next((e for e in generator.dashboard_data['engines'] if e['id'] == engine_id), None)
        
        if not engine:
            return jsonify({'error': 'Engine not found'}), 404
        
        # Add additional computed details
        engine_details = engine.copy()
        engine_details['maintenance_priority'] = generator.get_maintenance_priority(engine)
        engine_details['failure_analysis'] = generator.get_failure_analysis(engine)
        engine_details['recommended_actions'] = generator.get_recommended_actions(engine)
        
        return jsonify(engine_details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/refresh-data")
def refresh():
    """Refresh all dashboard data"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        return jsonify({
            'status': 'success', 
            'lastUpdated': generator.dashboard_data['lastUpdated'],
            'message': 'Dashboard data refreshed successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/alerts")
def get_alerts():
    """Get current system alerts with enhanced details"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        alerts = generator.generate_enhanced_alerts()
        return jsonify({'alerts': alerts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/summary")
def get_summary():
    """Get dashboard summary statistics"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        return jsonify(generator.dashboard_data['summary'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/charts/risk-distribution")
def risk_distribution():
    """Get data for risk distribution pie chart"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        summary = generator.dashboard_data['summary']
        chart_data = {
            'labels': ['Safe', 'Warning', 'Critical'],
            'data': [summary['safeEngines'], summary['warningEngines'], summary['criticalEngines']],
            'colors': ['#10b981', '#f59e0b', '#ef4444']
        }
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/charts/rul-histogram")
def rul_histogram():
    """Get data for RUL histogram"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        engines = generator.dashboard_data['engines']
        rul_values = [engine['Predicted_RUL'] for engine in engines]
        
        # Create histogram bins
        bins = ['0-50', '51-100', '101-150', '151-200', '201-250', '251+']
        counts = [0] * 6
        
        for rul in rul_values:
            if rul <= 50:
                counts[0] += 1
            elif rul <= 100:
                counts[1] += 1
            elif rul <= 150:
                counts[2] += 1
            elif rul <= 200:
                counts[3] += 1
            elif rul <= 250:
                counts[4] += 1
            else:
                counts[5] += 1
        
        chart_data = {
            'labels': bins,
            'data': counts,
            'backgroundColor': ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#10b981', '#059669'],
            'borderColor': 'rgba(255, 255, 255, 0.8)',
            'borderWidth': 2
        }
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/charts/failure-risk-scatter")
def failure_risk_scatter():
    """Get data for failure risk vs RUL scatter plot"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        engines = generator.dashboard_data['engines']
        scatter_data = []
        
        for engine in engines:
            color = '#10b981'  # Safe - green
            if engine['Risk_Category'] == 'Warning':
                color = '#f59e0b'  # Warning - orange
            elif engine['Risk_Category'] == 'Critical':
                color = '#ef4444'  # Critical - red
            
            scatter_data.append({
                'x': engine['Predicted_RUL'],
                'y': engine['Failure_Risk'],
                'label': engine['id'],
                'color': color,
                'risk': engine['Risk_Category'],
                'size': 8 + (engine['Failure_Risk'] / 10)  # Variable size based on risk
            })
        
        return jsonify(scatter_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/charts/health-trends")
def health_trends():
    """Get simulated health trends over time"""
    try:
        generator.load_results()
        generator.prepare_dashboard_data()
        
        import datetime
        from datetime import timedelta
        
        base_date = datetime.datetime.now() - timedelta(days=30)
        dates = [(base_date + timedelta(days=i)).strftime('%m/%d') for i in range(31)]
        
        summary = generator.dashboard_data['summary']
        
        # More realistic trends based on current data
        safe_base = summary['safeEngines']
        warning_base = summary['warningEngines']
        critical_base = summary['criticalEngines']
        
        safe_trend = [max(0, safe_base + np.random.randint(-2, 3)) for _ in range(31)]
        warning_trend = [max(0, warning_base + np.random.randint(-1, 2)) for _ in range(31)]
        critical_trend = [max(0, critical_base + np.random.randint(-1, 2)) for _ in range(31)]
        
        chart_data = {
            'labels': dates,
            'datasets': [
                {
                    'label': 'Safe Engines',
                    'data': safe_trend,
                    'borderColor': '#10b981',
                    'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                    'fill': True,
                    'tension': 0.4
                },
                {
                    'label': 'Warning Engines',
                    'data': warning_trend,
                    'borderColor': '#f59e0b',
                    'backgroundColor': 'rgba(245, 158, 11, 0.1)',
                    'fill': True,
                    'tension': 0.4
                },
                {
                    'label': 'Critical Engines',
                    'data': critical_trend,
                    'borderColor': '#ef4444',
                    'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                    'fill': True,
                    'tension': 0.4
                }
            ]
        }
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/export")
def export_data():
    """Export filtered data as CSV"""
    try:
        # Get same filters as engine_data endpoint
        risk_filter = request.args.get('risk_category', 'all')
        search_term = request.args.get('search', '')
        
        generator.load_results()
        generator.prepare_dashboard_data()
        
        # Apply same filtering logic
        engines = generator.dashboard_data['engines'].copy()
        
        if risk_filter and risk_filter.lower() != 'all':
            engines = [e for e in engines if e.get('Risk_Category', '').lower() == risk_filter.lower()]
        
        if search_term:
            search_lower = search_term.lower()
            engines = [e for e in engines if search_lower in str(e.get('id', '')).lower()]
        
        # Convert to CSV format
        import io
        import csv
        
        output = io.StringIO()
        if engines:
            fieldnames = engines[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(engines)
        
        return jsonify({
            'status': 'success',
            'data': output.getvalue(),
            'filename': f'engine_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Ensure directories exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)