import React, { useState, useEffect } from 'react';
import AlertsPanel from './AlertsPanel';
import GraphVisualization from './GraphVisualization';
import GeospatialMap from './GeospatialMap';
import MetricsPanel from './MetricsPanel';
import RealTimeEvents from './RealTimeEvents';
import { useWebSocket } from '../hooks/useWebSocket';
import { getDashboardSummary } from '../services/api';
import { DashboardSummary } from '../types';

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'overview' | 'alerts' | 'graph' | 'map' | 'metrics'>('overview');

  // WebSocket connection for real-time updates
  const { isConnected, subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    loadDashboardData();
    
    // Subscribe to real-time updates
    if (isConnected) {
      subscribe(['events', 'alerts', 'metrics']);
    }
    
    return () => {
      unsubscribe();
    };
  }, [isConnected]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const data = await getDashboardSummary();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const renderStatusIndicator = () => {
    const status = isConnected ? 'connected' : 'disconnected';
    const color = isConnected ? '#10B981' : '#EF4444';
    
    return (
      <div className="status-indicator">
        <div 
          className="status-dot" 
          style={{ backgroundColor: color }}
        />
        <span className="status-text">
          {status} • {dashboardData?.systemHealth || 0}% health
        </span>
      </div>
    );
  };

  const renderOverview = () => (
    <div className="overview-grid">
      <div className="overview-card">
        <h3>System Status</h3>
        <div className="overview-stats">
          <div className="stat">
            <span className="stat-label">Events (24h)</span>
            <span className="stat-value">{dashboardData?.totalEvents || 0}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Active Alerts</span>
            <span className="stat-value critical">{dashboardData?.totalAlerts || 0}</span>
          </div>
          <div className="stat">
            <span className="stat-label">High-Risk Transactions</span>
            <span className="stat-value warning">{dashboardData?.highRiskTransactions || 0}</span>
          </div>
        </div>
      </div>
      
      <div className="overview-card">
        <h3>Recent Critical Alerts</h3>
        <div className="alert-list">
          {dashboardData?.criticalAlerts?.map((alert, index) => (
            <div key={index} className="alert-item">
              <div className="alert-icon">⚠️</div>
              <div className="alert-content">
                <div className="alert-title">{alert.alertType}</div>
                <div className="alert-description">{alert.description}</div>
                <div className="alert-time">
                  {new Date(alert.createdAt).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="overview-card">
        <h3>Performance Metrics</h3>
        <div className="metrics-grid">
          {dashboardData?.performanceMetrics?.map((metric, index) => (
            <div key={index} className="metric-item">
              <div className="metric-label">{metric.metricType}</div>
              <div className="metric-value">
                {metric.value.toFixed(2)} {metric.unit}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="overview-card">
        <h3>Real-Time Events</h3>
        <RealTimeEvents />
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="loading-spinner" />
        <p>Loading FinSecure Nexus Dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">❌</div>
        <h3>Dashboard Error</h3>
        <p>{error}</p>
        <button onClick={loadDashboardData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="dashboard-title">
          <h2>Security Operations Center</h2>
          <p>Multi-domain threat detection and response</p>
        </div>
        {renderStatusIndicator()}
      </div>
      
      <div className="dashboard-nav">
        <button 
          className={`nav-button ${selectedView === 'overview' ? 'active' : ''}`}
          onClick={() => setSelectedView('overview')}
        >
          📊 Overview
        </button>
        <button 
          className={`nav-button ${selectedView === 'alerts' ? 'active' : ''}`}
          onClick={() => setSelectedView('alerts')}
        >
          🚨 Alerts
        </button>
        <button 
          className={`nav-button ${selectedView === 'graph' ? 'active' : ''}`}
          onClick={() => setSelectedView('graph')}
        >
          🕸️ Graph Analysis
        </button>
        <button 
          className={`nav-button ${selectedView === 'map' ? 'active' : ''}`}
          onClick={() => setSelectedView('map')}
        >
          🗺️ Geospatial
        </button>
        <button 
          className={`nav-button ${selectedView === 'metrics' ? 'active' : ''}`}
          onClick={() => setSelectedView('metrics')}
        >
          📈 Metrics
        </button>
      </div>
      
      <div className="dashboard-content">
        {selectedView === 'overview' && renderOverview()}
        {selectedView === 'alerts' && <AlertsPanel />}
        {selectedView === 'graph' && <GraphVisualization />}
        {selectedView === 'map' && <GeospatialMap />}
        {selectedView === 'metrics' && <MetricsPanel />}
      </div>
    </div>
  );
};

export default Dashboard;
