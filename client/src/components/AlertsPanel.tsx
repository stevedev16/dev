import React, { useState, useEffect } from 'react';
import { getAlerts, updateAlert } from '../services/api';
import { Alert } from '../types';

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');

  useEffect(() => {
    loadAlerts();
  }, []);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const data = await getAlerts();
      setAlerts(data);
      setError(null);
    } catch (err) {
      console.error('Error loading alerts:', err);
      setError('Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateAlert = async (alertId: number, updates: Partial<Alert>) => {
    try {
      const updatedAlert = await updateAlert(alertId, updates);
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId ? updatedAlert : alert
      ));
    } catch (err) {
      console.error('Error updating alert:', err);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#DC2626';
      case 'high': return '#EA580C';
      case 'medium': return '#D97706';
      case 'low': return '#65A30D';
      default: return '#6B7280';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return '#DC2626';
      case 'acknowledged': return '#D97706';
      case 'resolved': return '#059669';
      case 'false_positive': return '#6B7280';
      default: return '#6B7280';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    const severityMatch = selectedSeverity === 'all' || alert.severity === selectedSeverity;
    const statusMatch = selectedStatus === 'all' || alert.status === selectedStatus;
    return severityMatch && statusMatch;
  });

  const renderAlertActions = (alert: Alert) => (
    <div className="alert-actions">
      {alert.status === 'open' && (
        <button 
          className="action-button acknowledge"
          onClick={() => handleUpdateAlert(alert.id, { status: 'acknowledged' })}
        >
          Acknowledge
        </button>
      )}
      {alert.status !== 'resolved' && (
        <button 
          className="action-button resolve"
          onClick={() => handleUpdateAlert(alert.id, { status: 'resolved' })}
        >
          Resolve
        </button>
      )}
      <button 
        className="action-button false-positive"
        onClick={() => handleUpdateAlert(alert.id, { status: 'false_positive' })}
      >
        Mark False Positive
      </button>
    </div>
  );

  if (loading) {
    return (
      <div className="alerts-loading">
        <div className="loading-spinner" />
        <p>Loading alerts...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alerts-error">
        <div className="error-icon">❌</div>
        <h3>Error Loading Alerts</h3>
        <p>{error}</p>
        <button onClick={loadAlerts} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="alerts-panel">
      <div className="alerts-header">
        <h3>Security Alerts</h3>
        <div className="alerts-filters">
          <select 
            value={selectedSeverity} 
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          
          <select 
            value={selectedStatus} 
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Statuses</option>
            <option value="open">Open</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
            <option value="false_positive">False Positive</option>
          </select>
        </div>
      </div>
      
      <div className="alerts-stats">
        <div className="stat">
          <span className="stat-label">Total Alerts</span>
          <span className="stat-value">{filteredAlerts.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Critical</span>
          <span className="stat-value critical">
            {filteredAlerts.filter(a => a.severity === 'critical').length}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Open</span>
          <span className="stat-value warning">
            {filteredAlerts.filter(a => a.status === 'open').length}
          </span>
        </div>
      </div>
      
      <div className="alerts-list">
        {filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            <div className="no-alerts-icon">🎉</div>
            <h4>No alerts found</h4>
            <p>No alerts match your current filters</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div key={alert.id} className="alert-card">
              <div className="alert-header">
                <div className="alert-severity">
                  <span 
                    className="severity-badge"
                    style={{ backgroundColor: getSeverityColor(alert.severity) }}
                  >
                    {alert.severity.toUpperCase()}
                  </span>
                  <span className="alert-type">{alert.alertType}</span>
                </div>
                <div className="alert-status">
                  <span 
                    className="status-badge"
                    style={{ color: getStatusColor(alert.status) }}
                  >
                    {alert.status.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
              </div>
              
              <div className="alert-content">
                <h4 className="alert-title">{alert.description}</h4>
                <div className="alert-details">
                  <div className="alert-meta">
                    <span className="meta-item">
                      <strong>Confidence:</strong> {(alert.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="meta-item">
                      <strong>Created:</strong> {new Date(alert.createdAt).toLocaleString()}
                    </span>
                    {alert.correlatedAlerts && alert.correlatedAlerts.length > 0 && (
                      <span className="meta-item">
                        <strong>Correlated:</strong> {alert.correlatedAlerts.length} alerts
                      </span>
                    )}
                  </div>
                  
                  {alert.responseActions && alert.responseActions.length > 0 && (
                    <div className="response-actions">
                      <strong>Recommended Actions:</strong>
                      <ul>
                        {alert.responseActions.map((action: any, index: number) => (
                          <li key={index}>{action.description}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
              
              {renderAlertActions(alert)}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertsPanel;
