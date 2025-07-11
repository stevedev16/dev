import React, { useState, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { getSystemMetrics, getLatestMetrics } from '../services/api';
import { SystemMetric } from '../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const MetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetricType, setSelectedMetricType] = useState<string>('all');

  useEffect(() => {
    loadMetrics();
    
    // Auto-refresh metrics every 30 seconds
    const interval = setInterval(loadMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const data = await getLatestMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Error loading metrics:', err);
      setError('Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  const getMetricsByType = (type: string) => {
    return metrics.filter(m => m.metricType === type);
  };

  const renderSystemHealthChart = () => {
    const healthMetrics = getMetricsByType('system_health');
    if (healthMetrics.length === 0) return null;

    const data = {
      labels: healthMetrics.map(m => new Date(m.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'System Health %',
        data: healthMetrics.map(m => m.value),
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'System Health Over Time'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      }
    };

    return <Line data={data} options={options} />;
  };

  const renderLatencyChart = () => {
    const latencyMetrics = getMetricsByType('processing_latency');
    if (latencyMetrics.length === 0) return null;

    const data = {
      labels: latencyMetrics.map(m => new Date(m.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'Processing Latency (ms)',
        data: latencyMetrics.map(m => m.value),
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Processing Latency'
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    };

    return <Line data={data} options={options} />;
  };

  const renderThroughputChart = () => {
    const throughputMetrics = getMetricsByType('throughput');
    if (throughputMetrics.length === 0) return null;

    const data = {
      labels: throughputMetrics.map(m => new Date(m.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'Events per Second',
        data: throughputMetrics.map(m => m.value),
        backgroundColor: 'rgba(245, 158, 11, 0.8)',
        borderColor: '#F59E0B',
        borderWidth: 1
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'System Throughput'
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    };

    return <Bar data={data} options={options} />;
  };

  const renderResourceUsageChart = () => {
    const cpuMetrics = getMetricsByType('cpu_usage');
    const memoryMetrics = getMetricsByType('memory_usage');
    
    if (cpuMetrics.length === 0 || memoryMetrics.length === 0) return null;

    const data = {
      labels: cpuMetrics.map(m => new Date(m.timestamp).toLocaleTimeString()),
      datasets: [
        {
          label: 'CPU Usage %',
          data: cpuMetrics.map(m => m.value),
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.4
        },
        {
          label: 'Memory Usage %',
          data: memoryMetrics.map(m => m.value),
          borderColor: '#8B5CF6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          fill: true,
          tension: 0.4
        }
      ]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Resource Usage'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      }
    };

    return <Line data={data} options={options} />;
  };

  const renderMetricTypeDistribution = () => {
    const typeDistribution = metrics.reduce((acc, metric) => {
      acc[metric.metricType] = (acc[metric.metricType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const data = {
      labels: Object.keys(typeDistribution),
      datasets: [{
        data: Object.values(typeDistribution),
        backgroundColor: [
          '#3B82F6',
          '#10B981',
          '#F59E0B',
          '#EF4444',
          '#8B5CF6',
          '#EC4899'
        ]
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'right' as const,
        },
        title: {
          display: true,
          text: 'Metric Type Distribution'
        }
      }
    };

    return <Doughnut data={data} options={options} />;
  };

  const renderMetricsSummary = () => {
    const latestMetrics = metrics.slice(0, 10);
    const avgLatency = getMetricsByType('processing_latency')
      .reduce((sum, m) => sum + m.value, 0) / Math.max(getMetricsByType('processing_latency').length, 1);
    
    const avgThroughput = getMetricsByType('throughput')
      .reduce((sum, m) => sum + m.value, 0) / Math.max(getMetricsByType('throughput').length, 1);
    
    const systemHealth = getMetricsByType('system_health')[0]?.value || 0;

    return (
      <div className="metrics-summary">
        <div className="summary-cards">
          <div className="summary-card">
            <div className="card-icon">⚡</div>
            <div className="card-content">
              <div className="card-title">Avg Latency</div>
              <div className="card-value">{avgLatency.toFixed(2)}ms</div>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="card-icon">📊</div>
            <div className="card-content">
              <div className="card-title">Throughput</div>
              <div className="card-value">{avgThroughput.toFixed(0)} eps</div>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="card-icon">💚</div>
            <div className="card-content">
              <div className="card-title">System Health</div>
              <div className="card-value">{systemHealth.toFixed(1)}%</div>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="card-icon">📈</div>
            <div className="card-content">
              <div className="card-title">Total Metrics</div>
              <div className="card-value">{metrics.length}</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="metrics-loading">
        <div className="loading-spinner" />
        <p>Loading metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="metrics-error">
        <div className="error-icon">❌</div>
        <h3>Error Loading Metrics</h3>
        <p>{error}</p>
        <button onClick={loadMetrics} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="metrics-panel">
      <div className="metrics-header">
        <h3>System Metrics</h3>
        <div className="metrics-controls">
          <select 
            value={selectedMetricType} 
            onChange={(e) => setSelectedMetricType(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Metrics</option>
            <option value="system_health">System Health</option>
            <option value="processing_latency">Processing Latency</option>
            <option value="throughput">Throughput</option>
            <option value="cpu_usage">CPU Usage</option>
            <option value="memory_usage">Memory Usage</option>
          </select>
          
          <button onClick={loadMetrics} className="refresh-button">
            🔄 Refresh
          </button>
        </div>
      </div>
      
      {renderMetricsSummary()}
      
      <div className="metrics-charts">
        <div className="chart-container">
          {renderSystemHealthChart()}
        </div>
        
        <div className="chart-container">
          {renderLatencyChart()}
        </div>
        
        <div className="chart-container">
          {renderThroughputChart()}
        </div>
        
        <div className="chart-container">
          {renderResourceUsageChart()}
        </div>
        
        <div className="chart-container">
          {renderMetricTypeDistribution()}
        </div>
      </div>
      
      <div className="metrics-table">
        <h4>Latest Metrics</h4>
        <table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Value</th>
              <th>Unit</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {metrics.slice(0, 20).map((metric) => (
              <tr key={metric.id}>
                <td>{metric.metricType}</td>
                <td>{metric.value.toFixed(2)}</td>
                <td>{metric.unit}</td>
                <td>{new Date(metric.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MetricsPanel;
