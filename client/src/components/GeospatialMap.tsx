import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { getEvents, getFinancialTransactions } from '../services/api';
import { Event, FinancialTransaction } from '../types';
import 'leaflet/dist/leaflet.css';

interface LocationEvent {
  id: string;
  type: 'event' | 'transaction';
  latitude: number;
  longitude: number;
  severity: string;
  description: string;
  timestamp: Date;
  riskScore?: number;
}

const GeospatialMap: React.FC = () => {
  const [locationEvents, setLocationEvents] = useState<LocationEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEventType, setSelectedEventType] = useState<'all' | 'events' | 'transactions'>('all');
  const [selectedSeverity, setSelectedSeverity] = useState<'all' | 'low' | 'medium' | 'high' | 'critical'>('all');
  const [heatmapEnabled, setHeatmapEnabled] = useState(false);

  useEffect(() => {
    loadGeospatialData();
  }, []);

  const loadGeospatialData = async () => {
    try {
      setLoading(true);
      const [events, transactions] = await Promise.all([
        getEvents(),
        getFinancialTransactions()
      ]);

      const locationEvents: LocationEvent[] = [];

      // Process events with location data
      events.forEach(event => {
        if (event.location && event.location.latitude && event.location.longitude) {
          locationEvents.push({
            id: `event-${event.id}`,
            type: 'event',
            latitude: event.location.latitude,
            longitude: event.location.longitude,
            severity: event.severity,
            description: `${event.eventType} - ${event.sourceType}`,
            timestamp: new Date(event.timestamp)
          });
        }
      });

      // Process transactions with location data
      transactions.forEach(transaction => {
        if (transaction.location && transaction.location.latitude && transaction.location.longitude) {
          const severity = transaction.riskScore && transaction.riskScore > 0.7 ? 'high' : 
                          transaction.riskScore && transaction.riskScore > 0.5 ? 'medium' : 'low';
          
          locationEvents.push({
            id: `transaction-${transaction.id}`,
            type: 'transaction',
            latitude: transaction.location.latitude,
            longitude: transaction.location.longitude,
            severity,
            description: `${transaction.transactionType} - ${transaction.amount} ${transaction.currency}`,
            timestamp: new Date(transaction.timestamp),
            riskScore: transaction.riskScore
          });
        }
      });

      setLocationEvents(locationEvents);
      setError(null);
    } catch (err) {
      console.error('Error loading geospatial data:', err);
      setError('Failed to load geospatial data');
    } finally {
      setLoading(false);
    }
  };

  const getMarkerColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#DC2626';
      case 'high': return '#EA580C';
      case 'medium': return '#D97706';
      case 'low': return '#65A30D';
      default: return '#6B7280';
    }
  };

  const getMarkerSize = (event: LocationEvent) => {
    if (event.type === 'transaction' && event.riskScore) {
      return event.riskScore * 20 + 5;
    }
    switch (event.severity) {
      case 'critical': return 15;
      case 'high': return 12;
      case 'medium': return 9;
      case 'low': return 6;
      default: return 5;
    }
  };

  const filteredEvents = locationEvents.filter(event => {
    const typeMatch = selectedEventType === 'all' || 
                     (selectedEventType === 'events' && event.type === 'event') ||
                     (selectedEventType === 'transactions' && event.type === 'transaction');
    
    const severityMatch = selectedSeverity === 'all' || event.severity === selectedSeverity;
    
    return typeMatch && severityMatch;
  });

  const renderHeatmapLayer = () => {
    if (!heatmapEnabled) return null;

    // This would typically use a heatmap library like react-leaflet-heatmap-layer
    // For now, we'll show a placeholder
    return (
      <div className="heatmap-placeholder">
        <p>Heatmap visualization would be rendered here</p>
      </div>
    );
  };

  const renderLocationClusters = () => {
    const locationClusters = new Map<string, LocationEvent[]>();
    
    filteredEvents.forEach(event => {
      const key = `${event.latitude.toFixed(2)},${event.longitude.toFixed(2)}`;
      if (!locationClusters.has(key)) {
        locationClusters.set(key, []);
      }
      locationClusters.get(key)!.push(event);
    });

    return Array.from(locationClusters.entries()).map(([key, events]) => {
      const [lat, lng] = key.split(',').map(Number);
      const highestSeverity = events.reduce((max, event) => {
        const severityLevel = { critical: 4, high: 3, medium: 2, low: 1 }[event.severity] || 0;
        const maxLevel = { critical: 4, high: 3, medium: 2, low: 1 }[max] || 0;
        return severityLevel > maxLevel ? event.severity : max;
      }, 'low');

      return (
        <CircleMarker
          key={key}
          center={[lat, lng]}
          radius={Math.min(events.length * 3 + 5, 25)}
          color={getMarkerColor(highestSeverity)}
          fillColor={getMarkerColor(highestSeverity)}
          fillOpacity={0.6}
          weight={2}
        >
          <Popup>
            <div className="map-popup">
              <h4>Location Cluster</h4>
              <p><strong>Events:</strong> {events.length}</p>
              <p><strong>Highest Severity:</strong> {highestSeverity}</p>
              <div className="popup-events">
                {events.slice(0, 5).map(event => (
                  <div key={event.id} className="popup-event">
                    <div className="event-type">{event.type}</div>
                    <div className="event-description">{event.description}</div>
                    <div className="event-time">
                      {event.timestamp.toLocaleString()}
                    </div>
                  </div>
                ))}
                {events.length > 5 && (
                  <div className="popup-more">
                    +{events.length - 5} more events
                  </div>
                )}
              </div>
            </div>
          </Popup>
        </CircleMarker>
      );
    });
  };

  if (loading) {
    return (
      <div className="map-loading">
        <div className="loading-spinner" />
        <p>Loading geospatial data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="map-error">
        <div className="error-icon">❌</div>
        <h3>Error Loading Map</h3>
        <p>{error}</p>
        <button onClick={loadGeospatialData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="geospatial-map">
      <div className="map-header">
        <h3>Geospatial Analysis</h3>
        <div className="map-controls">
          <select 
            value={selectedEventType} 
            onChange={(e) => setSelectedEventType(e.target.value as any)}
            className="filter-select"
          >
            <option value="all">All Types</option>
            <option value="events">Events Only</option>
            <option value="transactions">Transactions Only</option>
          </select>
          
          <select 
            value={selectedSeverity} 
            onChange={(e) => setSelectedSeverity(e.target.value as any)}
            className="filter-select"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          
          <button 
            className={`toggle-button ${heatmapEnabled ? 'active' : ''}`}
            onClick={() => setHeatmapEnabled(!heatmapEnabled)}
          >
            Heatmap
          </button>
        </div>
      </div>
      
      <div className="map-stats">
        <div className="stat">
          <span className="stat-label">Total Events</span>
          <span className="stat-value">{filteredEvents.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">High Risk</span>
          <span className="stat-value critical">
            {filteredEvents.filter(e => e.severity === 'high' || e.severity === 'critical').length}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Transactions</span>
          <span className="stat-value">
            {filteredEvents.filter(e => e.type === 'transaction').length}
          </span>
        </div>
      </div>
      
      <div className="map-container">
        <MapContainer
          center={[39.8283, -98.5795]} // Center of US
          zoom={4}
          style={{ height: '600px', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {renderLocationClusters()}
          {renderHeatmapLayer()}
        </MapContainer>
      </div>
      
      <div className="map-legend">
        <h4>Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-marker" style={{ backgroundColor: '#DC2626' }} />
            <span>Critical</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker" style={{ backgroundColor: '#EA580C' }} />
            <span>High</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker" style={{ backgroundColor: '#D97706' }} />
            <span>Medium</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker" style={{ backgroundColor: '#65A30D' }} />
            <span>Low</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GeospatialMap;
