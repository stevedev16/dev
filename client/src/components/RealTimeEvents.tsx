import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { Event } from '../types';

const RealTimeEvents: React.FC = () => {
  const [events, setEvents] = useState<Event[]>([]);
  const [maxEvents, setMaxEvents] = useState(50);
  const { isConnected, messages } = useWebSocket();

  useEffect(() => {
    // Process incoming WebSocket messages
    messages.forEach(message => {
      if (message.channel === 'events') {
        const newEvent = message.data as Event;
        setEvents(prev => {
          const updated = [newEvent, ...prev];
          return updated.slice(0, maxEvents);
        });
      }
    });
  }, [messages, maxEvents]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#DC2626';
      case 'high': return '#EA580C';
      case 'medium': return '#D97706';
      case 'low': return '#65A30D';
      default: return '#6B7280';
    }
  };

  const getEventIcon = (sourceType: string) => {
    switch (sourceType) {
      case 'cloud': return '☁️';
      case 'network': return '🌐';
      case 'financial': return '💰';
      case 'iam': return '🔐';
      case 'spatial': return '📍';
      default: return '📊';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const renderConnectionStatus = () => (
    <div className="connection-status">
      <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
        <div className="status-dot" />
        <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div className="event-count">
        {events.length} events
      </div>
    </div>
  );

  const renderEventFilters = () => (
    <div className="event-filters">
      <select 
        value={maxEvents} 
        onChange={(e) => setMaxEvents(parseInt(e.target.value))}
        className="filter-select"
      >
        <option value={25}>25 events</option>
        <option value={50}>50 events</option>
        <option value={100}>100 events</option>
      </select>
      
      <button 
        onClick={() => setEvents([])}
        className="clear-button"
      >
        Clear Events
      </button>
    </div>
  );

  return (
    <div className="real-time-events">
      <div className="events-header">
        <h4>Real-Time Events</h4>
        {renderConnectionStatus()}
      </div>
      
      {renderEventFilters()}
      
      <div className="events-stream">
        {events.length === 0 ? (
          <div className="no-events">
            <div className="no-events-icon">📡</div>
            <p>Waiting for real-time events...</p>
            <small>Events will appear here as they are processed</small>
          </div>
        ) : (
          <div className="events-list">
            {events.map((event, index) => (
              <div key={`${event.id}-${index}`} className="event-item">
                <div className="event-icon">
                  {getEventIcon(event.sourceType)}
                </div>
                
                <div className="event-content">
                  <div className="event-header">
                    <div className="event-type">
                      {event.eventType}
                    </div>
                    <div className="event-source">
                      {event.sourceType}
                    </div>
                    <div 
                      className="event-severity"
                      style={{ color: getSeverityColor(event.severity) }}
                    >
                      {event.severity}
                    </div>
                  </div>
                  
                  <div className="event-details">
                    {event.userId && (
                      <span className="event-detail">
                        👤 {event.userId}
                      </span>
                    )}
                    {event.ipAddress && (
                      <span className="event-detail">
                        🌐 {event.ipAddress}
                      </span>
                    )}
                    {event.location && (
                      <span className="event-detail">
                        📍 {event.location.city}, {event.location.country}
                      </span>
                    )}
                  </div>
                  
                  <div className="event-timestamp">
                    {formatTimestamp(event.timestamp)}
                  </div>
                </div>
                
                <div className="event-actions">
                  <button 
                    className="event-action-button"
                    onClick={() => console.log('View event details:', event)}
                  >
                    👁️
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {events.length > 0 && (
        <div className="events-summary">
          <div className="summary-stats">
            <div className="stat">
              <span className="stat-label">Critical:</span>
              <span className="stat-value critical">
                {events.filter(e => e.severity === 'critical').length}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">High:</span>
              <span className="stat-value high">
                {events.filter(e => e.severity === 'high').length}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Medium:</span>
              <span className="stat-value medium">
                {events.filter(e => e.severity === 'medium').length}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Low:</span>
              <span className="stat-value low">
                {events.filter(e => e.severity === 'low').length}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RealTimeEvents;
