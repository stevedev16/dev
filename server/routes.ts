import express, { Request, Response } from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import { storage } from './storage';
import { IngestionService } from './services/ingestion';
import { MLInferenceService } from './services/ml-inference';
import { AlertCorrelationService } from './services/alert-correlation';
import { FeatureEngineeringService } from './services/feature-engineering';
import { GraphAnalysisService } from './services/graph-analysis';

const app = express();
const httpServer = createServer(app);

// WebSocket server for real-time updates
const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

// Services
const ingestionService = new IngestionService(storage);
const mlInferenceService = new MLInferenceService(storage);
const alertCorrelationService = new AlertCorrelationService(storage);
const featureEngineeringService = new FeatureEngineeringService(storage);
const graphAnalysisService = new GraphAnalysisService(storage);

// Middleware
app.use(express.json());
app.use(express.static('client/dist'));

// WebSocket connection handling
wss.on('connection', (ws: WebSocket) => {
  console.log('New WebSocket connection established');
  
  ws.on('message', async (message: Buffer) => {
    try {
      const data = JSON.parse(message.toString());
      
      switch (data.type) {
        case 'subscribe':
          // Handle subscription to real-time updates
          (ws as any).subscriptions = data.channels || [];
          break;
        case 'unsubscribe':
          // Handle unsubscription
          (ws as any).subscriptions = [];
          break;
      }
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  });

  ws.on('close', () => {
    console.log('WebSocket connection closed');
  });
});

// Broadcast function for real-time updates
export const broadcast = (channel: string, data: any) => {
  wss.clients.forEach((client: WebSocket) => {
    if (client.readyState === WebSocket.OPEN) {
      const subscriptions = (client as any).subscriptions || [];
      if (subscriptions.includes(channel)) {
        client.send(JSON.stringify({ channel, data }));
      }
    }
  });
};

// API Routes

// Events
app.get('/api/events', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    const events = await storage.getEvents(limit, offset);
    res.json(events);
  } catch (error) {
    console.error('Error fetching events:', error);
    res.status(500).json({ error: 'Failed to fetch events' });
  }
});

app.get('/api/events/type/:type', async (req, res) => {
  try {
    const events = await storage.getEventsByType(req.params.type);
    res.json(events);
  } catch (error) {
    console.error('Error fetching events by type:', error);
    res.status(500).json({ error: 'Failed to fetch events by type' });
  }
});

app.post('/api/events', async (req, res) => {
  try {
    const event = await ingestionService.processEvent(req.body);
    broadcast('events', event);
    res.json(event);
  } catch (error) {
    console.error('Error creating event:', error);
    res.status(500).json({ error: 'Failed to create event' });
  }
});

// Alerts
app.get('/api/alerts', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    const alerts = await storage.getAlerts(limit, offset);
    res.json(alerts);
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

app.get('/api/alerts/status/:status', async (req, res) => {
  try {
    const alerts = await storage.getAlertsByStatus(req.params.status);
    res.json(alerts);
  } catch (error) {
    console.error('Error fetching alerts by status:', error);
    res.status(500).json({ error: 'Failed to fetch alerts by status' });
  }
});

app.put('/api/alerts/:id', async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    const alert = await storage.updateAlert(id, req.body);
    broadcast('alerts', alert);
    res.json(alert);
  } catch (error) {
    console.error('Error updating alert:', error);
    res.status(500).json({ error: 'Failed to update alert' });
  }
});

// ML Models
app.get('/api/ml-models', async (req, res) => {
  try {
    const models = await storage.getActiveMLModels();
    res.json(models);
  } catch (error) {
    console.error('Error fetching ML models:', error);
    res.status(500).json({ error: 'Failed to fetch ML models' });
  }
});

app.post('/api/ml-models/predict', async (req, res) => {
  try {
    const prediction = await mlInferenceService.predict(req.body);
    res.json(prediction);
  } catch (error) {
    console.error('Error making prediction:', error);
    res.status(500).json({ error: 'Failed to make prediction' });
  }
});

// Financial Transactions
app.get('/api/transactions', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const offset = parseInt(req.query.offset as string) || 0;
    const transactions = await storage.getFinancialTransactions(limit, offset);
    res.json(transactions);
  } catch (error) {
    console.error('Error fetching transactions:', error);
    res.status(500).json({ error: 'Failed to fetch transactions' });
  }
});

app.get('/api/transactions/high-risk', async (req, res) => {
  try {
    const threshold = parseFloat(req.query.threshold as string) || 0.7;
    const transactions = await storage.getHighRiskTransactions(threshold);
    res.json(transactions);
  } catch (error) {
    console.error('Error fetching high-risk transactions:', error);
    res.status(500).json({ error: 'Failed to fetch high-risk transactions' });
  }
});

// Graph Analysis
app.get('/api/graph/entities', async (req, res) => {
  try {
    const entityType = req.query.type as string;
    const entities = await storage.getGraphEntities(entityType);
    res.json(entities);
  } catch (error) {
    console.error('Error fetching graph entities:', error);
    res.status(500).json({ error: 'Failed to fetch graph entities' });
  }
});

app.get('/api/graph/relationships', async (req, res) => {
  try {
    const fromEntityId = req.query.from as string;
    const toEntityId = req.query.to as string;
    const relationships = await storage.getGraphRelationships(fromEntityId, toEntityId);
    res.json(relationships);
  } catch (error) {
    console.error('Error fetching graph relationships:', error);
    res.status(500).json({ error: 'Failed to fetch graph relationships' });
  }
});

app.post('/api/graph/analyze', async (req, res) => {
  try {
    const analysis = await graphAnalysisService.analyzeGraph(req.body);
    res.json(analysis);
  } catch (error) {
    console.error('Error analyzing graph:', error);
    res.status(500).json({ error: 'Failed to analyze graph' });
  }
});

// System Metrics
app.get('/api/metrics', async (req, res) => {
  try {
    const metricType = req.query.type as string;
    const limit = parseInt(req.query.limit as string) || 100;
    const metrics = await storage.getSystemMetrics(metricType, limit);
    res.json(metrics);
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ error: 'Failed to fetch metrics' });
  }
});

app.get('/api/metrics/latest', async (req, res) => {
  try {
    const metrics = await storage.getLatestMetrics();
    res.json(metrics);
  } catch (error) {
    console.error('Error fetching latest metrics:', error);
    res.status(500).json({ error: 'Failed to fetch latest metrics' });
  }
});

// Dashboard summary
app.get('/api/dashboard/summary', async (req, res) => {
  try {
    const [events, alerts, highRiskTransactions, metrics] = await Promise.all([
      storage.getEvents(10),
      storage.getAlerts(10),
      storage.getHighRiskTransactions(0.7),
      storage.getLatestMetrics()
    ]);

    const summary = {
      totalEvents: events.length,
      totalAlerts: alerts.length,
      highRiskTransactions: highRiskTransactions.length,
      systemHealth: metrics.filter(m => m.metricType === 'system_health').slice(0, 1)[0]?.value || 100,
      recentEvents: events.slice(0, 5),
      criticalAlerts: alerts.filter(a => a.severity === 'critical').slice(0, 5),
      performanceMetrics: metrics.filter(m => ['latency', 'throughput'].includes(m.metricType)).slice(0, 10)
    };

    res.json(summary);
  } catch (error) {
    console.error('Error fetching dashboard summary:', error);
    res.status(500).json({ error: 'Failed to fetch dashboard summary' });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Fallback for SPA
app.get('*', (req, res) => {
  res.sendFile('index.html', { root: 'client/dist' });
});

export { app, httpServer };
