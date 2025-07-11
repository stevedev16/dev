import express, { Request, Response } from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import { storage } from './storage';

const app = express();
const httpServer = createServer(app);

// WebSocket server for real-time updates
const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

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
          (ws as any).subscriptions = data.channels || [];
          break;
        case 'unsubscribe':
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

app.get('/api/events/by-type/:type', async (req: Request, res: Response) => {
  try {
    const { type } = req.params;
    const events = await storage.getEventsByType(type);
    res.json(events);
  } catch (error) {
    console.error('Error fetching events by type:', error);
    res.status(500).json({ error: 'Failed to fetch events by type' });
  }
});

// Alerts
app.get('/api/alerts', async (req: Request, res: Response) => {
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

app.get('/api/alerts/by-severity/:severity', async (req: Request, res: Response) => {
  try {
    const { severity } = req.params;
    const alerts = await storage.getAlertsBySeverity(severity);
    res.json(alerts);
  } catch (error) {
    console.error('Error fetching alerts by severity:', error);
    res.status(500).json({ error: 'Failed to fetch alerts by severity' });
  }
});

// Transactions
app.get('/api/transactions', async (req: Request, res: Response) => {
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

app.get('/api/transactions/high-risk', async (req: Request, res: Response) => {
  try {
    const threshold = parseFloat(req.query.threshold as string) || 0.7;
    const transactions = await storage.getHighRiskTransactions(threshold);
    res.json(transactions);
  } catch (error) {
    console.error('Error fetching high-risk transactions:', error);
    res.status(500).json({ error: 'Failed to fetch high-risk transactions' });
  }
});

// Graph
app.get('/api/graph/entities', async (req: Request, res: Response) => {
  try {
    const entityType = req.query.type as string;
    const entities = await storage.getGraphEntities(entityType);
    res.json(entities);
  } catch (error) {
    console.error('Error fetching graph entities:', error);
    res.status(500).json({ error: 'Failed to fetch graph entities' });
  }
});

app.get('/api/graph/relationships', async (req: Request, res: Response) => {
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

// Metrics
app.get('/api/metrics', async (req: Request, res: Response) => {
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

app.get('/api/metrics/latest', async (req: Request, res: Response) => {
  try {
    const metrics = await storage.getLatestMetrics();
    res.json(metrics);
  } catch (error) {
    console.error('Error fetching latest metrics:', error);
    res.status(500).json({ error: 'Failed to fetch latest metrics' });
  }
});

// Dashboard summary
app.get('/api/dashboard/summary', async (req: Request, res: Response) => {
  try {
    const [events, alerts, metrics] = await Promise.all([
      storage.getEvents(10),
      storage.getAlerts(10),
      storage.getLatestMetrics()
    ]);

    const summary = {
      totalEvents: events.length,
      totalAlerts: alerts.length,
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
app.get('/api/health', (req: Request, res: Response) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Fallback for SPA
app.get('*', (req: Request, res: Response) => {
  res.sendFile('index.html', { root: 'client/dist' });
});

export { app, httpServer };