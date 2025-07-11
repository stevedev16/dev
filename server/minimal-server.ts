import express from 'express';
import { createServer } from 'http';
import { storage } from './storage';
import path from 'path';

const app = express();
const httpServer = createServer(app);
const PORT = process.env.PORT || 5000;

// Middleware
app.use(express.json());

// Debug static file serving
const staticPath = path.join(__dirname, '../client/dist');
console.log('Static files path:', staticPath);
console.log('Static files exist:', require('fs').existsSync(staticPath));

app.use(express.static(staticPath));

// Test the server first with minimal routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.get('/api/events', async (req, res) => {
  try {
    const events = await storage.getEvents(10);
    res.json(events);
  } catch (error) {
    console.error('Error fetching events:', error);
    res.status(500).json({ error: 'Failed to fetch events' });
  }
});

app.get('/api/alerts', async (req, res) => {
  try {
    const alerts = await storage.getAlerts(10);
    res.json(alerts);
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

app.post('/api/events', async (req, res) => {
  try {
    const eventData = {
      ...req.body,
      timestamp: new Date(),
      userId: req.body.userId || 'system',
      ipAddress: req.body.ipAddress || '127.0.0.1'
    };
    const event = await storage.createEvent(eventData);
    res.status(201).json(event);
  } catch (error) {
    console.error('Error creating event:', error);
    res.status(500).json({ error: 'Failed to create event' });
  }
});

app.post('/api/alerts', async (req, res) => {
  try {
    const alertData = {
      ...req.body,
      confidence: req.body.confidence || 0.8
    };
    const alert = await storage.createAlert(alertData);
    res.status(201).json(alert);
  } catch (error) {
    console.error('Error creating alert:', error);
    res.status(500).json({ error: 'Failed to create alert' });
  }
});

// Serve React app for root route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/dist/index.html'));
});

async function startServer() {
  try {
    console.log('Starting minimal FinSecure Nexus server...');
    
    // Initialize database
    await initializeDatabase();
    
    // Start HTTP server
    httpServer.listen(PORT, '0.0.0.0', () => {
      console.log(`🚀 Server running on http://0.0.0.0:${PORT}`);
      console.log(`📊 API available at http://localhost:${PORT}/api`);
    });
    
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

async function initializeDatabase() {
  console.log('Initializing database...');
  
  try {
    // Create default admin user
    const adminUser = await storage.getUserByUsername('admin');
    if (!adminUser) {
      await storage.createUser({
        username: 'admin',
        email: 'admin@finsecure.com',
        role: 'admin'
      });
      console.log('✅ Admin user created');
    }
    
    console.log('✅ Database initialized successfully');
  } catch (error) {
    console.error('Database initialization failed:', error);
    throw error;
  }
}

startServer();