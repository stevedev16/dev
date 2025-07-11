import express from 'express';
import { createServer } from 'http';
import { storage } from './storage';

const app = express();
const httpServer = createServer(app);
const PORT = 5000;

app.use(express.json());

// Test route
app.get('/api/test', async (req, res) => {
  res.json({ message: 'Server is running', timestamp: new Date().toISOString() });
});

// Test database connection
app.get('/api/test/db', async (req, res) => {
  try {
    const users = await storage.getUsers ? await storage.getUsers() : [];
    res.json({ message: 'Database connected', userCount: users.length });
  } catch (error) {
    res.status(500).json({ error: 'Database connection failed', details: error.message });
  }
});

httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Test server running on http://0.0.0.0:${PORT}`);
});