import { app, httpServer } from './routes-fixed';
import { storage } from './storage';

const PORT = 5000;

async function startServer() {
  try {
    console.log('Starting FinSecure Nexus server...');
    
    // Initialize database and seed with some data
    await initializeDatabase();
    
    // Start HTTP server
    httpServer.listen(PORT, '0.0.0.0', () => {
      console.log(`🚀 Server running on http://0.0.0.0:${PORT}`);
      console.log(`📊 Dashboard available at http://localhost:${PORT}`);
      console.log(`🔌 WebSocket endpoint: ws://localhost:${PORT}/ws`);
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
    
    // Create default analyst user
    const analystUser = await storage.getUserByUsername('analyst');
    if (!analystUser) {
      await storage.createUser({
        username: 'analyst',
        email: 'analyst@finsecure.com',
        role: 'analyst'
      });
      console.log('✅ Analyst user created');
    }
    
    console.log('✅ Database initialized successfully');
  } catch (error) {
    console.error('Database initialization failed:', error);
    throw error;
  }
}

startServer();