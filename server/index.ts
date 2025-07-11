import { app, httpServer } from './routes';
import { storage } from './storage';
import { SyntheticDataGenerator } from './utils/synthetic-data';

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 5000;

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
    
    // Start background processes
    startBackgroundProcesses();
    
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
    
    // Initialize ML models
    await initializeMLModels();
    
    // Generate initial synthetic data for demonstration
    await generateInitialData();
    
    console.log('✅ Database initialized successfully');
  } catch (error) {
    console.error('Database initialization failed:', error);
    throw error;
  }
}

async function initializeMLModels() {
  console.log('Initializing ML models...');
  
  const models = [
    {
      name: 'CNN Network Anomaly Detector',
      type: 'cnn',
      version: '1.0.0',
      accuracy: 0.92,
      precision: 0.89,
      recall: 0.94,
      f1Score: 0.91,
      isActive: true,
      modelPath: '/models/cnn_network_v1.pkl'
    },
    {
      name: 'LSTM Financial Fraud Detector',
      type: 'lstm',
      version: '1.0.0',
      accuracy: 0.95,
      precision: 0.93,
      recall: 0.97,
      f1Score: 0.95,
      isActive: true,
      modelPath: '/models/lstm_financial_v1.pkl'
    },
    {
      name: 'Transformer Behavioral Analyzer',
      type: 'transformer',
      version: '1.0.0',
      accuracy: 0.89,
      precision: 0.87,
      recall: 0.91,
      f1Score: 0.89,
      isActive: true,
      modelPath: '/models/transformer_behavioral_v1.pkl'
    }
  ];
  
  for (const model of models) {
    try {
      await storage.createMLModel(model);
      console.log(`✅ ML model initialized: ${model.name}`);
    } catch (error) {
      console.log(`ℹ️  ML model already exists: ${model.name}`);
    }
  }
}

async function generateInitialData() {
  console.log('Generating initial synthetic data...');
  
  const generator = new SyntheticDataGenerator();
  
  try {
    // Generate graph entities
    const entities = generator.generateGraphEntities(100);
    for (const entity of entities) {
      try {
        await storage.createGraphEntity(entity);
      } catch (error) {
        // Entity might already exist
      }
    }
    
    // Generate graph relationships
    const relationships = generator.generateGraphRelationships(entities);
    for (const relationship of relationships) {
      try {
        await storage.createGraphRelationship(relationship);
      } catch (error) {
        // Relationship might already exist
      }
    }
    
    // Generate financial transactions
    const transactions = generator.generateFinancialTransactions(50);
    for (const transaction of transactions) {
      try {
        await storage.createFinancialTransaction(transaction);
      } catch (error) {
        // Transaction might already exist
      }
    }
    
    // Generate some initial events
    const events = generator.generateEvents(20);
    for (const event of events) {
      try {
        await storage.createEvent(event);
      } catch (error) {
        // Event might already exist
      }
    }
    
    console.log('✅ Initial synthetic data generated');
  } catch (error) {
    console.error('Failed to generate initial data:', error);
  }
}

function startBackgroundProcesses() {
  console.log('Starting background processes...');
  
  // Start synthetic data generation for demo
  startSyntheticDataGeneration();
  
  // Start system metrics collection
  startMetricsCollection();
  
  console.log('✅ Background processes started');
}

function startSyntheticDataGeneration() {
  const generator = new SyntheticDataGenerator();
  
  // Generate normal events every 5 seconds
  setInterval(async () => {
    try {
      const events = generator.generateEvents(1);
      for (const event of events) {
        await storage.createEvent(event);
      }
    } catch (error) {
      console.error('Error generating synthetic events:', error);
    }
  }, 5000);
  
  // Generate anomalous events every 30 seconds
  setInterval(async () => {
    try {
      const events = generator.generateAnomalousEvents(1);
      for (const event of events) {
        await storage.createEvent(event);
      }
    } catch (error) {
      console.error('Error generating anomalous events:', error);
    }
  }, 30000);
  
  // Generate financial transactions every 10 seconds
  setInterval(async () => {
    try {
      const transactions = generator.generateFinancialTransactions(1);
      for (const transaction of transactions) {
        await storage.createFinancialTransaction(transaction);
      }
    } catch (error) {
      console.error('Error generating financial transactions:', error);
    }
  }, 10000);
}

function startMetricsCollection() {
  // System health metrics
  setInterval(async () => {
    try {
      const metrics = [
        {
          metricType: 'system_health',
          value: Math.random() * 20 + 80, // 80-100% health
          unit: 'percentage',
          metadata: { component: 'overall' }
        },
        {
          metricType: 'cpu_usage',
          value: Math.random() * 30 + 20, // 20-50% CPU
          unit: 'percentage',
          metadata: { component: 'server' }
        },
        {
          metricType: 'memory_usage',
          value: Math.random() * 40 + 30, // 30-70% memory
          unit: 'percentage',
          metadata: { component: 'server' }
        },
        {
          metricType: 'throughput',
          value: Math.random() * 1000 + 500, // 500-1500 events/sec
          unit: 'events_per_second',
          metadata: { component: 'ingestion' }
        }
      ];
      
      for (const metric of metrics) {
        await storage.createSystemMetric(metric);
      }
    } catch (error) {
      console.error('Error collecting system metrics:', error);
    }
  }, 15000); // Every 15 seconds
}

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\nShutting down server...');
  process.exit(0);
});

// Start the server
startServer();
