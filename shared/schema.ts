import { pgTable, serial, text, timestamp, real, integer, boolean, jsonb } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  username: text('username').notNull().unique(),
  email: text('email').notNull().unique(),
  role: text('role').notNull().default('analyst'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const events = pgTable('events', {
  id: serial('id').primaryKey(),
  sourceType: text('source_type').notNull(), // 'cloud', 'network', 'financial', 'iam', 'spatial'
  eventType: text('event_type').notNull(),
  timestamp: timestamp('timestamp').notNull(),
  rawData: jsonb('raw_data').notNull(),
  processedFeatures: jsonb('processed_features'),
  severity: text('severity').notNull().default('low'),
  userId: text('user_id'),
  ipAddress: text('ip_address'),
  location: jsonb('location'), // {lat, lng, country, city}
  createdAt: timestamp('created_at').defaultNow()
});

export const alerts = pgTable('alerts', {
  id: serial('id').primaryKey(),
  eventId: integer('event_id').references(() => events.id),
  alertType: text('alert_type').notNull(), // 'anomaly', 'threat', 'fraud', 'compliance'
  severity: text('severity').notNull(), // 'low', 'medium', 'high', 'critical'
  confidence: real('confidence').notNull(),
  description: text('description').notNull(),
  status: text('status').notNull().default('open'), // 'open', 'acknowledged', 'resolved', 'false_positive'
  assignedTo: integer('assigned_to').references(() => users.id),
  correlatedAlerts: jsonb('correlated_alerts'), // array of alert IDs
  responseActions: jsonb('response_actions'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const mlModels = pgTable('ml_models', {
  id: serial('id').primaryKey(),
  name: text('name').notNull(),
  type: text('type').notNull(), // 'cnn', 'lstm', 'transformer'
  version: text('version').notNull(),
  accuracy: real('accuracy'),
  precision: real('precision'),
  recall: real('recall'),
  f1Score: real('f1_score'),
  isActive: boolean('is_active').default(false),
  modelPath: text('model_path').notNull(),
  trainingData: jsonb('training_data'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const financialTransactions = pgTable('financial_transactions', {
  id: serial('id').primaryKey(),
  transactionId: text('transaction_id').notNull().unique(),
  fromAccount: text('from_account').notNull(),
  toAccount: text('to_account').notNull(),
  amount: real('amount').notNull(),
  currency: text('currency').notNull(),
  transactionType: text('transaction_type').notNull(),
  timestamp: timestamp('timestamp').notNull(),
  location: jsonb('location'),
  riskScore: real('risk_score'),
  flags: jsonb('flags'),
  createdAt: timestamp('created_at').defaultNow()
});

export const graphEntities = pgTable('graph_entities', {
  id: serial('id').primaryKey(),
  entityId: text('entity_id').notNull().unique(),
  entityType: text('entity_type').notNull(), // 'account', 'user', 'device', 'location'
  properties: jsonb('properties').notNull(),
  riskScore: real('risk_score'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const graphRelationships = pgTable('graph_relationships', {
  id: serial('id').primaryKey(),
  fromEntityId: text('from_entity_id').notNull(),
  toEntityId: text('to_entity_id').notNull(),
  relationshipType: text('relationship_type').notNull(),
  weight: real('weight'),
  properties: jsonb('properties'),
  createdAt: timestamp('created_at').defaultNow()
});

export const systemMetrics = pgTable('system_metrics', {
  id: serial('id').primaryKey(),
  metricType: text('metric_type').notNull(), // 'throughput', 'latency', 'accuracy', 'resources'
  value: real('value').notNull(),
  unit: text('unit').notNull(),
  timestamp: timestamp('timestamp').defaultNow(),
  metadata: jsonb('metadata')
});

// Relations
export const eventsRelations = relations(events, ({ one, many }) => ({
  alerts: many(alerts)
}));

export const alertsRelations = relations(alerts, ({ one }) => ({
  event: one(events, {
    fields: [alerts.eventId],
    references: [events.id]
  }),
  assignedUser: one(users, {
    fields: [alerts.assignedTo],
    references: [users.id]
  })
}));

export const usersRelations = relations(users, ({ many }) => ({
  assignedAlerts: many(alerts)
}));

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;
export type Event = typeof events.$inferSelect;
export type InsertEvent = typeof events.$inferInsert;
export type Alert = typeof alerts.$inferSelect;
export type InsertAlert = typeof alerts.$inferInsert;
export type MLModel = typeof mlModels.$inferSelect;
export type InsertMLModel = typeof mlModels.$inferInsert;
export type FinancialTransaction = typeof financialTransactions.$inferSelect;
export type InsertFinancialTransaction = typeof financialTransactions.$inferInsert;
export type GraphEntity = typeof graphEntities.$inferSelect;
export type InsertGraphEntity = typeof graphEntities.$inferInsert;
export type GraphRelationship = typeof graphRelationships.$inferSelect;
export type InsertGraphRelationship = typeof graphRelationships.$inferInsert;
export type SystemMetric = typeof systemMetrics.$inferSelect;
export type InsertSystemMetric = typeof systemMetrics.$inferInsert;
