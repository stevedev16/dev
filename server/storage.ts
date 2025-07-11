import { 
  users, events, alerts, mlModels, financialTransactions, 
  graphEntities, graphRelationships, systemMetrics,
  type User, type InsertUser, type Event, type InsertEvent,
  type Alert, type InsertAlert, type MLModel, type InsertMLModel,
  type FinancialTransaction, type InsertFinancialTransaction,
  type GraphEntity, type InsertGraphEntity, type GraphRelationship,
  type InsertGraphRelationship, type SystemMetric, type InsertSystemMetric
} from "../shared/schema";

// Re-export types for use in other modules
export type {
  User, InsertUser, Event, InsertEvent,
  Alert, InsertAlert, MLModel, InsertMLModel,
  FinancialTransaction, InsertFinancialTransaction,
  GraphEntity, InsertGraphEntity, GraphRelationship,
  InsertGraphRelationship, SystemMetric, InsertSystemMetric
};
import { db } from "./db";
import { eq, desc, and, gte, lte, sql } from "drizzle-orm";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(insertUser: InsertUser): Promise<User>;
  
  // Event operations
  createEvent(insertEvent: InsertEvent): Promise<Event>;
  getEvents(limit?: number, offset?: number): Promise<Event[]>;
  getEventsByTimeRange(start: Date, end: Date): Promise<Event[]>;
  getEventsByType(sourceType: string): Promise<Event[]>;
  
  // Alert operations
  createAlert(insertAlert: InsertAlert): Promise<Alert>;
  getAlerts(limit?: number, offset?: number): Promise<Alert[]>;
  getAlertsByStatus(status: string): Promise<Alert[]>;
  getAlertsBySeverity(severity: string): Promise<Alert[]>;
  updateAlert(id: number, updates: Partial<Alert>): Promise<Alert>;
  
  // ML Model operations
  createMLModel(insertModel: InsertMLModel): Promise<MLModel>;
  getActiveMLModels(): Promise<MLModel[]>;
  getMLModelByType(type: string): Promise<MLModel | undefined>;
  updateMLModel(id: number, updates: Partial<MLModel>): Promise<MLModel>;
  
  // Financial Transaction operations
  createFinancialTransaction(insertTransaction: InsertFinancialTransaction): Promise<FinancialTransaction>;
  getFinancialTransactions(limit?: number, offset?: number): Promise<FinancialTransaction[]>;
  getHighRiskTransactions(threshold: number): Promise<FinancialTransaction[]>;
  
  // Graph operations
  createGraphEntity(insertEntity: InsertGraphEntity): Promise<GraphEntity>;
  createGraphRelationship(insertRelationship: InsertGraphRelationship): Promise<GraphRelationship>;
  getGraphEntities(entityType?: string): Promise<GraphEntity[]>;
  getGraphRelationships(fromEntityId?: string, toEntityId?: string): Promise<GraphRelationship[]>;
  
  // System Metrics operations
  createSystemMetric(insertMetric: InsertSystemMetric): Promise<SystemMetric>;
  getSystemMetrics(metricType?: string, limit?: number): Promise<SystemMetric[]>;
  getLatestMetrics(): Promise<SystemMetric[]>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }

  // Event operations
  async createEvent(insertEvent: InsertEvent): Promise<Event> {
    const [event] = await db.insert(events).values(insertEvent).returning();
    return event;
  }

  async getEvents(limit: number = 100, offset: number = 0): Promise<Event[]> {
    return await db.select().from(events)
      .orderBy(desc(events.timestamp))
      .limit(limit)
      .offset(offset);
  }

  async getEventsByTimeRange(start: Date, end: Date): Promise<Event[]> {
    return await db.select().from(events)
      .where(and(
        gte(events.timestamp, start),
        lte(events.timestamp, end)
      ))
      .orderBy(desc(events.timestamp));
  }

  async getEventsByType(sourceType: string): Promise<Event[]> {
    return await db.select().from(events)
      .where(eq(events.sourceType, sourceType))
      .orderBy(desc(events.timestamp));
  }

  // Alert operations
  async createAlert(insertAlert: InsertAlert): Promise<Alert> {
    const [alert] = await db.insert(alerts).values(insertAlert).returning();
    return alert;
  }

  async getAlerts(limit: number = 100, offset: number = 0): Promise<Alert[]> {
    return await db.select().from(alerts)
      .orderBy(desc(alerts.createdAt))
      .limit(limit)
      .offset(offset);
  }

  async getAlertsByStatus(status: string): Promise<Alert[]> {
    return await db.select().from(alerts)
      .where(eq(alerts.status, status))
      .orderBy(desc(alerts.createdAt));
  }

  async getAlertsBySeverity(severity: string): Promise<Alert[]> {
    return await db.select().from(alerts)
      .where(eq(alerts.severity, severity))
      .orderBy(desc(alerts.createdAt));
  }

  async updateAlert(id: number, updates: Partial<Alert>): Promise<Alert> {
    const [alert] = await db.update(alerts)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(alerts.id, id))
      .returning();
    return alert;
  }

  // ML Model operations
  async createMLModel(insertModel: InsertMLModel): Promise<MLModel> {
    const [model] = await db.insert(mlModels).values(insertModel).returning();
    return model;
  }

  async getActiveMLModels(): Promise<MLModel[]> {
    return await db.select().from(mlModels)
      .where(eq(mlModels.isActive, true))
      .orderBy(desc(mlModels.createdAt));
  }

  async getMLModelByType(type: string): Promise<MLModel | undefined> {
    const [model] = await db.select().from(mlModels)
      .where(and(eq(mlModels.type, type), eq(mlModels.isActive, true)))
      .orderBy(desc(mlModels.createdAt));
    return model || undefined;
  }

  async updateMLModel(id: number, updates: Partial<MLModel>): Promise<MLModel> {
    const [model] = await db.update(mlModels)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(mlModels.id, id))
      .returning();
    return model;
  }

  // Financial Transaction operations
  async createFinancialTransaction(insertTransaction: InsertFinancialTransaction): Promise<FinancialTransaction> {
    const [transaction] = await db.insert(financialTransactions).values(insertTransaction).returning();
    return transaction;
  }

  async getFinancialTransactions(limit: number = 100, offset: number = 0): Promise<FinancialTransaction[]> {
    return await db.select().from(financialTransactions)
      .orderBy(desc(financialTransactions.timestamp))
      .limit(limit)
      .offset(offset);
  }

  async getHighRiskTransactions(threshold: number): Promise<FinancialTransaction[]> {
    return await db.select().from(financialTransactions)
      .where(gte(financialTransactions.riskScore, threshold))
      .orderBy(desc(financialTransactions.riskScore));
  }

  // Graph operations
  async createGraphEntity(insertEntity: InsertGraphEntity): Promise<GraphEntity> {
    const [entity] = await db.insert(graphEntities).values(insertEntity).returning();
    return entity;
  }

  async createGraphRelationship(insertRelationship: InsertGraphRelationship): Promise<GraphRelationship> {
    const [relationship] = await db.insert(graphRelationships).values(insertRelationship).returning();
    return relationship;
  }

  async getGraphEntities(entityType?: string): Promise<GraphEntity[]> {
    const query = db.select().from(graphEntities);
    if (entityType) {
      return await query.where(eq(graphEntities.entityType, entityType));
    }
    return await query.orderBy(desc(graphEntities.createdAt));
  }

  async getGraphRelationships(fromEntityId?: string, toEntityId?: string): Promise<GraphRelationship[]> {
    const query = db.select().from(graphRelationships);
    if (fromEntityId && toEntityId) {
      return await query.where(and(
        eq(graphRelationships.fromEntityId, fromEntityId),
        eq(graphRelationships.toEntityId, toEntityId)
      ));
    } else if (fromEntityId) {
      return await query.where(eq(graphRelationships.fromEntityId, fromEntityId));
    } else if (toEntityId) {
      return await query.where(eq(graphRelationships.toEntityId, toEntityId));
    }
    return await query.orderBy(desc(graphRelationships.createdAt));
  }

  // System Metrics operations
  async createSystemMetric(insertMetric: InsertSystemMetric): Promise<SystemMetric> {
    const [metric] = await db.insert(systemMetrics).values(insertMetric).returning();
    return metric;
  }

  async getSystemMetrics(metricType?: string, limit: number = 100): Promise<SystemMetric[]> {
    const query = db.select().from(systemMetrics);
    if (metricType) {
      return await query.where(eq(systemMetrics.metricType, metricType))
        .orderBy(desc(systemMetrics.timestamp))
        .limit(limit);
    }
    return await query.orderBy(desc(systemMetrics.timestamp)).limit(limit);
  }

  async getLatestMetrics(): Promise<SystemMetric[]> {
    return await db.select().from(systemMetrics)
      .orderBy(desc(systemMetrics.timestamp))
      .limit(50);
  }
}

export const storage = new DatabaseStorage();
