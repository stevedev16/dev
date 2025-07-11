// Core entity types
export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Event {
  id: number;
  sourceType: string;
  eventType: string;
  timestamp: Date;
  rawData: any;
  processedFeatures?: any;
  severity: string;
  userId?: string;
  ipAddress?: string;
  location?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  createdAt: Date;
}

export interface Alert {
  id: number;
  eventId?: number;
  alertType: string;
  severity: string;
  confidence: number;
  description: string;
  status: string;
  assignedTo?: number;
  correlatedAlerts?: number[];
  responseActions?: any[];
  createdAt: Date;
  updatedAt: Date;
}

export interface MLModel {
  id: number;
  name: string;
  type: string;
  version: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  isActive: boolean;
  modelPath: string;
  trainingData?: any;
  createdAt: Date;
  updatedAt: Date;
}

export interface FinancialTransaction {
  id: number;
  transactionId: string;
  fromAccount: string;
  toAccount: string;
  amount: number;
  currency: string;
  transactionType: string;
  timestamp: Date;
  location?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  riskScore?: number;
  flags?: any[];
  createdAt: Date;
}

export interface GraphEntity {
  id: number;
  entityId: string;
  entityType: string;
  properties: any;
  riskScore?: number;
  createdAt: Date;
  updatedAt: Date;
  // For D3.js simulation
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphRelationship {
  id: number;
  fromEntityId: string;
  toEntityId: string;
  relationshipType: string;
  weight?: number;
  properties?: any;
  createdAt: Date;
  // For D3.js simulation
  source?: GraphEntity | string;
  target?: GraphEntity | string;
}

export interface SystemMetric {
  id: number;
  metricType: string;
  value: number;
  unit: string;
  timestamp: Date;
  metadata?: any;
}

// Dashboard types
export interface DashboardSummary {
  totalEvents: number;
  totalAlerts: number;
  highRiskTransactions: number;
  systemHealth: number;
  recentEvents: Event[];
  criticalAlerts: Alert[];
  performanceMetrics: SystemMetric[];
}

// API response types
export interface ApiResponse<T> {
  data: T;
  error?: string;
  timestamp: string;
}

// WebSocket message types
export interface WebSocketMessage {
  channel: string;
  data: any;
  timestamp: Date;
}

// Form types
export interface EventFilters {
  sourceType?: string;
  eventType?: string;
  severity?: string;
  startDate?: Date;
  endDate?: Date;
  limit?: number;
  offset?: number;
}

export interface AlertFilters {
  alertType?: string;
  severity?: string;
  status?: string;
  assignedTo?: number;
  startDate?: Date;
  endDate?: Date;
  limit?: number;
  offset?: number;
}

// Chart data types
export interface ChartDataPoint {
  x: string | number | Date;
  y: number;
  label?: string;
}

export interface ChartDataset {
  label: string;
  data: ChartDataPoint[];
  borderColor?: string;
  backgroundColor?: string;
  fill?: boolean;
}

// ML and Analysis types
export interface MLPrediction {
  isAnomaly: boolean;
  confidence: number;
  anomalyScore: number;
  modelType: string;
  threatType?: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  explanation: string[];
}

export interface GraphAnalysisRequest {
  entityId?: string;
  analysisType: 'community_detection' | 'centrality' | 'anomaly_detection' | 'path_analysis';
  parameters?: any;
}

export interface GraphAnalysisResult {
  analysisType: string;
  entityId?: string;
  results: any;
  insights: string[];
  riskScore: number;
  timestamp: Date;
}

// Feature engineering types
export interface ExtractedFeatures {
  eventFrequency: number;
  uniqueUsers: number;
  uniqueIPs: number;
  hourOfDay: number;
  dayOfWeek: number;
  isWeekend: boolean;
  timeFromLastEvent: number;
  location?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  distanceFromLastLocation?: number;
  locationRisk: number;
  userConnectivity?: number;
  accountCentrality?: number;
  transactionChainLength?: number;
  failedLoginAttempts?: number;
  transactionAmount?: number;
  suspiciousPatterns: string[];
  userBehaviorDeviation: number;
  accessPatternAnomaly: number;
  networkActivity?: number;
  protocolDistribution?: { [key: string]: number };
  last24hActivity: number;
  last7dActivity: number;
  riskScore: number;
  confidenceScore: number;
}

// Component prop types
export interface ComponentProps {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

// Error types
export interface ApiError {
  message: string;
  statusCode?: number;
  timestamp: Date;
}

// Utility types
export type Severity = 'low' | 'medium' | 'high' | 'critical';
export type AlertStatus = 'open' | 'acknowledged' | 'resolved' | 'false_positive';
export type EntityType = 'user' | 'account' | 'device' | 'location' | 'merchant';
export type SourceType = 'cloud' | 'network' | 'financial' | 'iam' | 'spatial';
export type ModelType = 'cnn' | 'lstm' | 'transformer' | 'ensemble';
