import { IStorage, InsertAlert, Event, Alert } from '../storage';
import { MLPrediction } from './ml-inference';

export interface CorrelationRule {
  id: string;
  name: string;
  conditions: CorrelationCondition[];
  timeWindow: number; // in minutes
  threshold: number;
  severity: string;
  description: string;
}

export interface CorrelationCondition {
  field: string;
  operator: 'equals' | 'contains' | 'greater_than' | 'less_than' | 'in_range';
  value: any;
  weight: number;
}

export class AlertCorrelationService {
  private correlationRules: CorrelationRule[] = [
    {
      id: 'multiple_failed_logins',
      name: 'Multiple Failed Login Attempts',
      conditions: [
        { field: 'eventType', operator: 'equals', value: 'login_failed', weight: 1.0 },
        { field: 'sourceType', operator: 'equals', value: 'iam', weight: 0.8 }
      ],
      timeWindow: 10,
      threshold: 5,
      severity: 'high',
      description: 'Multiple failed login attempts detected from same source'
    },
    {
      id: 'suspicious_financial_pattern',
      name: 'Suspicious Financial Activity',
      conditions: [
        { field: 'sourceType', operator: 'equals', value: 'financial', weight: 1.0 },
        { field: 'amount', operator: 'greater_than', value: 10000, weight: 0.9 }
      ],
      timeWindow: 30,
      threshold: 3,
      severity: 'critical',
      description: 'Suspicious high-value financial transactions detected'
    },
    {
      id: 'cross_domain_anomaly',
      name: 'Cross-Domain Anomaly Correlation',
      conditions: [
        { field: 'anomalyScore', operator: 'greater_than', value: 0.8, weight: 1.0 }
      ],
      timeWindow: 15,
      threshold: 2,
      severity: 'high',
      description: 'Multiple anomalies detected across different domains'
    }
  ];

  constructor(private storage: IStorage) {}

  async createAlert(event: Event, prediction: MLPrediction) {
    // Create individual alert for the anomaly
    const alert = await this.createIndividualAlert(event, prediction);
    
    // Check for correlation with existing alerts
    const correlatedAlerts = await this.findCorrelatedAlerts(event, prediction);
    
    if (correlatedAlerts.length > 0) {
      // Update alert with correlation information
      await this.updateAlertWithCorrelation(alert, correlatedAlerts);
    }
    
    return alert;
  }

  private async createIndividualAlert(event: Event, prediction: MLPrediction): Promise<Alert> {
    const alertData: InsertAlert = {
      eventId: event.id,
      alertType: this.determineAlertType(event, prediction),
      severity: prediction.riskLevel,
      confidence: prediction.confidence,
      description: this.generateAlertDescription(event, prediction),
      status: 'open',
      responseActions: this.generateResponseActions(event, prediction),
      createdAt: new Date()
    };

    return await this.storage.createAlert(alertData);
  }

  private determineAlertType(event: Event, prediction: MLPrediction): string {
    if (prediction.threatType) {
      return prediction.threatType;
    }

    // Determine alert type based on source and event type
    switch (event.sourceType) {
      case 'financial':
        return 'fraud';
      case 'network':
        return 'intrusion';
      case 'iam':
        return 'unauthorized_access';
      case 'cloud':
        return 'cloud_security';
      default:
        return 'anomaly';
    }
  }

  private generateAlertDescription(event: Event, prediction: MLPrediction): string {
    const baseDescription = `${prediction.modelType.toUpperCase()} model detected ${prediction.riskLevel} risk anomaly`;
    const explanations = prediction.explanation.length > 0 
      ? ` - ${prediction.explanation.join(', ')}` 
      : '';
    
    return baseDescription + explanations;
  }

  private generateResponseActions(event: Event, prediction: MLPrediction): any {
    const actions: any[] = [];

    // Add response actions based on severity and type
    if (prediction.riskLevel === 'critical') {
      actions.push({
        type: 'immediate_block',
        target: event.userId || event.ipAddress,
        description: 'Immediately block suspicious entity'
      });
    }

    if (prediction.riskLevel === 'high') {
      actions.push({
        type: 'enhanced_monitoring',
        target: event.userId || event.ipAddress,
        duration: '24h',
        description: 'Enable enhanced monitoring for 24 hours'
      });
    }

    if (event.sourceType === 'financial') {
      actions.push({
        type: 'transaction_review',
        target: event.rawData.transactionId,
        description: 'Flag transaction for manual review'
      });
    }

    return actions;
  }

  private async findCorrelatedAlerts(event: Event, prediction: MLPrediction): Promise<Alert[]> {
    const correlatedAlerts: Alert[] = [];
    
    for (const rule of this.correlationRules) {
      const matchingAlerts = await this.findAlertsMatchingRule(event, rule);
      
      if (matchingAlerts.length >= rule.threshold) {
        correlatedAlerts.push(...matchingAlerts);
      }
    }
    
    return correlatedAlerts;
  }

  private async findAlertsMatchingRule(event: Event, rule: CorrelationRule): Promise<Alert[]> {
    // Get recent events within the time window
    const timeWindow = new Date(Date.now() - rule.timeWindow * 60 * 1000);
    const recentEvents = await this.storage.getEventsByTimeRange(timeWindow, new Date());
    
    const matchingAlerts: Alert[] = [];
    
    for (const recentEvent of recentEvents) {
      if (this.eventMatchesRule(recentEvent, rule)) {
        // Find alerts for this event
        const alerts = await this.storage.getAlerts();
        const eventAlerts = alerts.filter(alert => alert.eventId === recentEvent.id);
        matchingAlerts.push(...eventAlerts);
      }
    }
    
    return matchingAlerts;
  }

  private eventMatchesRule(event: Event, rule: CorrelationRule): boolean {
    let totalWeight = 0;
    let matchedWeight = 0;
    
    for (const condition of rule.conditions) {
      totalWeight += condition.weight;
      
      if (this.evaluateCondition(event, condition)) {
        matchedWeight += condition.weight;
      }
    }
    
    return matchedWeight / totalWeight >= 0.7; // 70% match threshold
  }

  private evaluateCondition(event: Event, condition: CorrelationCondition): boolean {
    const fieldValue = this.getFieldValue(event, condition.field);
    
    switch (condition.operator) {
      case 'equals':
        return fieldValue === condition.value;
      case 'contains':
        return typeof fieldValue === 'string' && fieldValue.includes(condition.value);
      case 'greater_than':
        return typeof fieldValue === 'number' && fieldValue > condition.value;
      case 'less_than':
        return typeof fieldValue === 'number' && fieldValue < condition.value;
      case 'in_range':
        return typeof fieldValue === 'number' && 
               fieldValue >= condition.value[0] && 
               fieldValue <= condition.value[1];
      default:
        return false;
    }
  }

  private getFieldValue(event: Event, field: string): any {
    // Extract field value from event
    switch (field) {
      case 'eventType':
        return event.eventType;
      case 'sourceType':
        return event.sourceType;
      case 'userId':
        return event.userId;
      case 'ipAddress':
        return event.ipAddress;
      case 'severity':
        return event.severity;
      case 'amount':
        return event.rawData?.amount;
      case 'anomalyScore':
        return event.processedFeatures?.anomalyScore;
      default:
        return event.rawData?.[field] || event.processedFeatures?.[field];
    }
  }

  private async updateAlertWithCorrelation(alert: Alert, correlatedAlerts: Alert[]) {
    const correlatedAlertIds = correlatedAlerts.map(a => a.id);
    
    await this.storage.updateAlert(alert.id, {
      correlatedAlerts: correlatedAlertIds,
      severity: this.escalateSeverity(alert.severity, correlatedAlerts.length),
      description: alert.description + ` (Correlated with ${correlatedAlerts.length} other alerts)`
    });
  }

  private escalateSeverity(currentSeverity: string, correlationCount: number): string {
    if (correlationCount >= 5) return 'critical';
    if (correlationCount >= 3) return 'high';
    if (correlationCount >= 2 && currentSeverity === 'low') return 'medium';
    return currentSeverity;
  }

  async getCorrelationStats() {
    const alerts = await this.storage.getAlerts(1000);
    
    const correlatedAlerts = alerts.filter(alert => 
      alert.correlatedAlerts && Array.isArray(alert.correlatedAlerts) && alert.correlatedAlerts.length > 0
    );
    
    return {
      totalAlerts: alerts.length,
      correlatedAlerts: correlatedAlerts.length,
      correlationRate: correlatedAlerts.length / alerts.length,
      avgCorrelations: correlatedAlerts.reduce((sum, alert) => 
        sum + (alert.correlatedAlerts as any[]).length, 0) / correlatedAlerts.length,
      severityDistribution: this.calculateSeverityDistribution(alerts)
    };
  }

  private calculateSeverityDistribution(alerts: Alert[]) {
    const distribution: { [key: string]: number } = {};
    
    alerts.forEach(alert => {
      distribution[alert.severity] = (distribution[alert.severity] || 0) + 1;
    });
    
    return distribution;
  }
}
