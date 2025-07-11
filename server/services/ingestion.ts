import { IStorage, InsertEvent, InsertSystemMetric } from '../storage';
import { FeatureEngineeringService } from './feature-engineering';
import { MLInferenceService } from './ml-inference';
import { AlertCorrelationService } from './alert-correlation';

export class IngestionService {
  private featureService: FeatureEngineeringService;
  private mlService: MLInferenceService;
  private alertService: AlertCorrelationService;

  constructor(private storage: IStorage) {
    this.featureService = new FeatureEngineeringService(storage);
    this.mlService = new MLInferenceService(storage);
    this.alertService = new AlertCorrelationService(storage);
  }

  async processEvent(eventData: any) {
    const startTime = Date.now();
    
    try {
      // Validate and normalize event data
      const normalizedEvent = this.normalizeEvent(eventData);
      
      // Create event record
      const event = await this.storage.createEvent(normalizedEvent);
      
      // Extract features
      const features = await this.featureService.extractFeatures(event);
      
      // Update event with processed features
      const updatedEvent = {
        ...event,
        processedFeatures: features
      };
      
      // Run ML inference
      const prediction = await this.mlService.predict({
        eventId: event.id,
        features: features,
        eventType: event.eventType,
        sourceType: event.sourceType
      });
      
      // Check for anomalies and create alerts if needed
      if (prediction.isAnomaly) {
        await this.alertService.createAlert(event, prediction);
      }
      
      // Record processing metrics
      const processingTime = Date.now() - startTime;
      await this.recordMetrics(processingTime, event.sourceType);
      
      return updatedEvent;
      
    } catch (error) {
      console.error('Error processing event:', error);
      
      // Record error metrics
      await this.recordErrorMetrics(error, eventData.sourceType);
      
      throw error;
    }
  }

  private normalizeEvent(eventData: any): InsertEvent {
    const now = new Date();
    
    return {
      sourceType: eventData.sourceType || 'unknown',
      eventType: eventData.eventType || 'generic',
      timestamp: eventData.timestamp ? new Date(eventData.timestamp) : now,
      rawData: eventData,
      severity: this.determineSeverity(eventData),
      userId: eventData.userId || eventData.user_id || null,
      ipAddress: eventData.ipAddress || eventData.ip_address || null,
      location: eventData.location || null,
      createdAt: now
    };
  }

  private determineSeverity(eventData: any): string {
    // Basic severity determination logic
    if (eventData.severity) {
      return eventData.severity;
    }
    
    // Check for high-risk indicators
    if (eventData.failed_login_attempts > 5) return 'high';
    if (eventData.suspicious_activity) return 'medium';
    if (eventData.amount && eventData.amount > 10000) return 'medium';
    if (eventData.location && eventData.location.country === 'unknown') return 'medium';
    
    return 'low';
  }

  private async recordMetrics(processingTime: number, sourceType: string) {
    const metrics: InsertSystemMetric[] = [
      {
        metricType: 'processing_latency',
        value: processingTime,
        unit: 'ms',
        metadata: { sourceType }
      },
      {
        metricType: 'events_processed',
        value: 1,
        unit: 'count',
        metadata: { sourceType }
      }
    ];

    await Promise.all(
      metrics.map(metric => this.storage.createSystemMetric(metric))
    );
  }

  private async recordErrorMetrics(error: any, sourceType: string) {
    const errorMetric: InsertSystemMetric = {
      metricType: 'processing_errors',
      value: 1,
      unit: 'count',
      metadata: { 
        sourceType,
        errorType: error.name || 'unknown',
        errorMessage: error.message || 'unknown error'
      }
    };

    await this.storage.createSystemMetric(errorMetric);
  }

  async getIngestionStats() {
    const [
      latencyMetrics,
      throughputMetrics,
      errorMetrics
    ] = await Promise.all([
      this.storage.getSystemMetrics('processing_latency', 100),
      this.storage.getSystemMetrics('events_processed', 100),
      this.storage.getSystemMetrics('processing_errors', 100)
    ]);

    return {
      avgLatency: latencyMetrics.reduce((sum, m) => sum + m.value, 0) / latencyMetrics.length,
      totalProcessed: throughputMetrics.reduce((sum, m) => sum + m.value, 0),
      errorRate: errorMetrics.reduce((sum, m) => sum + m.value, 0) / throughputMetrics.length,
      recentLatency: latencyMetrics.slice(0, 10).map(m => ({
        timestamp: m.timestamp,
        value: m.value
      }))
    };
  }
}
