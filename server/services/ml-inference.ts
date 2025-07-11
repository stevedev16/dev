import { IStorage, InsertSystemMetric } from '../storage';
import axios from 'axios';

export interface MLPrediction {
  isAnomaly: boolean;
  confidence: number;
  anomalyScore: number;
  modelType: string;
  threatType?: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  explanation: string[];
}

export interface MLInferenceRequest {
  eventId: number;
  features: any;
  eventType: string;
  sourceType: string;
}

export class MLInferenceService {
  private readonly ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
  private readonly ANOMALY_THRESHOLD = 0.7;

  constructor(private storage: IStorage) {}

  async predict(request: MLInferenceRequest): Promise<MLPrediction> {
    const startTime = Date.now();
    
    try {
      // Determine which model to use based on source type
      const modelType = this.selectModelType(request.sourceType, request.eventType);
      
      // Make prediction request to ML service
      const response = await axios.post(`${this.ML_SERVICE_URL}/predict`, {
        model_type: modelType,
        features: request.features,
        event_type: request.eventType,
        source_type: request.sourceType
      }, {
        timeout: 5000 // 5 second timeout for real-time requirements
      });

      const prediction = this.processPredictionResponse(response.data, modelType);
      
      // Record inference metrics
      const inferenceTime = Date.now() - startTime;
      await this.recordInferenceMetrics(inferenceTime, modelType, prediction);
      
      return prediction;
      
    } catch (error) {
      console.error('ML inference error:', error);
      
      // Fallback to rule-based detection
      const fallbackPrediction = this.fallbackDetection(request);
      
      // Record error metrics
      await this.recordErrorMetrics(error, request.sourceType);
      
      return fallbackPrediction;
    }
  }

  private selectModelType(sourceType: string, eventType: string): string {
    // Select appropriate model based on event characteristics
    switch (sourceType) {
      case 'network':
        return 'cnn'; // CNN for network traffic patterns
      case 'financial':
        return 'lstm'; // LSTM for sequential financial patterns
      case 'cloud':
      case 'iam':
        return 'transformer'; // Transformer for complex log patterns
      default:
        return 'cnn'; // Default to CNN
    }
  }

  private processPredictionResponse(response: any, modelType: string): MLPrediction {
    const anomalyScore = response.anomaly_score || 0;
    const isAnomaly = anomalyScore > this.ANOMALY_THRESHOLD;
    
    return {
      isAnomaly,
      confidence: response.confidence || 0,
      anomalyScore,
      modelType,
      threatType: response.threat_type,
      riskLevel: this.calculateRiskLevel(anomalyScore),
      explanation: response.explanation || []
    };
  }

  private calculateRiskLevel(anomalyScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (anomalyScore >= 0.9) return 'critical';
    if (anomalyScore >= 0.8) return 'high';
    if (anomalyScore >= 0.6) return 'medium';
    return 'low';
  }

  private fallbackDetection(request: MLInferenceRequest): MLPrediction {
    // Simple rule-based fallback detection
    const features = request.features;
    let anomalyScore = 0;
    const explanation: string[] = [];

    // Check for common anomaly indicators
    if (features.failed_login_attempts > 5) {
      anomalyScore += 0.3;
      explanation.push('Multiple failed login attempts detected');
    }

    if (features.unusual_time_pattern) {
      anomalyScore += 0.2;
      explanation.push('Unusual time pattern detected');
    }

    if (features.suspicious_location) {
      anomalyScore += 0.2;
      explanation.push('Suspicious location detected');
    }

    if (features.high_transaction_amount) {
      anomalyScore += 0.3;
      explanation.push('High transaction amount detected');
    }

    if (features.rare_user_behavior) {
      anomalyScore += 0.2;
      explanation.push('Rare user behavior pattern detected');
    }

    return {
      isAnomaly: anomalyScore > this.ANOMALY_THRESHOLD,
      confidence: Math.min(anomalyScore + 0.1, 1.0),
      anomalyScore,
      modelType: 'fallback_rules',
      riskLevel: this.calculateRiskLevel(anomalyScore),
      explanation
    };
  }

  private async recordInferenceMetrics(inferenceTime: number, modelType: string, prediction: MLPrediction) {
    const metrics: InsertSystemMetric[] = [
      {
        metricType: 'inference_latency',
        value: inferenceTime,
        unit: 'ms',
        metadata: { modelType }
      },
      {
        metricType: 'anomaly_detection_rate',
        value: prediction.isAnomaly ? 1 : 0,
        unit: 'boolean',
        metadata: { modelType, riskLevel: prediction.riskLevel }
      },
      {
        metricType: 'model_confidence',
        value: prediction.confidence,
        unit: 'probability',
        metadata: { modelType }
      }
    ];

    await Promise.all(
      metrics.map(metric => this.storage.createSystemMetric(metric))
    );
  }

  private async recordErrorMetrics(error: any, sourceType: string) {
    const errorMetric: InsertSystemMetric = {
      metricType: 'inference_errors',
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

  async getModelPerformance() {
    const [
      latencyMetrics,
      accuracyMetrics,
      confidenceMetrics
    ] = await Promise.all([
      this.storage.getSystemMetrics('inference_latency', 100),
      this.storage.getSystemMetrics('anomaly_detection_rate', 100),
      this.storage.getSystemMetrics('model_confidence', 100)
    ]);

    return {
      avgLatency: latencyMetrics.reduce((sum, m) => sum + m.value, 0) / latencyMetrics.length,
      detectionRate: accuracyMetrics.reduce((sum, m) => sum + m.value, 0) / accuracyMetrics.length,
      avgConfidence: confidenceMetrics.reduce((sum, m) => sum + m.value, 0) / confidenceMetrics.length,
      modelDistribution: this.calculateModelDistribution(latencyMetrics)
    };
  }

  private calculateModelDistribution(metrics: any[]) {
    const distribution: { [key: string]: number } = {};
    
    metrics.forEach(metric => {
      const modelType = metric.metadata?.modelType || 'unknown';
      distribution[modelType] = (distribution[modelType] || 0) + 1;
    });
    
    return distribution;
  }
}
