import { MLPrediction } from '../services/ml-inference';

export interface AnomalyDetectionModel {
  name: string;
  type: 'cnn' | 'lstm' | 'transformer' | 'ensemble';
  version: string;
  threshold: number;
  
  predict(features: any): Promise<MLPrediction>;
  train(trainingData: any[]): Promise<void>;
  evaluate(testData: any[]): Promise<ModelEvaluationMetrics>;
}

export interface ModelEvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
}

export class CNNAnomalyDetector implements AnomalyDetectionModel {
  name = 'CNN Anomaly Detector';
  type: 'cnn' = 'cnn';
  version = '1.0.0';
  threshold = 0.7;

  async predict(features: any): Promise<MLPrediction> {
    // Simulate CNN prediction for network traffic patterns
    const networkFeatures = this.extractNetworkFeatures(features);
    const anomalyScore = this.calculateCNNScore(networkFeatures);
    
    return {
      isAnomaly: anomalyScore > this.threshold,
      confidence: Math.min(anomalyScore + 0.1, 1.0),
      anomalyScore,
      modelType: this.type,
      riskLevel: this.calculateRiskLevel(anomalyScore),
      explanation: this.generateExplanation(networkFeatures, anomalyScore)
    };
  }

  async train(trainingData: any[]): Promise<void> {
    // Simulate CNN training process
    console.log(`Training CNN model with ${trainingData.length} samples`);
    // In a real implementation, this would train the CNN model
  }

  async evaluate(testData: any[]): Promise<ModelEvaluationMetrics> {
    // Simulate model evaluation
    const predictions = await Promise.all(
      testData.map(data => this.predict(data.features))
    );
    
    return this.calculateMetrics(predictions, testData);
  }

  private extractNetworkFeatures(features: any): any {
    return {
      packetSize: features.packetSize || 0,
      protocolType: features.protocolType || 'tcp',
      connectionDuration: features.connectionDuration || 0,
      bytesTransferred: features.bytesTransferred || 0,
      portNumber: features.portNumber || 80,
      flagsSet: features.flagsSet || []
    };
  }

  private calculateCNNScore(networkFeatures: any): number {
    let score = 0;
    
    // Analyze packet size patterns
    if (networkFeatures.packetSize > 1500 || networkFeatures.packetSize < 64) {
      score += 0.2;
    }
    
    // Analyze protocol anomalies
    if (networkFeatures.protocolType === 'unknown') {
      score += 0.3;
    }
    
    // Analyze connection duration
    if (networkFeatures.connectionDuration > 3600) { // > 1 hour
      score += 0.15;
    }
    
    // Analyze bytes transferred
    if (networkFeatures.bytesTransferred > 1000000) { // > 1MB
      score += 0.25;
    }
    
    // Analyze port usage
    const suspiciousPorts = [1433, 3389, 22, 23, 135, 139, 445];
    if (suspiciousPorts.includes(networkFeatures.portNumber)) {
      score += 0.1;
    }
    
    return Math.min(score, 1.0);
  }

  private calculateRiskLevel(anomalyScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (anomalyScore >= 0.9) return 'critical';
    if (anomalyScore >= 0.8) return 'high';
    if (anomalyScore >= 0.6) return 'medium';
    return 'low';
  }

  private generateExplanation(networkFeatures: any, anomalyScore: number): string[] {
    const explanations: string[] = [];
    
    if (networkFeatures.packetSize > 1500) {
      explanations.push('Unusually large packet size detected');
    }
    
    if (networkFeatures.protocolType === 'unknown') {
      explanations.push('Unknown protocol type detected');
    }
    
    if (networkFeatures.bytesTransferred > 1000000) {
      explanations.push('High data transfer volume detected');
    }
    
    if (anomalyScore > 0.8) {
      explanations.push('Multiple anomaly indicators present');
    }
    
    return explanations;
  }

  private calculateMetrics(predictions: MLPrediction[], testData: any[]): ModelEvaluationMetrics {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    predictions.forEach((pred, idx) => {
      const actual = testData[idx].isAnomaly;
      const predicted = pred.isAnomaly;
      
      if (actual && predicted) tp++;
      else if (!actual && predicted) fp++;
      else if (!actual && !predicted) tn++;
      else if (actual && !predicted) fn++;
    });
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      auc: 0.85, // Simulated AUC
      falsePositiveRate: fp / (fp + tn) || 0,
      falseNegativeRate: fn / (fn + tp) || 0
    };
  }
}

export class LSTMAnomalyDetector implements AnomalyDetectionModel {
  name = 'LSTM Anomaly Detector';
  type: 'lstm' = 'lstm';
  version = '1.0.0';
  threshold = 0.7;

  async predict(features: any): Promise<MLPrediction> {
    // Simulate LSTM prediction for sequential patterns
    const sequenceFeatures = this.extractSequenceFeatures(features);
    const anomalyScore = this.calculateLSTMScore(sequenceFeatures);
    
    return {
      isAnomaly: anomalyScore > this.threshold,
      confidence: Math.min(anomalyScore + 0.1, 1.0),
      anomalyScore,
      modelType: this.type,
      riskLevel: this.calculateRiskLevel(anomalyScore),
      explanation: this.generateExplanation(sequenceFeatures, anomalyScore)
    };
  }

  async train(trainingData: any[]): Promise<void> {
    console.log(`Training LSTM model with ${trainingData.length} samples`);
  }

  async evaluate(testData: any[]): Promise<ModelEvaluationMetrics> {
    const predictions = await Promise.all(
      testData.map(data => this.predict(data.features))
    );
    
    return this.calculateMetrics(predictions, testData);
  }

  private extractSequenceFeatures(features: any): any {
    return {
      transactionSequence: features.transactionSequence || [],
      timeIntervals: features.timeIntervals || [],
      amountPatterns: features.amountPatterns || [],
      locationSequence: features.locationSequence || []
    };
  }

  private calculateLSTMScore(sequenceFeatures: any): number {
    let score = 0;
    
    // Analyze transaction sequence patterns
    if (sequenceFeatures.transactionSequence.length > 10) {
      score += 0.2;
    }
    
    // Analyze time intervals
    const avgInterval = sequenceFeatures.timeIntervals.reduce((sum: number, interval: number) => sum + interval, 0) / sequenceFeatures.timeIntervals.length;
    if (avgInterval < 60) { // < 1 minute between transactions
      score += 0.3;
    }
    
    // Analyze amount patterns
    const amounts = sequenceFeatures.amountPatterns;
    if (amounts.length > 0) {
      const avgAmount = amounts.reduce((sum: number, amount: number) => sum + amount, 0) / amounts.length;
      if (avgAmount > 5000) {
        score += 0.25;
      }
    }
    
    // Analyze location sequence
    if (sequenceFeatures.locationSequence.length > 5) {
      score += 0.25;
    }
    
    return Math.min(score, 1.0);
  }

  private calculateRiskLevel(anomalyScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (anomalyScore >= 0.9) return 'critical';
    if (anomalyScore >= 0.8) return 'high';
    if (anomalyScore >= 0.6) return 'medium';
    return 'low';
  }

  private generateExplanation(sequenceFeatures: any, anomalyScore: number): string[] {
    const explanations: string[] = [];
    
    if (sequenceFeatures.transactionSequence.length > 10) {
      explanations.push('Unusually long transaction sequence detected');
    }
    
    const avgInterval = sequenceFeatures.timeIntervals.reduce((sum: number, interval: number) => sum + interval, 0) / sequenceFeatures.timeIntervals.length;
    if (avgInterval < 60) {
      explanations.push('Rapid-fire transaction pattern detected');
    }
    
    if (sequenceFeatures.locationSequence.length > 5) {
      explanations.push('Multiple location changes in short time');
    }
    
    return explanations;
  }

  private calculateMetrics(predictions: MLPrediction[], testData: any[]): ModelEvaluationMetrics {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    predictions.forEach((pred, idx) => {
      const actual = testData[idx].isAnomaly;
      const predicted = pred.isAnomaly;
      
      if (actual && predicted) tp++;
      else if (!actual && predicted) fp++;
      else if (!actual && !predicted) tn++;
      else if (actual && !predicted) fn++;
    });
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      auc: 0.87, // Simulated AUC
      falsePositiveRate: fp / (fp + tn) || 0,
      falseNegativeRate: fn / (fn + tp) || 0
    };
  }
}

export class TransformerAnomalyDetector implements AnomalyDetectionModel {
  name = 'Transformer Anomaly Detector';
  type: 'transformer' = 'transformer';
  version = '1.0.0';
  threshold = 0.7;

  async predict(features: any): Promise<MLPrediction> {
    // Simulate Transformer prediction for complex patterns
    const contextFeatures = this.extractContextFeatures(features);
    const anomalyScore = this.calculateTransformerScore(contextFeatures);
    
    return {
      isAnomaly: anomalyScore > this.threshold,
      confidence: Math.min(anomalyScore + 0.1, 1.0),
      anomalyScore,
      modelType: this.type,
      riskLevel: this.calculateRiskLevel(anomalyScore),
      explanation: this.generateExplanation(contextFeatures, anomalyScore)
    };
  }

  async train(trainingData: any[]): Promise<void> {
    console.log(`Training Transformer model with ${trainingData.length} samples`);
  }

  async evaluate(testData: any[]): Promise<ModelEvaluationMetrics> {
    const predictions = await Promise.all(
      testData.map(data => this.predict(data.features))
    );
    
    return this.calculateMetrics(predictions, testData);
  }

  private extractContextFeatures(features: any): any {
    return {
      userContext: features.userContext || {},
      sessionContext: features.sessionContext || {},
      systemContext: features.systemContext || {},
      temporalContext: features.temporalContext || {},
      spatialContext: features.spatialContext || {}
    };
  }

  private calculateTransformerScore(contextFeatures: any): number {
    let score = 0;
    
    // Analyze user context anomalies
    if (contextFeatures.userContext.newDevice) {
      score += 0.15;
    }
    
    if (contextFeatures.userContext.unusualBehavior) {
      score += 0.2;
    }
    
    // Analyze session context
    if (contextFeatures.sessionContext.duration > 7200) { // > 2 hours
      score += 0.1;
    }
    
    // Analyze system context
    if (contextFeatures.systemContext.highCpuUsage) {
      score += 0.15;
    }
    
    // Analyze temporal context
    if (contextFeatures.temporalContext.outsideBusinessHours) {
      score += 0.2;
    }
    
    // Analyze spatial context
    if (contextFeatures.spatialContext.suspiciousLocation) {
      score += 0.2;
    }
    
    return Math.min(score, 1.0);
  }

  private calculateRiskLevel(anomalyScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (anomalyScore >= 0.9) return 'critical';
    if (anomalyScore >= 0.8) return 'high';
    if (anomalyScore >= 0.6) return 'medium';
    return 'low';
  }

  private generateExplanation(contextFeatures: any, anomalyScore: number): string[] {
    const explanations: string[] = [];
    
    if (contextFeatures.userContext.newDevice) {
      explanations.push('Access from new device detected');
    }
    
    if (contextFeatures.temporalContext.outsideBusinessHours) {
      explanations.push('Activity outside business hours');
    }
    
    if (contextFeatures.spatialContext.suspiciousLocation) {
      explanations.push('Access from suspicious location');
    }
    
    if (contextFeatures.userContext.unusualBehavior) {
      explanations.push('Unusual user behavior pattern detected');
    }
    
    return explanations;
  }

  private calculateMetrics(predictions: MLPrediction[], testData: any[]): ModelEvaluationMetrics {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    predictions.forEach((pred, idx) => {
      const actual = testData[idx].isAnomaly;
      const predicted = pred.isAnomaly;
      
      if (actual && predicted) tp++;
      else if (!actual && predicted) fp++;
      else if (!actual && !predicted) tn++;
      else if (actual && !predicted) fn++;
    });
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      auc: 0.89, // Simulated AUC
      falsePositiveRate: fp / (fp + tn) || 0,
      falseNegativeRate: fn / (fn + tp) || 0
    };
  }
}
