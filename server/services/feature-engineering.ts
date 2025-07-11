import { IStorage, Event } from '../storage';

export interface ExtractedFeatures {
  // Traditional features
  eventFrequency: number;
  uniqueUsers: number;
  uniqueIPs: number;
  
  // Temporal features
  hourOfDay: number;
  dayOfWeek: number;
  isWeekend: boolean;
  timeFromLastEvent: number;
  
  // Spatial features
  location?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  distanceFromLastLocation?: number;
  locationRisk: number;
  
  // Graph-derived features
  userConnectivity?: number;
  accountCentrality?: number;
  transactionChainLength?: number;
  
  // Domain-specific features
  failedLoginAttempts?: number;
  transactionAmount?: number;
  suspiciousPatterns: string[];
  
  // Behavioral features
  userBehaviorDeviation: number;
  accessPatternAnomaly: number;
  
  // Network features
  networkActivity?: number;
  protocolDistribution?: { [key: string]: number };
  
  // Aggregated features
  last24hActivity: number;
  last7dActivity: number;
  
  // Risk indicators
  riskScore: number;
  confidenceScore: number;
}

export class FeatureEngineeringService {
  constructor(private storage: IStorage) {}

  async extractFeatures(event: Event): Promise<ExtractedFeatures> {
    const [
      traditionalFeatures,
      temporalFeatures,
      spatialFeatures,
      graphFeatures,
      behavioralFeatures,
      networkFeatures
    ] = await Promise.all([
      this.extractTraditionalFeatures(event),
      this.extractTemporalFeatures(event),
      this.extractSpatialFeatures(event),
      this.extractGraphFeatures(event),
      this.extractBehavioralFeatures(event),
      this.extractNetworkFeatures(event)
    ]);

    const features: ExtractedFeatures = {
      ...traditionalFeatures,
      ...temporalFeatures,
      ...spatialFeatures,
      ...graphFeatures,
      ...behavioralFeatures,
      ...networkFeatures
    };

    // Calculate overall risk score
    features.riskScore = this.calculateRiskScore(features);
    features.confidenceScore = this.calculateConfidenceScore(features);

    return features;
  }

  private async extractTraditionalFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    const last1Hour = new Date(Date.now() - 60 * 60 * 1000);
    const recentEvents = await this.storage.getEventsByTimeRange(last1Hour, new Date());

    const uniqueUsers = new Set(recentEvents.map(e => e.userId).filter(Boolean)).size;
    const uniqueIPs = new Set(recentEvents.map(e => e.ipAddress).filter(Boolean)).size;

    let failedLoginAttempts = 0;
    let transactionAmount = 0;
    const suspiciousPatterns: string[] = [];

    if (event.sourceType === 'iam') {
      failedLoginAttempts = recentEvents.filter(e => 
        e.eventType === 'login_failed' && e.userId === event.userId
      ).length;
      
      if (failedLoginAttempts > 3) {
        suspiciousPatterns.push('multiple_failed_logins');
      }
    }

    if (event.sourceType === 'financial') {
      transactionAmount = event.rawData?.amount || 0;
      
      if (transactionAmount > 10000) {
        suspiciousPatterns.push('high_value_transaction');
      }
    }

    return {
      eventFrequency: recentEvents.length,
      uniqueUsers,
      uniqueIPs,
      failedLoginAttempts,
      transactionAmount,
      suspiciousPatterns
    };
  }

  private async extractTemporalFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    const eventTime = new Date(event.timestamp);
    const hourOfDay = eventTime.getHours();
    const dayOfWeek = eventTime.getDay();
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

    // Find last event from same user/IP
    const recentEvents = await this.storage.getEvents(100);
    const lastEvent = recentEvents.find(e => 
      e.userId === event.userId || e.ipAddress === event.ipAddress
    );

    const timeFromLastEvent = lastEvent 
      ? eventTime.getTime() - new Date(lastEvent.timestamp).getTime()
      : 0;

    return {
      hourOfDay,
      dayOfWeek,
      isWeekend,
      timeFromLastEvent
    };
  }

  private async extractSpatialFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    const location = event.location as any;
    if (!location) {
      return { locationRisk: 0.5 }; // Medium risk for unknown location
    }

    // Calculate distance from last known location
    const recentEvents = await this.storage.getEvents(50);
    const lastLocationEvent = recentEvents.find(e => 
      e.userId === event.userId && e.location
    );

    let distanceFromLastLocation = 0;
    if (lastLocationEvent?.location) {
      const lastLoc = lastLocationEvent.location as any;
      distanceFromLastLocation = this.calculateDistance(
        location.latitude, location.longitude,
        lastLoc.latitude, lastLoc.longitude
      );
    }

    // Calculate location risk based on various factors
    const locationRisk = this.calculateLocationRisk(location, distanceFromLastLocation);

    return {
      location: {
        latitude: location.latitude,
        longitude: location.longitude,
        country: location.country,
        city: location.city
      },
      distanceFromLastLocation,
      locationRisk
    };
  }

  private async extractGraphFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    if (!event.userId) {
      return {};
    }

    // Get graph entities and relationships
    const entities = await this.storage.getGraphEntities('user');
    const relationships = await this.storage.getGraphRelationships(event.userId);

    const userEntity = entities.find(e => e.entityId === event.userId);
    
    if (!userEntity) {
      return {};
    }

    // Calculate graph-derived features
    const userConnectivity = relationships.length;
    const accountCentrality = this.calculateCentrality(userEntity, relationships);
    const transactionChainLength = await this.calculateTransactionChainLength(event.userId);

    return {
      userConnectivity,
      accountCentrality,
      transactionChainLength
    };
  }

  private async extractBehavioralFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    if (!event.userId) {
      return { userBehaviorDeviation: 0.5, accessPatternAnomaly: 0.5 };
    }

    // Get historical behavior for this user
    const last30Days = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const userEvents = await this.storage.getEventsByTimeRange(last30Days, new Date());
    const userSpecificEvents = userEvents.filter(e => e.userId === event.userId);

    // Calculate behavioral deviation
    const userBehaviorDeviation = this.calculateBehavioralDeviation(event, userSpecificEvents);
    const accessPatternAnomaly = this.calculateAccessPatternAnomaly(event, userSpecificEvents);

    return {
      userBehaviorDeviation,
      accessPatternAnomaly
    };
  }

  private async extractNetworkFeatures(event: Event): Promise<Partial<ExtractedFeatures>> {
    if (event.sourceType !== 'network') {
      return {};
    }

    const networkData = event.rawData;
    const protocolDistribution: { [key: string]: number } = {};

    // Analyze network patterns
    if (networkData.protocol) {
      protocolDistribution[networkData.protocol] = 1;
    }

    const networkActivity = networkData.bytes_transferred || 0;

    return {
      networkActivity,
      protocolDistribution
    };
  }

  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRadians(lat2 - lat1);
    const dLon = this.toRadians(lon2 - lon1);
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }

  private toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  private calculateLocationRisk(location: any, distance: number): number {
    let risk = 0;

    // High-risk countries
    const highRiskCountries = ['Unknown', 'TOR', 'VPN'];
    if (highRiskCountries.includes(location.country)) {
      risk += 0.4;
    }

    // Large distance from last location
    if (distance > 1000) { // > 1000 km
      risk += 0.3;
    }

    // Time-based risk (unusual hours)
    const hour = new Date().getHours();
    if (hour < 6 || hour > 22) {
      risk += 0.2;
    }

    return Math.min(risk, 1.0);
  }

  private calculateCentrality(entity: any, relationships: any[]): number {
    // Simple degree centrality calculation
    return relationships.length / 10; // Normalized by expected max connections
  }

  private async calculateTransactionChainLength(userId: string): Promise<number> {
    const transactions = await this.storage.getFinancialTransactions(1000);
    const userTransactions = transactions.filter(t => 
      t.fromAccount === userId || t.toAccount === userId
    );

    // Calculate chain length using simple traversal
    let maxChainLength = 0;
    const visited = new Set<string>();

    const dfs = (accountId: string, depth: number) => {
      if (visited.has(accountId) || depth > 10) return;
      
      visited.add(accountId);
      maxChainLength = Math.max(maxChainLength, depth);

      const connectedTransactions = userTransactions.filter(t => 
        t.fromAccount === accountId || t.toAccount === accountId
      );

      connectedTransactions.forEach(t => {
        const nextAccount = t.fromAccount === accountId ? t.toAccount : t.fromAccount;
        dfs(nextAccount, depth + 1);
      });
    };

    dfs(userId, 0);
    return maxChainLength;
  }

  private calculateBehavioralDeviation(event: Event, historicalEvents: Event[]): number {
    if (historicalEvents.length === 0) return 0.5;

    // Calculate typical behavior patterns
    const typicalHours = historicalEvents.map(e => new Date(e.timestamp).getHours());
    const avgHour = typicalHours.reduce((sum, h) => sum + h, 0) / typicalHours.length;
    
    const currentHour = new Date(event.timestamp).getHours();
    const hourDeviation = Math.abs(currentHour - avgHour) / 12; // Normalized

    // Calculate typical source types
    const sourceTypes = historicalEvents.map(e => e.sourceType);
    const currentSourceFreq = sourceTypes.filter(s => s === event.sourceType).length;
    const sourceDeviation = currentSourceFreq === 0 ? 1 : 1 / (currentSourceFreq + 1);

    return (hourDeviation + sourceDeviation) / 2;
  }

  private calculateAccessPatternAnomaly(event: Event, historicalEvents: Event[]): number {
    if (historicalEvents.length === 0) return 0.5;

    // Analyze access patterns
    const eventTypes = historicalEvents.map(e => e.eventType);
    const currentEventFreq = eventTypes.filter(t => t === event.eventType).length;
    
    // Calculate pattern anomaly
    const patternAnomaly = currentEventFreq === 0 ? 1 : 1 / (currentEventFreq + 1);

    return Math.min(patternAnomaly, 1.0);
  }

  private calculateRiskScore(features: ExtractedFeatures): number {
    let riskScore = 0;
    let totalWeight = 0;

    // Weight different feature types
    const weights = {
      failedLogins: 0.3,
      highValueTransaction: 0.25,
      locationRisk: 0.15,
      behavioralDeviation: 0.15,
      temporalAnomaly: 0.1,
      networkActivity: 0.05
    };

    if (features.failedLoginAttempts && features.failedLoginAttempts > 3) {
      riskScore += weights.failedLogins;
    }
    totalWeight += weights.failedLogins;

    if (features.transactionAmount && features.transactionAmount > 10000) {
      riskScore += weights.highValueTransaction;
    }
    totalWeight += weights.highValueTransaction;

    riskScore += features.locationRisk * weights.locationRisk;
    totalWeight += weights.locationRisk;

    riskScore += features.userBehaviorDeviation * weights.behavioralDeviation;
    totalWeight += weights.behavioralDeviation;

    // Temporal anomaly (unusual hours)
    const isUnusualHour = features.hourOfDay < 6 || features.hourOfDay > 22;
    if (isUnusualHour) {
      riskScore += weights.temporalAnomaly;
    }
    totalWeight += weights.temporalAnomaly;

    if (features.networkActivity && features.networkActivity > 1000000) {
      riskScore += weights.networkActivity;
    }
    totalWeight += weights.networkActivity;

    return Math.min(riskScore / totalWeight, 1.0);
  }

  private calculateConfidenceScore(features: ExtractedFeatures): number {
    // Calculate confidence based on data completeness and quality
    let confidence = 0;
    let factors = 0;

    if (features.eventFrequency > 0) {
      confidence += 0.2;
    }
    factors += 0.2;

    if (features.location) {
      confidence += 0.2;
    }
    factors += 0.2;

    if (features.userConnectivity !== undefined) {
      confidence += 0.2;
    }
    factors += 0.2;

    if (features.suspiciousPatterns.length > 0) {
      confidence += 0.2;
    }
    factors += 0.2;

    if (features.last24hActivity > 0) {
      confidence += 0.2;
    }
    factors += 0.2;

    return confidence / factors;
  }
}
