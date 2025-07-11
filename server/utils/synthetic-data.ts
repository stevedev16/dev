import { InsertEvent, InsertFinancialTransaction, InsertGraphEntity, InsertGraphRelationship } from '../storage';

export class SyntheticDataGenerator {
  private userIds: string[] = [];
  private accountIds: string[] = [];
  private ipAddresses: string[] = [];
  private locations: any[] = [];

  constructor() {
    this.initializeBaseData();
  }

  private initializeBaseData() {
    // Generate user IDs
    for (let i = 1; i <= 1000; i++) {
      this.userIds.push(`user_${i.toString().padStart(4, '0')}`);
    }

    // Generate account IDs
    for (let i = 1; i <= 1000; i++) {
      this.accountIds.push(`acc_${i.toString().padStart(6, '0')}`);
    }

    // Generate IP addresses
    for (let i = 0; i < 255; i++) {
      this.ipAddresses.push(`192.168.1.${i}`);
      this.ipAddresses.push(`10.0.0.${i}`);
    }

    // Generate locations
    this.locations = [
      { latitude: 40.7128, longitude: -74.0060, country: 'USA', city: 'New York' },
      { latitude: 34.0522, longitude: -118.2437, country: 'USA', city: 'Los Angeles' },
      { latitude: 51.5074, longitude: -0.1278, country: 'UK', city: 'London' },
      { latitude: 48.8566, longitude: 2.3522, country: 'France', city: 'Paris' },
      { latitude: 35.6762, longitude: 139.6503, country: 'Japan', city: 'Tokyo' },
      { latitude: 52.5200, longitude: 13.4050, country: 'Germany', city: 'Berlin' },
      { latitude: 43.6532, longitude: -79.3832, country: 'Canada', city: 'Toronto' },
      { latitude: -33.8688, longitude: 151.2093, country: 'Australia', city: 'Sydney' }
    ];
  }

  generateEvents(count: number): InsertEvent[] {
    const events: InsertEvent[] = [];
    const eventTypes = ['login', 'logout', 'transaction', 'file_access', 'api_call', 'database_query'];
    const sourceTypes = ['cloud', 'network', 'financial', 'iam', 'spatial'];

    for (let i = 0; i < count; i++) {
      const sourceType = this.randomChoice(sourceTypes);
      const eventType = this.randomChoice(eventTypes);
      const userId = this.randomChoice(this.userIds);
      const ipAddress = this.randomChoice(this.ipAddresses);
      const location = this.randomChoice(this.locations);

      const event: InsertEvent = {
        sourceType,
        eventType,
        timestamp: this.generateRandomTimestamp(),
        rawData: this.generateEventRawData(sourceType, eventType),
        severity: this.randomChoice(['low', 'medium', 'high']),
        userId,
        ipAddress,
        location,
        createdAt: new Date()
      };

      events.push(event);
    }

    return events;
  }

  generateAnomalousEvents(count: number): InsertEvent[] {
    const events: InsertEvent[] = [];
    const anomalousPatterns = [
      'multiple_failed_logins',
      'unusual_time_access',
      'suspicious_location',
      'high_value_transaction',
      'rapid_api_calls',
      'unusual_file_access'
    ];

    for (let i = 0; i < count; i++) {
      const pattern = this.randomChoice(anomalousPatterns);
      const event = this.generateAnomalousEvent(pattern);
      events.push(event);
    }

    return events;
  }

  private generateAnomalousEvent(pattern: string): InsertEvent {
    const userId = this.randomChoice(this.userIds);
    const ipAddress = this.randomChoice(this.ipAddresses);
    
    switch (pattern) {
      case 'multiple_failed_logins':
        return {
          sourceType: 'iam',
          eventType: 'login_failed',
          timestamp: new Date(),
          rawData: {
            userId,
            ipAddress,
            failedAttempts: Math.floor(Math.random() * 10) + 5,
            userAgent: 'Suspicious Bot/1.0',
            reason: 'invalid_credentials'
          },
          severity: 'high',
          userId,
          ipAddress,
          location: this.randomChoice(this.locations),
          createdAt: new Date()
        };

      case 'unusual_time_access':
        const unusualTime = new Date();
        unusualTime.setHours(Math.floor(Math.random() * 6) + 2); // 2-8 AM
        return {
          sourceType: 'iam',
          eventType: 'login',
          timestamp: unusualTime,
          rawData: {
            userId,
            ipAddress,
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            accessType: 'admin_panel'
          },
          severity: 'medium',
          userId,
          ipAddress,
          location: this.randomChoice(this.locations),
          createdAt: new Date()
        };

      case 'suspicious_location':
        return {
          sourceType: 'iam',
          eventType: 'login',
          timestamp: new Date(),
          rawData: {
            userId,
            ipAddress: '192.168.1.999', // Invalid IP
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            vpnDetected: true
          },
          severity: 'high',
          userId,
          ipAddress: '192.168.1.999',
          location: { latitude: 0, longitude: 0, country: 'Unknown', city: 'Unknown' },
          createdAt: new Date()
        };

      case 'high_value_transaction':
        return {
          sourceType: 'financial',
          eventType: 'transaction',
          timestamp: new Date(),
          rawData: {
            transactionId: `tx_${Date.now()}`,
            fromAccount: this.randomChoice(this.accountIds),
            toAccount: this.randomChoice(this.accountIds),
            amount: Math.floor(Math.random() * 50000) + 50000, // $50k-$100k
            currency: 'USD',
            type: 'wire_transfer'
          },
          severity: 'high',
          userId,
          ipAddress,
          location: this.randomChoice(this.locations),
          createdAt: new Date()
        };

      case 'rapid_api_calls':
        return {
          sourceType: 'cloud',
          eventType: 'api_call',
          timestamp: new Date(),
          rawData: {
            apiEndpoint: '/api/sensitive-data',
            method: 'GET',
            requestCount: Math.floor(Math.random() * 100) + 50, // 50-150 calls
            rateLimitExceeded: true,
            responseTime: Math.random() * 50 + 10
          },
          severity: 'medium',
          userId,
          ipAddress,
          location: this.randomChoice(this.locations),
          createdAt: new Date()
        };

      case 'unusual_file_access':
        return {
          sourceType: 'cloud',
          eventType: 'file_access',
          timestamp: new Date(),
          rawData: {
            fileName: 'sensitive_financial_data.csv',
            fileSize: Math.floor(Math.random() * 1000000) + 1000000, // 1MB+
            accessType: 'download',
            permissions: 'unauthorized',
            encryptionStatus: 'unencrypted'
          },
          severity: 'critical',
          userId,
          ipAddress,
          location: this.randomChoice(this.locations),
          createdAt: new Date()
        };

      default:
        return this.generateEvents(1)[0];
    }
  }

  generateFinancialTransactions(count: number): InsertFinancialTransaction[] {
    const transactions: InsertFinancialTransaction[] = [];
    const transactionTypes = ['transfer', 'withdrawal', 'deposit', 'payment', 'wire_transfer'];
    const currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD'];

    for (let i = 0; i < count; i++) {
      const fromAccount = this.randomChoice(this.accountIds);
      const toAccount = this.randomChoice(this.accountIds);
      const amount = Math.floor(Math.random() * 10000) + 100;
      const currency = this.randomChoice(currencies);
      const transactionType = this.randomChoice(transactionTypes);

      const transaction: InsertFinancialTransaction = {
        transactionId: `tx_${Date.now()}_${i}`,
        fromAccount,
        toAccount,
        amount,
        currency,
        transactionType,
        timestamp: this.generateRandomTimestamp(),
        location: this.randomChoice(this.locations),
        riskScore: Math.random(),
        flags: this.generateTransactionFlags(amount, transactionType),
        createdAt: new Date()
      };

      transactions.push(transaction);
    }

    return transactions;
  }

  generateGraphEntities(count: number): InsertGraphEntity[] {
    const entities: InsertGraphEntity[] = [];
    const entityTypes = ['user', 'account', 'device', 'location', 'merchant'];

    for (let i = 0; i < count; i++) {
      const entityType = this.randomChoice(entityTypes);
      const entityId = this.generateEntityId(entityType, i);

      const entity: InsertGraphEntity = {
        entityId,
        entityType,
        properties: this.generateEntityProperties(entityType),
        riskScore: Math.random(),
        createdAt: new Date(),
        updatedAt: new Date()
      };

      entities.push(entity);
    }

    return entities;
  }

  generateGraphRelationships(entities: InsertGraphEntity[]): InsertGraphRelationship[] {
    const relationships: InsertGraphRelationship[] = [];
    const relationshipTypes = ['transacts_with', 'owns', 'accesses', 'connected_to', 'similar_to'];

    // Generate relationships between entities
    for (let i = 0; i < entities.length; i++) {
      const numRelationships = Math.floor(Math.random() * 5) + 1;
      
      for (let j = 0; j < numRelationships; j++) {
        const fromEntity = entities[i];
        const toEntity = entities[Math.floor(Math.random() * entities.length)];
        
        if (fromEntity.entityId !== toEntity.entityId) {
          const relationship: InsertGraphRelationship = {
            fromEntityId: fromEntity.entityId,
            toEntityId: toEntity.entityId,
            relationshipType: this.randomChoice(relationshipTypes),
            weight: Math.random(),
            properties: this.generateRelationshipProperties(),
            createdAt: new Date()
          };

          relationships.push(relationship);
        }
      }
    }

    return relationships;
  }

  private generateEventRawData(sourceType: string, eventType: string): any {
    const baseData = {
      timestamp: new Date().toISOString(),
      sourceType,
      eventType
    };

    switch (sourceType) {
      case 'cloud':
        return {
          ...baseData,
          serviceName: this.randomChoice(['EC2', 'S3', 'Lambda', 'RDS']),
          resourceId: `resource_${Math.floor(Math.random() * 1000)}`,
          action: this.randomChoice(['create', 'delete', 'modify', 'access']),
          responseCode: this.randomChoice([200, 201, 400, 401, 403, 404, 500])
        };

      case 'network':
        return {
          ...baseData,
          protocol: this.randomChoice(['TCP', 'UDP', 'HTTP', 'HTTPS']),
          sourcePort: Math.floor(Math.random() * 65535),
          destinationPort: Math.floor(Math.random() * 65535),
          bytesTransferred: Math.floor(Math.random() * 1000000),
          packetCount: Math.floor(Math.random() * 1000)
        };

      case 'financial':
        return {
          ...baseData,
          amount: Math.floor(Math.random() * 10000) + 100,
          currency: this.randomChoice(['USD', 'EUR', 'GBP']),
          merchantId: `merchant_${Math.floor(Math.random() * 100)}`,
          cardLast4: Math.floor(Math.random() * 10000).toString().padStart(4, '0')
        };

      case 'iam':
        return {
          ...baseData,
          action: this.randomChoice(['login', 'logout', 'permission_change', 'role_assignment']),
          resourceAccessed: this.randomChoice(['dashboard', 'admin_panel', 'reports', 'settings']),
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        };

      case 'spatial':
        return {
          ...baseData,
          coordinates: {
            latitude: (Math.random() - 0.5) * 180,
            longitude: (Math.random() - 0.5) * 360
          },
          accuracy: Math.floor(Math.random() * 100) + 10,
          speed: Math.floor(Math.random() * 100),
          altitude: Math.floor(Math.random() * 1000)
        };

      default:
        return baseData;
    }
  }

  private generateTransactionFlags(amount: number, transactionType: string): any {
    const flags = [];
    
    if (amount > 10000) {
      flags.push('high_value');
    }
    
    if (transactionType === 'wire_transfer') {
      flags.push('wire_transfer');
    }
    
    if (Math.random() > 0.9) {
      flags.push('velocity_check');
    }
    
    return flags;
  }

  private generateEntityId(entityType: string, index: number): string {
    switch (entityType) {
      case 'user':
        return `user_${index.toString().padStart(4, '0')}`;
      case 'account':
        return `acc_${index.toString().padStart(6, '0')}`;
      case 'device':
        return `dev_${index.toString().padStart(4, '0')}`;
      case 'location':
        return `loc_${index.toString().padStart(4, '0')}`;
      case 'merchant':
        return `mer_${index.toString().padStart(4, '0')}`;
      default:
        return `ent_${index.toString().padStart(4, '0')}`;
    }
  }

  private generateEntityProperties(entityType: string): any {
    switch (entityType) {
      case 'user':
        return {
          name: `User ${Math.floor(Math.random() * 1000)}`,
          email: `user${Math.floor(Math.random() * 1000)}@example.com`,
          role: this.randomChoice(['user', 'admin', 'manager']),
          createdDate: this.generateRandomTimestamp().toISOString(),
          lastLogin: this.generateRandomTimestamp().toISOString()
        };

      case 'account':
        return {
          accountType: this.randomChoice(['checking', 'savings', 'credit', 'business']),
          balance: Math.floor(Math.random() * 100000),
          currency: this.randomChoice(['USD', 'EUR', 'GBP']),
          status: this.randomChoice(['active', 'inactive', 'frozen'])
        };

      case 'device':
        return {
          deviceType: this.randomChoice(['mobile', 'desktop', 'tablet', 'server']),
          os: this.randomChoice(['iOS', 'Android', 'Windows', 'macOS', 'Linux']),
          browser: this.randomChoice(['Chrome', 'Firefox', 'Safari', 'Edge']),
          ipAddress: this.randomChoice(this.ipAddresses)
        };

      case 'location':
        const loc = this.randomChoice(this.locations);
        return {
          ...loc,
          timezone: this.randomChoice(['UTC', 'EST', 'PST', 'GMT', 'CET']),
          riskLevel: this.randomChoice(['low', 'medium', 'high'])
        };

      case 'merchant':
        return {
          name: `Merchant ${Math.floor(Math.random() * 1000)}`,
          category: this.randomChoice(['retail', 'restaurant', 'gas', 'grocery', 'online']),
          location: this.randomChoice(this.locations),
          mcc: Math.floor(Math.random() * 9999).toString().padStart(4, '0')
        };

      default:
        return {};
    }
  }

  private generateRelationshipProperties(): any {
    return {
      strength: Math.random(),
      frequency: Math.floor(Math.random() * 100),
      lastInteraction: this.generateRandomTimestamp().toISOString(),
      trustScore: Math.random()
    };
  }

  private generateRandomTimestamp(): Date {
    const now = new Date();
    const pastDays = Math.floor(Math.random() * 30); // Within last 30 days
    const pastMs = pastDays * 24 * 60 * 60 * 1000;
    return new Date(now.getTime() - pastMs);
  }

  private randomChoice<T>(array: T[]): T {
    return array[Math.floor(Math.random() * array.length)];
  }
}
