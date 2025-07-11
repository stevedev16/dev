import { IStorage, GraphEntity, GraphRelationship } from '../storage';
import axios from 'axios';

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

export class GraphAnalysisService {
  private readonly ORACLE_GRAPH_URL = process.env.ORACLE_GRAPH_URL || 'http://localhost:7007/oraclegd';

  constructor(private storage: IStorage) {}

  async analyzeGraph(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    try {
      switch (request.analysisType) {
        case 'community_detection':
          return await this.performCommunityDetection(request);
        case 'centrality':
          return await this.performCentralityAnalysis(request);
        case 'anomaly_detection':
          return await this.performGraphAnomalyDetection(request);
        case 'path_analysis':
          return await this.performPathAnalysis(request);
        default:
          throw new Error(`Unknown analysis type: ${request.analysisType}`);
      }
    } catch (error) {
      console.error('Graph analysis error:', error);
      
      // Fallback to basic analysis
      return await this.performBasicAnalysis(request);
    }
  }

  private async performCommunityDetection(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    // Get all entities and relationships
    const entities = await this.storage.getGraphEntities();
    const relationships = await this.storage.getGraphRelationships();

    // Build adjacency lists for community detection
    const adjacencyList = this.buildAdjacencyList(entities, relationships);
    
    // Simple community detection using connected components
    const communities = this.findConnectedComponents(adjacencyList);
    
    const insights: string[] = [];
    let riskScore = 0;

    // Analyze communities
    communities.forEach((community, index) => {
      if (community.size > 10) {
        insights.push(`Large community detected with ${community.size} entities`);
        riskScore += 0.2;
      }
      
      if (community.size === 1) {
        insights.push(`Isolated entity detected in community ${index}`);
        riskScore += 0.1;
      }
    });

    return {
      analysisType: 'community_detection',
      results: {
        communities: communities.map(c => Array.from(c)),
        totalCommunities: communities.length,
        largestCommunity: Math.max(...communities.map(c => c.size))
      },
      insights,
      riskScore: Math.min(riskScore, 1.0),
      timestamp: new Date()
    };
  }

  private async performCentralityAnalysis(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    const entities = await this.storage.getGraphEntities();
    const relationships = await this.storage.getGraphRelationships();

    // Calculate different centrality measures
    const degreeCentrality = this.calculateDegreeCentrality(entities, relationships);
    const betweennessCentrality = this.calculateBetweennessCentrality(entities, relationships);
    
    const insights: string[] = [];
    let riskScore = 0;

    // Identify high-centrality entities
    const sortedByCentrality = Object.entries(degreeCentrality)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10);

    sortedByCentrality.forEach(([entityId, centrality]) => {
      if (centrality > 10) {
        insights.push(`High-centrality entity detected: ${entityId} (${centrality} connections)`);
        riskScore += 0.1;
      }
    });

    return {
      analysisType: 'centrality',
      entityId: request.entityId,
      results: {
        degreeCentrality,
        betweennessCentrality,
        topCentralEntities: sortedByCentrality
      },
      insights,
      riskScore: Math.min(riskScore, 1.0),
      timestamp: new Date()
    };
  }

  private async performGraphAnomalyDetection(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    const entities = await this.storage.getGraphEntities();
    const relationships = await this.storage.getGraphRelationships();

    const anomalies: any[] = [];
    const insights: string[] = [];
    let riskScore = 0;

    // Detect structural anomalies
    const degreeCentrality = this.calculateDegreeCentrality(entities, relationships);
    const avgDegree = Object.values(degreeCentrality).reduce((sum, d) => sum + d, 0) / entities.length;
    const stdDev = Math.sqrt(
      Object.values(degreeCentrality).reduce((sum, d) => sum + Math.pow(d - avgDegree, 2), 0) / entities.length
    );

    // Find outliers (entities with degree > 2 standard deviations from mean)
    Object.entries(degreeCentrality).forEach(([entityId, degree]) => {
      if (Math.abs(degree - avgDegree) > 2 * stdDev) {
        anomalies.push({
          entityId,
          anomalyType: 'degree_outlier',
          degree,
          zscore: (degree - avgDegree) / stdDev
        });
        insights.push(`Degree outlier detected: ${entityId} (${degree} connections)`);
        riskScore += 0.2;
      }
    });

    // Detect rapid connection changes
    const recentRelationships = relationships.filter(r => 
      new Date(r.createdAt).getTime() > Date.now() - 24 * 60 * 60 * 1000
    );

    const recentConnectionCounts: { [key: string]: number } = {};
    recentRelationships.forEach(r => {
      recentConnectionCounts[r.fromEntityId] = (recentConnectionCounts[r.fromEntityId] || 0) + 1;
      recentConnectionCounts[r.toEntityId] = (recentConnectionCounts[r.toEntityId] || 0) + 1;
    });

    Object.entries(recentConnectionCounts).forEach(([entityId, count]) => {
      if (count > 10) {
        anomalies.push({
          entityId,
          anomalyType: 'rapid_connections',
          recentConnections: count
        });
        insights.push(`Rapid connection growth detected: ${entityId} (${count} new connections in 24h)`);
        riskScore += 0.3;
      }
    });

    return {
      analysisType: 'anomaly_detection',
      results: {
        anomalies,
        totalAnomalies: anomalies.length,
        avgDegree,
        stdDev
      },
      insights,
      riskScore: Math.min(riskScore, 1.0),
      timestamp: new Date()
    };
  }

  private async performPathAnalysis(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    if (!request.entityId) {
      throw new Error('Entity ID required for path analysis');
    }

    const entities = await this.storage.getGraphEntities();
    const relationships = await this.storage.getGraphRelationships();

    // Build graph representation
    const graph = this.buildGraphRepresentation(entities, relationships);
    
    // Find shortest paths from the entity
    const shortestPaths = this.findShortestPaths(graph, request.entityId);
    
    const insights: string[] = [];
    let riskScore = 0;

    // Analyze path patterns
    const pathLengths = Object.values(shortestPaths);
    const avgPathLength = pathLengths.reduce((sum, len) => sum + len, 0) / pathLengths.length;

    if (avgPathLength < 2) {
      insights.push('Entity has very short paths to most other entities (potential hub)');
      riskScore += 0.2;
    }

    // Find entities within 2 hops
    const nearbyEntities = Object.entries(shortestPaths)
      .filter(([, distance]) => distance <= 2)
      .map(([entityId]) => entityId);

    if (nearbyEntities.length > 20) {
      insights.push(`Entity has ${nearbyEntities.length} entities within 2 hops`);
      riskScore += 0.15;
    }

    return {
      analysisType: 'path_analysis',
      entityId: request.entityId,
      results: {
        shortestPaths,
        avgPathLength,
        nearbyEntities,
        reachableEntities: Object.keys(shortestPaths).length
      },
      insights,
      riskScore: Math.min(riskScore, 1.0),
      timestamp: new Date()
    };
  }

  private async performBasicAnalysis(request: GraphAnalysisRequest): Promise<GraphAnalysisResult> {
    const entities = await this.storage.getGraphEntities();
    const relationships = await this.storage.getGraphRelationships();

    return {
      analysisType: 'basic',
      results: {
        totalEntities: entities.length,
        totalRelationships: relationships.length,
        entityTypes: this.getEntityTypeDistribution(entities),
        relationshipTypes: this.getRelationshipTypeDistribution(relationships)
      },
      insights: ['Basic graph statistics computed'],
      riskScore: 0,
      timestamp: new Date()
    };
  }

  private buildAdjacencyList(entities: GraphEntity[], relationships: GraphRelationship[]): Map<string, Set<string>> {
    const adjacencyList = new Map<string, Set<string>>();
    
    // Initialize with all entities
    entities.forEach(entity => {
      adjacencyList.set(entity.entityId, new Set());
    });

    // Add relationships
    relationships.forEach(rel => {
      if (!adjacencyList.has(rel.fromEntityId)) {
        adjacencyList.set(rel.fromEntityId, new Set());
      }
      if (!adjacencyList.has(rel.toEntityId)) {
        adjacencyList.set(rel.toEntityId, new Set());
      }
      
      adjacencyList.get(rel.fromEntityId)!.add(rel.toEntityId);
      adjacencyList.get(rel.toEntityId)!.add(rel.fromEntityId);
    });

    return adjacencyList;
  }

  private findConnectedComponents(adjacencyList: Map<string, Set<string>>): Set<string>[] {
    const visited = new Set<string>();
    const components: Set<string>[] = [];

    const dfs = (node: string, component: Set<string>) => {
      visited.add(node);
      component.add(node);
      
      const neighbors = adjacencyList.get(node) || new Set();
      neighbors.forEach(neighbor => {
        if (!visited.has(neighbor)) {
          dfs(neighbor, component);
        }
      });
    };

    adjacencyList.forEach((_, node) => {
      if (!visited.has(node)) {
        const component = new Set<string>();
        dfs(node, component);
        components.push(component);
      }
    });

    return components;
  }

  private calculateDegreeCentrality(entities: GraphEntity[], relationships: GraphRelationship[]): { [key: string]: number } {
    const centrality: { [key: string]: number } = {};
    
    // Initialize
    entities.forEach(entity => {
      centrality[entity.entityId] = 0;
    });

    // Count connections
    relationships.forEach(rel => {
      centrality[rel.fromEntityId] = (centrality[rel.fromEntityId] || 0) + 1;
      centrality[rel.toEntityId] = (centrality[rel.toEntityId] || 0) + 1;
    });

    return centrality;
  }

  private calculateBetweennessCentrality(entities: GraphEntity[], relationships: GraphRelationship[]): { [key: string]: number } {
    // Simplified betweenness centrality calculation
    const centrality: { [key: string]: number } = {};
    
    entities.forEach(entity => {
      centrality[entity.entityId] = 0;
    });

    // This is a simplified version - in practice, you'd implement the full algorithm
    // For now, we'll use a proxy based on degree and local clustering
    const degreeCentrality = this.calculateDegreeCentrality(entities, relationships);
    
    Object.entries(degreeCentrality).forEach(([entityId, degree]) => {
      centrality[entityId] = degree * 0.5; // Simplified proxy
    });

    return centrality;
  }

  private buildGraphRepresentation(entities: GraphEntity[], relationships: GraphRelationship[]): { [key: string]: string[] } {
    const graph: { [key: string]: string[] } = {};
    
    entities.forEach(entity => {
      graph[entity.entityId] = [];
    });

    relationships.forEach(rel => {
      if (!graph[rel.fromEntityId]) graph[rel.fromEntityId] = [];
      if (!graph[rel.toEntityId]) graph[rel.toEntityId] = [];
      
      graph[rel.fromEntityId].push(rel.toEntityId);
      graph[rel.toEntityId].push(rel.fromEntityId);
    });

    return graph;
  }

  private findShortestPaths(graph: { [key: string]: string[] }, startEntity: string): { [key: string]: number } {
    const distances: { [key: string]: number } = {};
    const queue: string[] = [startEntity];
    
    distances[startEntity] = 0;

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = graph[current] || [];
      
      neighbors.forEach(neighbor => {
        if (distances[neighbor] === undefined) {
          distances[neighbor] = distances[current] + 1;
          queue.push(neighbor);
        }
      });
    }

    return distances;
  }

  private getEntityTypeDistribution(entities: GraphEntity[]): { [key: string]: number } {
    const distribution: { [key: string]: number } = {};
    
    entities.forEach(entity => {
      distribution[entity.entityType] = (distribution[entity.entityType] || 0) + 1;
    });

    return distribution;
  }

  private getRelationshipTypeDistribution(relationships: GraphRelationship[]): { [key: string]: number } {
    const distribution: { [key: string]: number } = {};
    
    relationships.forEach(rel => {
      distribution[rel.relationshipType] = (distribution[rel.relationshipType] || 0) + 1;
    });

    return distribution;
  }
}
