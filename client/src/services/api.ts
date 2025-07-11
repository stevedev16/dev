import { 
  Event, 
  Alert, 
  FinancialTransaction, 
  GraphEntity, 
  GraphRelationship, 
  SystemMetric,
  DashboardSummary,
  MLModel 
} from '../types';

const API_BASE_URL = '/api';

class ApiClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Events API
  async getEvents(limit?: number, offset?: number): Promise<Event[]> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    return this.request<Event[]>(`/events?${params}`);
  }

  async getEventsByType(type: string): Promise<Event[]> {
    return this.request<Event[]>(`/events/type/${type}`);
  }

  async createEvent(event: Partial<Event>): Promise<Event> {
    return this.request<Event>('/events', {
      method: 'POST',
      body: JSON.stringify(event),
    });
  }

  // Alerts API
  async getAlerts(limit?: number, offset?: number): Promise<Alert[]> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    return this.request<Alert[]>(`/alerts?${params}`);
  }

  async getAlertsByStatus(status: string): Promise<Alert[]> {
    return this.request<Alert[]>(`/alerts/status/${status}`);
  }

  async updateAlert(id: number, updates: Partial<Alert>): Promise<Alert> {
    return this.request<Alert>(`/alerts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  // ML Models API
  async getMLModels(): Promise<MLModel[]> {
    return this.request<MLModel[]>('/ml-models');
  }

  async makePrediction(data: any): Promise<any> {
    return this.request<any>('/ml-models/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Financial Transactions API
  async getFinancialTransactions(limit?: number, offset?: number): Promise<FinancialTransaction[]> {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    return this.request<FinancialTransaction[]>(`/transactions?${params}`);
  }

  async getHighRiskTransactions(threshold: number = 0.7): Promise<FinancialTransaction[]> {
    return this.request<FinancialTransaction[]>(`/transactions/high-risk?threshold=${threshold}`);
  }

  // Graph API
  async getGraphEntities(type?: string): Promise<GraphEntity[]> {
    const params = type ? `?type=${type}` : '';
    return this.request<GraphEntity[]>(`/graph/entities${params}`);
  }

  async getGraphRelationships(from?: string, to?: string): Promise<GraphRelationship[]> {
    const params = new URLSearchParams();
    if (from) params.append('from', from);
    if (to) params.append('to', to);
    
    return this.request<GraphRelationship[]>(`/graph/relationships?${params}`);
  }

  async analyzeGraph(request: any): Promise<any> {
    return this.request<any>('/graph/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // System Metrics API
  async getSystemMetrics(type?: string, limit?: number): Promise<SystemMetric[]> {
    const params = new URLSearchParams();
    if (type) params.append('type', type);
    if (limit) params.append('limit', limit.toString());
    
    return this.request<SystemMetric[]>(`/metrics?${params}`);
  }

  async getLatestMetrics(): Promise<SystemMetric[]> {
    return this.request<SystemMetric[]>('/metrics/latest');
  }

  // Dashboard API
  async getDashboardSummary(): Promise<DashboardSummary> {
    return this.request<DashboardSummary>('/dashboard/summary');
  }

  // Health Check API
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request<{ status: string; timestamp: string }>('/health');
  }
}

// Export individual functions for easier imports
const apiClient = new ApiClient();

export const getEvents = apiClient.getEvents.bind(apiClient);
export const getEventsByType = apiClient.getEventsByType.bind(apiClient);
export const createEvent = apiClient.createEvent.bind(apiClient);

export const getAlerts = apiClient.getAlerts.bind(apiClient);
export const getAlertsByStatus = apiClient.getAlertsByStatus.bind(apiClient);
export const updateAlert = apiClient.updateAlert.bind(apiClient);

export const getMLModels = apiClient.getMLModels.bind(apiClient);
export const makePrediction = apiClient.makePrediction.bind(apiClient);

export const getFinancialTransactions = apiClient.getFinancialTransactions.bind(apiClient);
export const getHighRiskTransactions = apiClient.getHighRiskTransactions.bind(apiClient);

export const getGraphEntities = apiClient.getGraphEntities.bind(apiClient);
export const getGraphRelationships = apiClient.getGraphRelationships.bind(apiClient);
export const analyzeGraph = apiClient.analyzeGraph.bind(apiClient);

export const getSystemMetrics = apiClient.getSystemMetrics.bind(apiClient);
export const getLatestMetrics = apiClient.getLatestMetrics.bind(apiClient);

export const getDashboardSummary = apiClient.getDashboardSummary.bind(apiClient);
export const healthCheck = apiClient.healthCheck.bind(apiClient);

export default apiClient;
