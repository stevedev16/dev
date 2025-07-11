import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { getGraphEntities, getGraphRelationships, analyzeGraph } from '../services/api';
import { GraphEntity, GraphRelationship } from '../types';

interface GraphData {
  nodes: GraphEntity[];
  links: GraphRelationship[];
}

const GraphVisualization: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<GraphEntity | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [analysisType, setAnalysisType] = useState<'community_detection' | 'centrality' | 'anomaly_detection'>('community_detection');

  useEffect(() => {
    loadGraphData();
  }, []);

  useEffect(() => {
    if (graphData.nodes.length > 0 && graphData.links.length > 0) {
      renderGraph();
    }
  }, [graphData]);

  const loadGraphData = async () => {
    try {
      setLoading(true);
      const [entities, relationships] = await Promise.all([
        getGraphEntities(),
        getGraphRelationships()
      ]);
      
      setGraphData({
        nodes: entities,
        links: relationships
      });
      
      setError(null);
    } catch (err) {
      console.error('Error loading graph data:', err);
      setError('Failed to load graph data');
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    try {
      setLoading(true);
      const result = await analyzeGraph({
        analysisType,
        entityId: selectedEntity?.entityId
      });
      setAnalysisResults(result);
    } catch (err) {
      console.error('Error running graph analysis:', err);
      setError('Failed to run graph analysis');
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = () => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    const g = svg.append('g');

    // Create force simulation
    const simulation = d3.forceSimulation<GraphEntity>(graphData.nodes)
      .force('link', d3.forceLink<GraphEntity, GraphRelationship>(graphData.links)
        .id(d => d.entityId)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(graphData.links)
      .enter()
      .append('line')
      .attr('class', 'graph-link')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.weight || 1) * 2);

    // Create nodes
    const node = g.append('g')
      .selectAll('g')
      .data(graphData.nodes)
      .enter()
      .append('g')
      .attr('class', 'graph-node')
      .call(d3.drag<SVGGElement, GraphEntity>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // Add circles for nodes
    node.append('circle')
      .attr('r', d => Math.sqrt((d.riskScore || 0) * 100) + 8)
      .attr('fill', d => getNodeColor(d.entityType))
      .attr('stroke', d => selectedEntity?.entityId === d.entityId ? '#ff0000' : '#fff')
      .attr('stroke-width', d => selectedEntity?.entityId === d.entityId ? 3 : 1.5);

    // Add labels for nodes
    node.append('text')
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#333')
      .text(d => d.entityId.substring(0, 8));

    // Add click handler for nodes
    node.on('click', (event, d) => {
      setSelectedEntity(d);
      
      // Update visual selection
      node.select('circle')
        .attr('stroke', n => n.entityId === d.entityId ? '#ff0000' : '#fff')
        .attr('stroke-width', n => n.entityId === d.entityId ? 3 : 1.5);
    });

    // Add tooltips
    node.append('title')
      .text(d => `${d.entityType}: ${d.entityId}\nRisk Score: ${(d.riskScore || 0).toFixed(2)}`);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y);

      node
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event: any, d: GraphEntity) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: GraphEntity) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: GraphEntity) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  const getNodeColor = (entityType: string) => {
    switch (entityType) {
      case 'user': return '#3B82F6';
      case 'account': return '#10B981';
      case 'device': return '#F59E0B';
      case 'location': return '#EF4444';
      case 'merchant': return '#8B5CF6';
      default: return '#6B7280';
    }
  };

  const renderAnalysisResults = () => {
    if (!analysisResults) return null;

    return (
      <div className="analysis-results">
        <h4>Analysis Results: {analysisResults.analysisType}</h4>
        <div className="analysis-metrics">
          <div className="metric">
            <span className="metric-label">Risk Score:</span>
            <span className="metric-value">{(analysisResults.riskScore * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Timestamp:</span>
            <span className="metric-value">{new Date(analysisResults.timestamp).toLocaleString()}</span>
          </div>
        </div>
        
        <div className="analysis-insights">
          <h5>Insights:</h5>
          <ul>
            {analysisResults.insights.map((insight: string, index: number) => (
              <li key={index}>{insight}</li>
            ))}
          </ul>
        </div>
        
        {analysisResults.results && (
          <div className="analysis-details">
            <h5>Detailed Results:</h5>
            <pre>{JSON.stringify(analysisResults.results, null, 2)}</pre>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="graph-loading">
        <div className="loading-spinner" />
        <p>Loading graph visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-error">
        <div className="error-icon">❌</div>
        <h3>Error Loading Graph</h3>
        <p>{error}</p>
        <button onClick={loadGraphData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <div className="graph-header">
        <h3>Graph Analysis</h3>
        <div className="graph-controls">
          <select 
            value={analysisType} 
            onChange={(e) => setAnalysisType(e.target.value as any)}
            className="analysis-select"
          >
            <option value="community_detection">Community Detection</option>
            <option value="centrality">Centrality Analysis</option>
            <option value="anomaly_detection">Anomaly Detection</option>
          </select>
          <button onClick={runAnalysis} className="analyze-button">
            Run Analysis
          </button>
        </div>
      </div>
      
      <div className="graph-content">
        <div className="graph-main">
          <svg 
            ref={svgRef} 
            width="800" 
            height="600"
            className="graph-svg"
          />
          
          <div className="graph-legend">
            <h4>Entity Types</h4>
            <div className="legend-items">
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#3B82F6' }} />
                <span>User</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#10B981' }} />
                <span>Account</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#F59E0B' }} />
                <span>Device</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#EF4444' }} />
                <span>Location</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ backgroundColor: '#8B5CF6' }} />
                <span>Merchant</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="graph-sidebar">
          {selectedEntity && (
            <div className="entity-details">
              <h4>Entity Details</h4>
              <div className="entity-info">
                <div className="info-item">
                  <span className="info-label">ID:</span>
                  <span className="info-value">{selectedEntity.entityId}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Type:</span>
                  <span className="info-value">{selectedEntity.entityType}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Risk Score:</span>
                  <span className="info-value">{(selectedEntity.riskScore || 0).toFixed(2)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Created:</span>
                  <span className="info-value">{new Date(selectedEntity.createdAt).toLocaleString()}</span>
                </div>
              </div>
              
              {selectedEntity.properties && (
                <div className="entity-properties">
                  <h5>Properties:</h5>
                  <pre>{JSON.stringify(selectedEntity.properties, null, 2)}</pre>
                </div>
              )}
            </div>
          )}
          
          {renderAnalysisResults()}
        </div>
      </div>
    </div>
  );
};

export default GraphVisualization;
