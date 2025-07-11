package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

// StreamEvent represents an event in the stream processing pipeline
type StreamEvent struct {
	ID           string                 `json:"id"`
	SourceType   string                 `json:"sourceType"`
	EventType    string                 `json:"eventType"`
	Timestamp    time.Time              `json:"timestamp"`
	RawData      map[string]interface{} `json:"rawData"`
	Features     map[string]interface{} `json:"features,omitempty"`
	Severity     string                 `json:"severity"`
	UserID       string                 `json:"userId,omitempty"`
	IPAddress    string                 `json:"ipAddress,omitempty"`
	Location     *Location              `json:"location,omitempty"`
	ProcessingID string                 `json:"processingId"`
	Pipeline     []string               `json:"pipeline"`
}

// Location represents geographical coordinates
type Location struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Country   string  `json:"country"`
	City      string  `json:"city"`
}

// ProcessingRule defines stream processing rules
type ProcessingRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	SourceTypes []string          `json:"sourceTypes"`
	EventTypes  []string          `json:"eventTypes"`
	Conditions  []RuleCondition   `json:"conditions"`
	Actions     []RuleAction      `json:"actions"`
	Enabled     bool              `json:"enabled"`
	Priority    int               `json:"priority"`
	Metadata    map[string]string `json:"metadata"`
}

// RuleCondition defines a condition for rule matching
type RuleCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
	Type     string      `json:"type"`
}

// RuleAction defines an action to take when rule matches
type RuleAction struct {
	Type       string                 `json:"type"`
	Target     string                 `json:"target"`
	Parameters map[string]interface{} `json:"parameters"`
}

// StreamProcessor handles real-time stream processing
type StreamProcessor struct {
	inputQueue     chan StreamEvent
	outputQueue    chan StreamEvent
	alertQueue     chan StreamEvent
	rules          []ProcessingRule
	processors     map[string]Processor
	metrics        *ProcessingMetrics
	config         *ProcessorConfig
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	rulesMutex     sync.RWMutex
}

// Processor interface for different processing stages
type Processor interface {
	Process(event *StreamEvent) error
	GetName() string
	GetMetrics() map[string]interface{}
}

// ProcessingMetrics tracks stream processing performance
type ProcessingMetrics struct {
	TotalEvents       int64         `json:"totalEvents"`
	ProcessedEvents   int64         `json:"processedEvents"`
	FilteredEvents    int64         `json:"filteredEvents"`
	AlertsGenerated   int64         `json:"alertsGenerated"`
	AvgProcessingTime time.Duration `json:"avgProcessingTime"`
	Throughput        float64       `json:"throughput"`
	ErrorRate         float64       `json:"errorRate"`
	LastReset         time.Time     `json:"lastReset"`
	mutex             sync.RWMutex
}

// ProcessorConfig holds configuration
type ProcessorConfig struct {
	Port               int           `json:"port"`
	InputQueueSize     int           `json:"inputQueueSize"`
	OutputQueueSize    int           `json:"outputQueueSize"`
	AlertQueueSize     int           `json:"alertQueueSize"`
	WorkerCount        int           `json:"workerCount"`
	ProcessingTimeout  time.Duration `json:"processingTimeout"`
	MetricsInterval    time.Duration `json:"metricsInterval"`
	RulesRefreshInterval time.Duration `json:"rulesRefreshInterval"`
	MLServiceURL       string        `json:"mlServiceUrl"`
	DatabaseURL        string        `json:"databaseUrl"`
}

// Feature extraction processor
type FeatureExtractor struct {
	name    string
	metrics map[string]interface{}
	mutex   sync.RWMutex
}

// Anomaly detection processor
type AnomalyDetector struct {
	name         string
	mlServiceURL string
	metrics      map[string]interface{}
	mutex        sync.RWMutex
}

// Alert correlation processor
type AlertCorrelator struct {
	name    string
	metrics map[string]interface{}
	mutex   sync.RWMutex
}

// Data enrichment processor
type DataEnricher struct {
	name    string
	metrics map[string]interface{}
	mutex   sync.RWMutex
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(config *ProcessorConfig) *StreamProcessor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &StreamProcessor{
		inputQueue:  make(chan StreamEvent, config.InputQueueSize),
		outputQueue: make(chan StreamEvent, config.OutputQueueSize),
		alertQueue:  make(chan StreamEvent, config.AlertQueueSize),
		rules:       initDefaultRules(),
		processors:  initProcessors(config),
		metrics:     &ProcessingMetrics{LastReset: time.Now()},
		config:      config,
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start initializes and starts the stream processor
func (sp *StreamProcessor) Start() error {
	log.Printf("Starting FinSecure Nexus Stream Processor on port %d", sp.config.Port)
	
	// Start worker goroutines
	for i := 0; i < sp.config.WorkerCount; i++ {
		sp.wg.Add(1)
		go sp.processingWorker(i)
	}
	
	// Start output handler
	sp.wg.Add(1)
	go sp.outputHandler()
	
	// Start alert handler
	sp.wg.Add(1)
	go sp.alertHandler()
	
	// Start metrics collector
	sp.wg.Add(1)
	go sp.metricsCollector()
	
	// Start rules refresher
	sp.wg.Add(1)
	go sp.rulesRefresher()
	
	// Setup HTTP routes
	router := mux.NewRouter()
	sp.setupRoutes(router)
	
	// Setup CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
	})
	
	handler := c.Handler(router)
	
	// Start HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", sp.config.Port),
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	
	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()
	
	log.Printf("✅ Stream processor started successfully")
	
	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	
	log.Println("Shutting down stream processor...")
	
	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}
	
	sp.cancel()
	sp.wg.Wait()
	
	log.Println("✅ Stream processor shut down gracefully")
	return nil
}

// setupRoutes configures HTTP routes
func (sp *StreamProcessor) setupRoutes(router *mux.Router) {
	// Event processing endpoints
	router.HandleFunc("/api/process/event", sp.handleProcessEvent).Methods("POST")
	router.HandleFunc("/api/process/batch", sp.handleProcessBatch).Methods("POST")
	
	// Rule management endpoints
	router.HandleFunc("/api/rules", sp.handleGetRules).Methods("GET")
	router.HandleFunc("/api/rules", sp.handleCreateRule).Methods("POST")
	router.HandleFunc("/api/rules/{id}", sp.handleUpdateRule).Methods("PUT")
	router.HandleFunc("/api/rules/{id}", sp.handleDeleteRule).Methods("DELETE")
	
	// Health and metrics endpoints
	router.HandleFunc("/api/health", sp.handleHealth).Methods("GET")
	router.HandleFunc("/api/metrics", sp.handleMetrics).Methods("GET")
	router.HandleFunc("/api/status", sp.handleStatus).Methods("GET")
	
	// Queue status endpoints
	router.HandleFunc("/api/queues/status", sp.handleQueueStatus).Methods("GET")
	router.HandleFunc("/api/processors/status", sp.handleProcessorStatus).Methods("GET")
}

// processingWorker processes events from the input queue
func (sp *StreamProcessor) processingWorker(workerID int) {
	defer sp.wg.Done()
	
	log.Printf("Processing worker %d started", workerID)
	
	for {
		select {
		case event := <-sp.inputQueue:
			sp.processEvent(event, workerID)
			
		case <-sp.ctx.Done():
			log.Printf("Processing worker %d shutting down", workerID)
			return
		}
	}
}

// processEvent processes a single event through the pipeline
func (sp *StreamProcessor) processEvent(event StreamEvent, workerID int) {
	startTime := time.Now()
	
	// Update metrics
	sp.updateMetrics(true, false, false, 0)
	
	// Apply processing rules
	if !sp.applyRules(&event) {
		sp.updateMetrics(false, true, false, time.Since(startTime))
		return
	}
	
	// Add processing stage
	event.Pipeline = append(event.Pipeline, "stream_processor")
	
	// Process through each processor
	for _, processor := range sp.processors {
		if err := processor.Process(&event); err != nil {
			log.Printf("Worker %d: Processor %s failed: %v", workerID, processor.GetName(), err)
			sp.updateMetrics(false, false, false, time.Since(startTime))
			return
		}
	}
	
	// Check if event should generate alert
	if sp.shouldGenerateAlert(&event) {
		select {
		case sp.alertQueue <- event:
			sp.updateMetrics(false, false, true, time.Since(startTime))
		default:
			log.Printf("Alert queue full, dropping alert for event %s", event.ID)
		}
	}
	
	// Send to output queue
	select {
	case sp.outputQueue <- event:
		sp.updateMetrics(false, false, false, time.Since(startTime))
		log.Printf("Worker %d: Processed event %s (%s/%s)", 
			workerID, event.ID, event.SourceType, event.EventType)
	default:
		log.Printf("Output queue full, dropping event %s", event.ID)
	}
}

// applyRules applies processing rules to determine if event should be processed
func (sp *StreamProcessor) applyRules(event *StreamEvent) bool {
	sp.rulesMutex.RLock()
	defer sp.rulesMutex.RUnlock()
	
	for _, rule := range sp.rules {
		if !rule.Enabled {
			continue
		}
		
		// Check if rule applies to this event
		if !sp.ruleMatches(rule, event) {
			continue
		}
		
		// Apply rule actions
		for _, action := range rule.Actions {
			sp.applyRuleAction(action, event)
		}
		
		// If rule action is to filter, return false
		for _, action := range rule.Actions {
			if action.Type == "filter" {
				return false
			}
		}
	}
	
	return true
}

// ruleMatches checks if a rule matches the given event
func (sp *StreamProcessor) ruleMatches(rule ProcessingRule, event *StreamEvent) bool {
	// Check source types
	if len(rule.SourceTypes) > 0 && !contains(rule.SourceTypes, event.SourceType) {
		return false
	}
	
	// Check event types
	if len(rule.EventTypes) > 0 && !contains(rule.EventTypes, event.EventType) {
		return false
	}
	
	// Check conditions
	for _, condition := range rule.Conditions {
		if !sp.evaluateCondition(condition, event) {
			return false
		}
	}
	
	return true
}

// evaluateCondition evaluates a single rule condition
func (sp *StreamProcessor) evaluateCondition(condition RuleCondition, event *StreamEvent) bool {
	var fieldValue interface{}
	
	// Extract field value
	switch condition.Field {
	case "severity":
		fieldValue = event.Severity
	case "userId":
		fieldValue = event.UserID
	case "ipAddress":
		fieldValue = event.IPAddress
	default:
		if event.RawData != nil {
			fieldValue = event.RawData[condition.Field]
		}
	}
	
	// Apply operator
	switch condition.Operator {
	case "equals":
		return fieldValue == condition.Value
	case "not_equals":
		return fieldValue != condition.Value
	case "contains":
		if str, ok := fieldValue.(string); ok {
			if pattern, ok := condition.Value.(string); ok {
				return strings.Contains(strings.ToLower(str), strings.ToLower(pattern))
			}
		}
	case "greater_than":
		if num, ok := fieldValue.(float64); ok {
			if threshold, ok := condition.Value.(float64); ok {
				return num > threshold
			}
		}
	case "less_than":
		if num, ok := fieldValue.(float64); ok {
			if threshold, ok := condition.Value.(float64); ok {
				return num < threshold
			}
		}
	}
	
	return false
}

// applyRuleAction applies a rule action to an event
func (sp *StreamProcessor) applyRuleAction(action RuleAction, event *StreamEvent) {
	switch action.Type {
	case "enrich":
		// Add enrichment data
		if event.RawData == nil {
			event.RawData = make(map[string]interface{})
		}
		for key, value := range action.Parameters {
			event.RawData[key] = value
		}
		
	case "tag":
		// Add tags
		if event.RawData == nil {
			event.RawData = make(map[string]interface{})
		}
		tags, _ := event.RawData["tags"].([]string)
		if tag, ok := action.Parameters["tag"].(string); ok {
			tags = append(tags, tag)
			event.RawData["tags"] = tags
		}
		
	case "severity":
		// Update severity
		if newSeverity, ok := action.Parameters["severity"].(string); ok {
			event.Severity = newSeverity
		}
		
	case "alert":
		// Mark for alerting
		if event.RawData == nil {
			event.RawData = make(map[string]interface{})
		}
		event.RawData["alert_flag"] = true
		event.RawData["alert_reason"] = action.Parameters["reason"]
	}
}

// shouldGenerateAlert determines if an event should generate an alert
func (sp *StreamProcessor) shouldGenerateAlert(event *StreamEvent) bool {
	// Check alert flag
	if event.RawData != nil {
		if alertFlag, ok := event.RawData["alert_flag"].(bool); ok && alertFlag {
			return true
		}
	}
	
	// Check severity
	if event.Severity == "high" || event.Severity == "critical" {
		return true
	}
	
	// Check for anomaly indicators
	if event.Features != nil {
		if anomalyScore, ok := event.Features["anomaly_score"].(float64); ok && anomalyScore > 0.7 {
			return true
		}
	}
	
	return false
}

// outputHandler handles processed events
func (sp *StreamProcessor) outputHandler() {
	defer sp.wg.Done()
	
	log.Println("Output handler started")
	
	for {
		select {
		case event := <-sp.outputQueue:
			// In a real implementation, this would:
			// 1. Send to database for storage
			// 2. Send to downstream services
			// 3. Update search indexes
			// 4. Send to real-time dashboard
			
			log.Printf("Output: %s/%s (ID: %s, Severity: %s)", 
				event.SourceType, event.EventType, event.ID, event.Severity)
			
		case <-sp.ctx.Done():
			log.Println("Output handler shutting down")
			return
		}
	}
}

// alertHandler handles alert generation
func (sp *StreamProcessor) alertHandler() {
	defer sp.wg.Done()
	
	log.Println("Alert handler started")
	
	for {
		select {
		case event := <-sp.alertQueue:
			// Generate alert
			sp.generateAlert(event)
			
		case <-sp.ctx.Done():
			log.Println("Alert handler shutting down")
			return
		}
	}
}

// generateAlert creates an alert from an event
func (sp *StreamProcessor) generateAlert(event StreamEvent) {
	alert := map[string]interface{}{
		"event_id":    event.ID,
		"alert_type":  "anomaly",
		"severity":    event.Severity,
		"description": fmt.Sprintf("Anomaly detected in %s event", event.SourceType),
		"timestamp":   time.Now().Format(time.RFC3339),
		"event_data":  event,
	}
	
	// In a real implementation, this would send to alerting system
	log.Printf("ALERT: %s - %s", alert["severity"], alert["description"])
}

// metricsCollector periodically collects metrics
func (sp *StreamProcessor) metricsCollector() {
	defer sp.wg.Done()
	
	ticker := time.NewTicker(sp.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sp.logMetrics()
			
		case <-sp.ctx.Done():
			return
		}
	}
}

// rulesRefresher periodically refreshes processing rules
func (sp *StreamProcessor) rulesRefresher() {
	defer sp.wg.Done()
	
	ticker := time.NewTicker(sp.config.RulesRefreshInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sp.refreshRules()
			
		case <-sp.ctx.Done():
			return
		}
	}
}

// updateMetrics updates processing metrics
func (sp *StreamProcessor) updateMetrics(total, filtered, alert bool, processingTime time.Duration) {
	sp.metrics.mutex.Lock()
	defer sp.metrics.mutex.Unlock()
	
	if total {
		sp.metrics.TotalEvents++
	}
	
	if filtered {
		sp.metrics.FilteredEvents++
	} else if total {
		sp.metrics.ProcessedEvents++
	}
	
	if alert {
		sp.metrics.AlertsGenerated++
	}
	
	if processingTime > 0 {
		// Update average processing time
		if sp.metrics.ProcessedEvents == 1 {
			sp.metrics.AvgProcessingTime = processingTime
		} else {
			// Exponential moving average
			sp.metrics.AvgProcessingTime = time.Duration(
				0.9*float64(sp.metrics.AvgProcessingTime) + 0.1*float64(processingTime),
			)
		}
	}
	
	// Update throughput
	duration := time.Since(sp.metrics.LastReset).Seconds()
	if duration > 0 {
		sp.metrics.Throughput = float64(sp.metrics.ProcessedEvents) / duration
	}
	
	// Calculate error rate
	if sp.metrics.TotalEvents > 0 {
		failedEvents := sp.metrics.TotalEvents - sp.metrics.ProcessedEvents - sp.metrics.FilteredEvents
		sp.metrics.ErrorRate = float64(failedEvents) / float64(sp.metrics.TotalEvents) * 100
	}
}

// logMetrics logs current metrics
func (sp *StreamProcessor) logMetrics() {
	sp.metrics.mutex.RLock()
	defer sp.metrics.mutex.RUnlock()
	
	log.Printf("Metrics - Total: %d, Processed: %d, Filtered: %d, Alerts: %d, Avg Time: %v, Throughput: %.2f/s, Error Rate: %.2f%%",
		sp.metrics.TotalEvents,
		sp.metrics.ProcessedEvents,
		sp.metrics.FilteredEvents,
		sp.metrics.AlertsGenerated,
		sp.metrics.AvgProcessingTime,
		sp.metrics.Throughput,
		sp.metrics.ErrorRate,
	)
}

// refreshRules refreshes processing rules from database
func (sp *StreamProcessor) refreshRules() {
	// In a real implementation, this would load rules from database
	log.Println("Refreshing processing rules...")
}

// HTTP handlers

func (sp *StreamProcessor) handleProcessEvent(w http.ResponseWriter, r *http.Request) {
	var event StreamEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	// Add to processing queue
	select {
	case sp.inputQueue <- event:
		response := map[string]interface{}{
			"status":    "accepted",
			"event_id":  event.ID,
			"queue_size": len(sp.inputQueue),
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		
	default:
		http.Error(w, "Processing queue full", http.StatusServiceUnavailable)
	}
}

func (sp *StreamProcessor) handleProcessBatch(w http.ResponseWriter, r *http.Request) {
	var events []StreamEvent
	if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	accepted := 0
	for _, event := range events {
		select {
		case sp.inputQueue <- event:
			accepted++
		default:
			break
		}
	}
	
	response := map[string]interface{}{
		"status":        "processed",
		"total_events":  len(events),
		"accepted":      accepted,
		"queue_size":    len(sp.inputQueue),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (sp *StreamProcessor) handleGetRules(w http.ResponseWriter, r *http.Request) {
	sp.rulesMutex.RLock()
	rules := sp.rules
	sp.rulesMutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rules)
}

func (sp *StreamProcessor) handleCreateRule(w http.ResponseWriter, r *http.Request) {
	var rule ProcessingRule
	if err := json.NewDecoder(r.Body).Decode(&rule); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	sp.rulesMutex.Lock()
	sp.rules = append(sp.rules, rule)
	sp.rulesMutex.Unlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rule)
}

func (sp *StreamProcessor) handleUpdateRule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	ruleID := vars["id"]
	
	var updatedRule ProcessingRule
	if err := json.NewDecoder(r.Body).Decode(&updatedRule); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	sp.rulesMutex.Lock()
	for i, rule := range sp.rules {
		if rule.ID == ruleID {
			sp.rules[i] = updatedRule
			break
		}
	}
	sp.rulesMutex.Unlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(updatedRule)
}

func (sp *StreamProcessor) handleDeleteRule(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	ruleID := vars["id"]
	
	sp.rulesMutex.Lock()
	for i, rule := range sp.rules {
		if rule.ID == ruleID {
			sp.rules = append(sp.rules[:i], sp.rules[i+1:]...)
			break
		}
	}
	sp.rulesMutex.Unlock()
	
	w.WriteHeader(http.StatusNoContent)
}

func (sp *StreamProcessor) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":      "healthy",
		"service":     "finsecure-stream-processor",
		"version":     "1.0.0",
		"timestamp":   time.Now().Format(time.RFC3339),
		"queue_sizes": map[string]int{
			"input":  len(sp.inputQueue),
			"output": len(sp.outputQueue),
			"alert":  len(sp.alertQueue),
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (sp *StreamProcessor) handleMetrics(w http.ResponseWriter, r *http.Request) {
	sp.metrics.mutex.RLock()
	metrics := *sp.metrics
	sp.metrics.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (sp *StreamProcessor) handleStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"service":      "finsecure-stream-processor",
		"status":       "running",
		"worker_count": sp.config.WorkerCount,
		"queue_capacities": map[string]int{
			"input":  cap(sp.inputQueue),
			"output": cap(sp.outputQueue),
			"alert":  cap(sp.alertQueue),
		},
		"queue_sizes": map[string]int{
			"input":  len(sp.inputQueue),
			"output": len(sp.outputQueue),
			"alert":  len(sp.alertQueue),
		},
		"processors": len(sp.processors),
		"rules":      len(sp.rules),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (sp *StreamProcessor) handleQueueStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"input_queue": map[string]interface{}{
			"size":     len(sp.inputQueue),
			"capacity": cap(sp.inputQueue),
			"usage":    float64(len(sp.inputQueue)) / float64(cap(sp.inputQueue)) * 100,
		},
		"output_queue": map[string]interface{}{
			"size":     len(sp.outputQueue),
			"capacity": cap(sp.outputQueue),
			"usage":    float64(len(sp.outputQueue)) / float64(cap(sp.outputQueue)) * 100,
		},
		"alert_queue": map[string]interface{}{
			"size":     len(sp.alertQueue),
			"capacity": cap(sp.alertQueue),
			"usage":    float64(len(sp.alertQueue)) / float64(cap(sp.alertQueue)) * 100,
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (sp *StreamProcessor) handleProcessorStatus(w http.ResponseWriter, r *http.Request) {
	status := make(map[string]interface{})
	
	for name, processor := range sp.processors {
		status[name] = processor.GetMetrics()
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Processor implementations

func (fe *FeatureExtractor) Process(event *StreamEvent) error {
	fe.mutex.Lock()
	defer fe.mutex.Unlock()
	
	// Extract features from raw data
	features := make(map[string]interface{})
	
	// Basic features
	features["timestamp"] = event.Timestamp.Unix()
	features["hour_of_day"] = event.Timestamp.Hour()
	features["day_of_week"] = int(event.Timestamp.Weekday())
	features["source_type"] = event.SourceType
	features["event_type"] = event.EventType
	
	// Raw data features
	if event.RawData != nil {
		for key, value := range event.RawData {
			features[key] = value
		}
	}
	
	// Location features
	if event.Location != nil {
		features["has_location"] = true
		features["latitude"] = event.Location.Latitude
		features["longitude"] = event.Location.Longitude
		features["country"] = event.Location.Country
	} else {
		features["has_location"] = false
	}
	
	event.Features = features
	return nil
}

func (fe *FeatureExtractor) GetName() string {
	return fe.name
}

func (fe *FeatureExtractor) GetMetrics() map[string]interface{} {
	fe.mutex.RLock()
	defer fe.mutex.RUnlock()
	return fe.metrics
}

func (ad *AnomalyDetector) Process(event *StreamEvent) error {
	ad.mutex.Lock()
	defer ad.mutex.Unlock()
	
	// In a real implementation, this would call the ML service
	// For now, we'll add some basic anomaly detection logic
	
	anomalyScore := 0.0
	
	// Check for high severity
	if event.Severity == "high" || event.Severity == "critical" {
		anomalyScore += 0.3
	}
	
	// Check for unusual time
	hour := event.Timestamp.Hour()
	if hour < 6 || hour > 22 {
		anomalyScore += 0.2
	}
	
	// Check raw data for indicators
	if event.RawData != nil {
		if event.RawData["failed_attempts"] != nil {
			if attempts, ok := event.RawData["failed_attempts"].(float64); ok && attempts > 5 {
				anomalyScore += 0.4
			}
		}
		
		if event.RawData["amount"] != nil {
			if amount, ok := event.RawData["amount"].(float64); ok && amount > 10000 {
				anomalyScore += 0.3
			}
		}
	}
	
	// Add anomaly score to features
	if event.Features == nil {
		event.Features = make(map[string]interface{})
	}
	event.Features["anomaly_score"] = anomalyScore
	event.Features["is_anomaly"] = anomalyScore > 0.7
	
	return nil
}

func (ad *AnomalyDetector) GetName() string {
	return ad.name
}

func (ad *AnomalyDetector) GetMetrics() map[string]interface{} {
	ad.mutex.RLock()
	defer ad.mutex.RUnlock()
	return ad.metrics
}

func (ac *AlertCorrelator) Process(event *StreamEvent) error {
	ac.mutex.Lock()
	defer ac.mutex.Unlock()
	
	// Basic alert correlation logic
	if event.Features != nil {
		if anomalyScore, ok := event.Features["anomaly_score"].(float64); ok {
			if anomalyScore > 0.8 {
				if event.RawData == nil {
					event.RawData = make(map[string]interface{})
				}
				event.RawData["alert_priority"] = "high"
				event.RawData["correlation_reason"] = "high_anomaly_score"
			}
		}
	}
	
	return nil
}

func (ac *AlertCorrelator) GetName() string {
	return ac.name
}

func (ac *AlertCorrelator) GetMetrics() map[string]interface{} {
	ac.mutex.RLock()
	defer ac.mutex.RUnlock()
	return ac.metrics
}

func (de *DataEnricher) Process(event *StreamEvent) error {
	de.mutex.Lock()
	defer de.mutex.Unlock()
	
	// Add enrichment data
	if event.RawData == nil {
		event.RawData = make(map[string]interface{})
	}
	
	event.RawData["processed_timestamp"] = time.Now().Format(time.RFC3339)
	event.RawData["processor_version"] = "1.0.0"
	
	// Source-specific enrichment
	switch event.SourceType {
	case "financial":
		event.RawData["risk_category"] = "financial_crime"
	case "network":
		event.RawData["risk_category"] = "cyber_security"
	case "iam":
		event.RawData["risk_category"] = "access_control"
	}
	
	return nil
}

func (de *DataEnricher) GetName() string {
	return de.name
}

func (de *DataEnricher) GetMetrics() map[string]interface{} {
	de.mutex.RLock()
	defer de.mutex.RUnlock()
	return de.metrics
}

// Utility functions

func getDefaultConfig() *ProcessorConfig {
	return &ProcessorConfig{
		Port:                 8002,
		InputQueueSize:       50000,
		OutputQueueSize:      50000,
		AlertQueueSize:       10000,
		WorkerCount:          8,
		ProcessingTimeout:    time.Second * 30,
		MetricsInterval:      time.Second * 30,
		RulesRefreshInterval: time.Minute * 5,
		MLServiceURL:         "http://localhost:8000",
		DatabaseURL:          "postgres://localhost/finsecure",
	}
}

func initProcessors(config *ProcessorConfig) map[string]Processor {
	return map[string]Processor{
		"feature_extractor": &FeatureExtractor{
			name:    "FeatureExtractor",
			metrics: make(map[string]interface{}),
		},
		"anomaly_detector": &AnomalyDetector{
			name:         "AnomalyDetector",
			mlServiceURL: config.MLServiceURL,
			metrics:      make(map[string]interface{}),
		},
		"alert_correlator": &AlertCorrelator{
			name:    "AlertCorrelator",
			metrics: make(map[string]interface{}),
		},
		"data_enricher": &DataEnricher{
			name:    "DataEnricher",
			metrics: make(map[string]interface{}),
		},
	}
}

func initDefaultRules() []ProcessingRule {
	return []ProcessingRule{
		{
			ID:          "high_severity_alert",
			Name:        "High Severity Alert Rule",
			SourceTypes: []string{"financial", "network", "iam"},
			EventTypes:  []string{},
			Conditions: []RuleCondition{
				{Field: "severity", Operator: "equals", Value: "high", Type: "string"},
			},
			Actions: []RuleAction{
				{Type: "alert", Parameters: map[string]interface{}{"reason": "high_severity_event"}},
			},
			Enabled:  true,
			Priority: 1,
		},
		{
			ID:          "failed_login_correlation",
			Name:        "Failed Login Correlation Rule",
			SourceTypes: []string{"iam"},
			EventTypes:  []string{"login_failed"},
			Conditions: []RuleCondition{
				{Field: "failed_attempts", Operator: "greater_than", Value: 5.0, Type: "number"},
			},
			Actions: []RuleAction{
				{Type: "tag", Parameters: map[string]interface{}{"tag": "brute_force_attempt"}},
				{Type: "severity", Parameters: map[string]interface{}{"severity": "high"}},
			},
			Enabled:  true,
			Priority: 2,
		},
		{
			ID:          "large_transaction_filter",
			Name:        "Large Transaction Filter",
			SourceTypes: []string{"financial"},
			EventTypes:  []string{"transaction"},
			Conditions: []RuleCondition{
				{Field: "amount", Operator: "greater_than", Value: 50000.0, Type: "number"},
			},
			Actions: []RuleAction{
				{Type: "enrich", Parameters: map[string]interface{}{"large_transaction": true}},
				{Type: "alert", Parameters: map[string]interface{}{"reason": "large_financial_transaction"}},
			},
			Enabled:  true,
			Priority: 3,
		},
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func main() {
	// Load configuration
	config := getDefaultConfig()
	
	// Override with environment variables
	if port := os.Getenv("PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			config.Port = p
		}
	}
	
	if workerCount := os.Getenv("WORKER_COUNT"); workerCount != "" {
		if w, err := strconv.Atoi(workerCount); err == nil {
			config.WorkerCount = w
		}
	}
	
	if mlServiceURL := os.Getenv("ML_SERVICE_URL"); mlServiceURL != "" {
		config.MLServiceURL = mlServiceURL
	}
	
	if databaseURL := os.Getenv("DATABASE_URL"); databaseURL != "" {
		config.DatabaseURL = databaseURL
	}
	
	// Create and start service
	processor := NewStreamProcessor(config)
	
	if err := processor.Start(); err != nil {
		log.Fatalf("Stream processor failed to start: %v", err)
	}
}
