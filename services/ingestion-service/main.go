package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

// Event represents an incoming security event
type Event struct {
	ID           string                 `json:"id"`
	SourceType   string                 `json:"sourceType"`
	EventType    string                 `json:"eventType"`
	Timestamp    time.Time              `json:"timestamp"`
	RawData      map[string]interface{} `json:"rawData"`
	Severity     string                 `json:"severity"`
	UserID       string                 `json:"userId,omitempty"`
	IPAddress    string                 `json:"ipAddress,omitempty"`
	Location     *Location              `json:"location,omitempty"`
	ProcessingID string                 `json:"processingId"`
}

// Location represents geographical location data
type Location struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Country   string  `json:"country"`
	City      string  `json:"city"`
}

// ValidationRule represents a data validation rule
type ValidationRule struct {
	Field    string      `json:"field"`
	Required bool        `json:"required"`
	Type     string      `json:"type"`
	Pattern  string      `json:"pattern,omitempty"`
	MinLen   int         `json:"minLen,omitempty"`
	MaxLen   int         `json:"maxLen,omitempty"`
	Values   []string    `json:"values,omitempty"`
}

// IngestionMetrics tracks performance metrics
type IngestionMetrics struct {
	TotalEvents     int64         `json:"totalEvents"`
	ProcessedEvents int64         `json:"processedEvents"`
	FailedEvents    int64         `json:"failedEvents"`
	AvgLatency      time.Duration `json:"avgLatency"`
	Throughput      float64       `json:"throughput"`
	LastReset       time.Time     `json:"lastReset"`
	mutex           sync.RWMutex
}

// IngestionService handles high-throughput event ingestion
type IngestionService struct {
	eventQueue    chan Event
	batchQueue    chan []Event
	metrics       *IngestionMetrics
	validationRules map[string][]ValidationRule
	config        *Config
	ctx           context.Context
	cancel        context.CancelFunc
	wg            sync.WaitGroup
}

// Config holds service configuration
type Config struct {
	Port                int           `json:"port"`
	QueueSize          int           `json:"queueSize"`
	BatchSize          int           `json:"batchSize"`
	BatchTimeout       time.Duration `json:"batchTimeout"`
	WorkerCount        int           `json:"workerCount"`
	ValidationEnabled  bool          `json:"validationEnabled"`
	MetricsInterval    time.Duration `json:"metricsInterval"`
	HealthCheckEnabled bool          `json:"healthCheckEnabled"`
}

// Default configuration
func getDefaultConfig() *Config {
	return &Config{
		Port:                8001,
		QueueSize:          100000,
		BatchSize:          1000,
		BatchTimeout:       time.Second * 5,
		WorkerCount:        10,
		ValidationEnabled:  true,
		MetricsInterval:    time.Second * 30,
		HealthCheckEnabled: true,
	}
}

// NewIngestionService creates a new ingestion service
func NewIngestionService(config *Config) *IngestionService {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &IngestionService{
		eventQueue:    make(chan Event, config.QueueSize),
		batchQueue:    make(chan []Event, 100),
		metrics:       &IngestionMetrics{LastReset: time.Now()},
		validationRules: initValidationRules(),
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start initializes and starts the ingestion service
func (is *IngestionService) Start() error {
	log.Printf("Starting FinSecure Nexus Ingestion Service on port %d", is.config.Port)
	
	// Start worker goroutines
	for i := 0; i < is.config.WorkerCount; i++ {
		is.wg.Add(1)
		go is.eventWorker(i)
	}
	
	// Start batch processor
	is.wg.Add(1)
	go is.batchProcessor()
	
	// Start metrics collector
	if is.config.MetricsInterval > 0 {
		is.wg.Add(1)
		go is.metricsCollector()
	}
	
	// Setup HTTP routes
	router := mux.NewRouter()
	is.setupRoutes(router)
	
	// Setup CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
	})
	
	handler := c.Handler(router)
	
	// Start HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", is.config.Port),
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
	
	log.Printf("✅ Ingestion service started successfully")
	
	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	
	log.Println("Shutting down ingestion service...")
	
	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}
	
	is.cancel()
	is.wg.Wait()
	
	log.Println("✅ Ingestion service shut down gracefully")
	return nil
}

// setupRoutes configures HTTP routes
func (is *IngestionService) setupRoutes(router *mux.Router) {
	// Event ingestion endpoints
	router.HandleFunc("/api/ingest/event", is.handleSingleEvent).Methods("POST")
	router.HandleFunc("/api/ingest/batch", is.handleBatchEvents).Methods("POST")
	
	// Health and metrics endpoints
	router.HandleFunc("/api/health", is.handleHealth).Methods("GET")
	router.HandleFunc("/api/metrics", is.handleMetrics).Methods("GET")
	router.HandleFunc("/api/status", is.handleStatus).Methods("GET")
	
	// Configuration endpoints
	router.HandleFunc("/api/config", is.handleGetConfig).Methods("GET")
	router.HandleFunc("/api/validation-rules", is.handleGetValidationRules).Methods("GET")
	
	// Utility endpoints
	router.HandleFunc("/api/queue-status", is.handleQueueStatus).Methods("GET")
}

// handleSingleEvent processes a single event
func (is *IngestionService) handleSingleEvent(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	
	var event Event
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		is.updateMetrics(startTime, false)
		return
	}
	
	// Validate event
	if is.config.ValidationEnabled {
		if err := is.validateEvent(&event); err != nil {
			http.Error(w, fmt.Sprintf("Validation failed: %v", err), http.StatusBadRequest)
			is.updateMetrics(startTime, false)
			return
		}
	}
	
	// Enrich event
	is.enrichEvent(&event)
	
	// Queue event for processing
	select {
	case is.eventQueue <- event:
		is.updateMetrics(startTime, true)
		
		response := map[string]interface{}{
			"status":       "accepted",
			"processingId": event.ProcessingID,
			"queueSize":    len(is.eventQueue),
			"timestamp":    time.Now().Format(time.RFC3339),
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		
	default:
		http.Error(w, "Queue full - try again later", http.StatusServiceUnavailable)
		is.updateMetrics(startTime, false)
	}
}

// handleBatchEvents processes multiple events in a batch
func (is *IngestionService) handleBatchEvents(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	
	var events []Event
	if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if len(events) == 0 {
		http.Error(w, "Empty batch", http.StatusBadRequest)
		return
	}
	
	if len(events) > is.config.BatchSize {
		http.Error(w, fmt.Sprintf("Batch size too large (max: %d)", is.config.BatchSize), http.StatusBadRequest)
		return
	}
	
	validEvents := make([]Event, 0, len(events))
	failedEvents := make([]map[string]interface{}, 0)
	
	// Validate and enrich each event
	for i, event := range events {
		if is.config.ValidationEnabled {
			if err := is.validateEvent(&event); err != nil {
				failedEvents = append(failedEvents, map[string]interface{}{
					"index": i,
					"error": err.Error(),
					"event": event,
				})
				continue
			}
		}
		
		is.enrichEvent(&event)
		validEvents = append(validEvents, event)
	}
	
	// Queue valid events
	if len(validEvents) > 0 {
		select {
		case is.batchQueue <- validEvents:
			// Update metrics for successful events
			for range validEvents {
				is.updateMetrics(startTime, true)
			}
		default:
			http.Error(w, "Batch queue full - try again later", http.StatusServiceUnavailable)
			return
		}
	}
	
	response := map[string]interface{}{
		"status":        "processed",
		"totalEvents":   len(events),
		"validEvents":   len(validEvents),
		"failedEvents":  len(failedEvents),
		"failures":      failedEvents,
		"queueSize":     len(is.eventQueue),
		"batchQueueSize": len(is.batchQueue),
		"timestamp":     time.Now().Format(time.RFC3339),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// validateEvent validates an event against configured rules
func (is *IngestionService) validateEvent(event *Event) error {
	// Basic required field validation
	if event.SourceType == "" {
		return fmt.Errorf("sourceType is required")
	}
	
	if event.EventType == "" {
		return fmt.Errorf("eventType is required")
	}
	
	if event.Timestamp.IsZero() {
		return fmt.Errorf("timestamp is required")
	}
	
	// Check if timestamp is reasonable (not too far in past/future)
	now := time.Now()
	if event.Timestamp.Before(now.Add(-24*time.Hour)) || event.Timestamp.After(now.Add(time.Hour)) {
		return fmt.Errorf("timestamp is outside acceptable range")
	}
	
	// Validate source type
	validSourceTypes := []string{"cloud", "network", "financial", "iam", "spatial"}
	if !contains(validSourceTypes, event.SourceType) {
		return fmt.Errorf("invalid sourceType: %s", event.SourceType)
	}
	
	// Validate severity
	validSeverities := []string{"low", "medium", "high", "critical"}
	if event.Severity != "" && !contains(validSeverities, event.Severity) {
		return fmt.Errorf("invalid severity: %s", event.Severity)
	}
	
	// Apply source-specific validation rules
	if rules, exists := is.validationRules[event.SourceType]; exists {
		for _, rule := range rules {
			if err := is.applyValidationRule(event, rule); err != nil {
				return err
			}
		}
	}
	
	return nil
}

// applyValidationRule applies a specific validation rule
func (is *IngestionService) applyValidationRule(event *Event, rule ValidationRule) error {
	// Extract field value from event
	var fieldValue interface{}
	
	switch rule.Field {
	case "userId":
		fieldValue = event.UserID
	case "ipAddress":
		fieldValue = event.IPAddress
	case "severity":
		fieldValue = event.Severity
	default:
		// Check in raw data
		if event.RawData != nil {
			fieldValue = event.RawData[rule.Field]
		}
	}
	
	// Check required fields
	if rule.Required && (fieldValue == nil || fieldValue == "") {
		return fmt.Errorf("required field '%s' is missing", rule.Field)
	}
	
	// Skip further validation if field is empty and not required
	if fieldValue == nil || fieldValue == "" {
		return nil
	}
	
	// Type validation
	switch rule.Type {
	case "string":
		strVal, ok := fieldValue.(string)
		if !ok {
			return fmt.Errorf("field '%s' must be a string", rule.Field)
		}
		
		// Length validation
		if rule.MinLen > 0 && len(strVal) < rule.MinLen {
			return fmt.Errorf("field '%s' must be at least %d characters", rule.Field, rule.MinLen)
		}
		if rule.MaxLen > 0 && len(strVal) > rule.MaxLen {
			return fmt.Errorf("field '%s' must be at most %d characters", rule.Field, rule.MaxLen)
		}
		
		// Value validation
		if len(rule.Values) > 0 && !contains(rule.Values, strVal) {
			return fmt.Errorf("field '%s' has invalid value: %s", rule.Field, strVal)
		}
		
	case "number":
		_, ok := fieldValue.(float64)
		if !ok {
			return fmt.Errorf("field '%s' must be a number", rule.Field)
		}
	}
	
	return nil
}

// enrichEvent adds additional metadata to events
func (is *IngestionService) enrichEvent(event *Event) {
	// Generate processing ID if not present
	if event.ProcessingID == "" {
		event.ProcessingID = generateProcessingID()
	}
	
	// Set default timestamp if missing
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	
	// Set default severity if missing
	if event.Severity == "" {
		event.Severity = "low"
	}
	
	// Add processing metadata
	if event.RawData == nil {
		event.RawData = make(map[string]interface{})
	}
	
	event.RawData["ingestion_timestamp"] = time.Now().Format(time.RFC3339)
	event.RawData["ingestion_service"] = "finsecure-ingestion"
	event.RawData["processing_id"] = event.ProcessingID
	
	// IP geolocation enrichment (simplified)
	if event.IPAddress != "" && event.Location == nil {
		event.Location = is.geolocateIP(event.IPAddress)
	}
	
	// Source-specific enrichment
	switch event.SourceType {
	case "financial":
		is.enrichFinancialEvent(event)
	case "network":
		is.enrichNetworkEvent(event)
	case "iam":
		is.enrichIAMEvent(event)
	}
}

// enrichFinancialEvent adds financial-specific metadata
func (is *IngestionService) enrichFinancialEvent(event *Event) {
	if amount, ok := event.RawData["amount"].(float64); ok {
		if amount > 10000 {
			event.RawData["high_value_flag"] = true
		}
		
		// Round amount detection
		if amount > 0 && int(amount)%100 == 0 {
			event.RawData["round_amount_flag"] = true
		}
	}
	
	// Currency validation and normalization
	if currency, ok := event.RawData["currency"].(string); ok {
		event.RawData["currency"] = normalizeCurrency(currency)
	}
}

// enrichNetworkEvent adds network-specific metadata
func (is *IngestionService) enrichNetworkEvent(event *Event) {
	// Protocol normalization
	if protocol, ok := event.RawData["protocol"].(string); ok {
		event.RawData["protocol"] = normalizeProtocol(protocol)
	}
	
	// Port analysis
	if port, ok := event.RawData["destination_port"].(float64); ok {
		portInt := int(port)
		event.RawData["port_category"] = categorizePort(portInt)
		
		if isSuspiciousPort(portInt) {
			event.RawData["suspicious_port_flag"] = true
		}
	}
	
	// Traffic volume analysis
	if bytes, ok := event.RawData["bytes_transferred"].(float64); ok {
		if bytes > 100*1024*1024 { // > 100MB
			event.RawData["large_transfer_flag"] = true
		}
	}
}

// enrichIAMEvent adds IAM-specific metadata
func (is *IngestionService) enrichIAMEvent(event *Event) {
	// Login attempt analysis
	if event.EventType == "login_failed" {
		event.RawData["security_event_flag"] = true
	}
	
	// Privilege escalation detection
	if privilegeLevel, ok := event.RawData["privilege_level"].(string); ok {
		if privilegeLevel == "admin" || privilegeLevel == "root" {
			event.RawData["elevated_privilege_flag"] = true
		}
	}
	
	// User agent analysis
	if userAgent, ok := event.RawData["user_agent"].(string); ok {
		if isBot(userAgent) {
			event.RawData["bot_flag"] = true
		}
	}
}

// geolocateIP performs basic IP geolocation (simplified)
func (is *IngestionService) geolocateIP(ip string) *Location {
	// This is a simplified implementation
	// In production, you would use a real geolocation service
	
	if isPrivateIP(ip) {
		return &Location{
			Latitude:  40.7128,
			Longitude: -74.0060,
			Country:   "US",
			City:      "Internal Network",
		}
	}
	
	// Default to unknown location
	return &Location{
		Latitude:  0.0,
		Longitude: 0.0,
		Country:   "Unknown",
		City:      "Unknown",
	}
}

// eventWorker processes events from the queue
func (is *IngestionService) eventWorker(workerID int) {
	defer is.wg.Done()
	
	log.Printf("Event worker %d started", workerID)
	
	for {
		select {
		case event := <-is.eventQueue:
			is.processEvent(event, workerID)
			
		case <-is.ctx.Done():
			log.Printf("Event worker %d shutting down", workerID)
			return
		}
	}
}

// batchProcessor handles batch processing of events
func (is *IngestionService) batchProcessor() {
	defer is.wg.Done()
	
	log.Println("Batch processor started")
	
	ticker := time.NewTicker(is.config.BatchTimeout)
	defer ticker.Stop()
	
	var currentBatch []Event
	
	for {
		select {
		case events := <-is.batchQueue:
			currentBatch = append(currentBatch, events...)
			
			if len(currentBatch) >= is.config.BatchSize {
				is.processBatch(currentBatch)
				currentBatch = nil
			}
			
		case <-ticker.C:
			if len(currentBatch) > 0 {
				is.processBatch(currentBatch)
				currentBatch = nil
			}
			
		case <-is.ctx.Done():
			if len(currentBatch) > 0 {
				is.processBatch(currentBatch)
			}
			log.Println("Batch processor shutting down")
			return
		}
	}
}

// processEvent handles individual event processing
func (is *IngestionService) processEvent(event Event, workerID int) {
	// In a real implementation, this would:
	// 1. Send to stream processor (Kafka/Kinesis)
	// 2. Store in database
	// 3. Trigger ML feature extraction
	// 4. Send to alerting system if needed
	
	log.Printf("Worker %d processing event: %s/%s (ID: %s)", 
		workerID, event.SourceType, event.EventType, event.ProcessingID)
	
	// Simulate processing time
	time.Sleep(time.Millisecond * 1)
	
	// Here you would integrate with:
	// - Kafka producer
	// - Database client
	// - ML service client
	// - Alert service client
}

// processBatch handles batch event processing
func (is *IngestionService) processBatch(events []Event) {
	log.Printf("Processing batch of %d events", len(events))
	
	// In a real implementation, this would send the entire batch
	// to downstream services for more efficient processing
	
	for _, event := range events {
		is.processEvent(event, -1) // -1 indicates batch processing
	}
}

// metricsCollector periodically collects and logs metrics
func (is *IngestionService) metricsCollector() {
	defer is.wg.Done()
	
	ticker := time.NewTicker(is.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			is.logMetrics()
			
		case <-is.ctx.Done():
			return
		}
	}
}

// updateMetrics updates ingestion metrics
func (is *IngestionService) updateMetrics(startTime time.Time, success bool) {
	is.metrics.mutex.Lock()
	defer is.metrics.mutex.Unlock()
	
	is.metrics.TotalEvents++
	
	if success {
		is.metrics.ProcessedEvents++
	} else {
		is.metrics.FailedEvents++
	}
	
	// Update average latency
	latency := time.Since(startTime)
	if is.metrics.ProcessedEvents == 1 {
		is.metrics.AvgLatency = latency
	} else {
		// Exponential moving average
		is.metrics.AvgLatency = time.Duration(
			0.9*float64(is.metrics.AvgLatency) + 0.1*float64(latency),
		)
	}
	
	// Update throughput (events per second)
	duration := time.Since(is.metrics.LastReset).Seconds()
	if duration > 0 {
		is.metrics.Throughput = float64(is.metrics.ProcessedEvents) / duration
	}
}

// logMetrics logs current metrics
func (is *IngestionService) logMetrics() {
	is.metrics.mutex.RLock()
	defer is.metrics.mutex.RUnlock()
	
	log.Printf("Metrics - Total: %d, Processed: %d, Failed: %d, Avg Latency: %v, Throughput: %.2f/s",
		is.metrics.TotalEvents,
		is.metrics.ProcessedEvents,
		is.metrics.FailedEvents,
		is.metrics.AvgLatency,
		is.metrics.Throughput,
	)
}

// HTTP handlers

func (is *IngestionService) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":      "healthy",
		"service":     "finsecure-ingestion",
		"version":     "1.0.0",
		"timestamp":   time.Now().Format(time.RFC3339),
		"queue_size":  len(is.eventQueue),
		"batch_queue_size": len(is.batchQueue),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (is *IngestionService) handleMetrics(w http.ResponseWriter, r *http.Request) {
	is.metrics.mutex.RLock()
	metrics := *is.metrics
	is.metrics.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (is *IngestionService) handleStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"service":           "finsecure-ingestion",
		"status":            "running",
		"queue_capacity":    cap(is.eventQueue),
		"queue_size":        len(is.eventQueue),
		"batch_queue_size":  len(is.batchQueue),
		"worker_count":      is.config.WorkerCount,
		"validation_enabled": is.config.ValidationEnabled,
		"uptime":            time.Since(is.metrics.LastReset).String(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (is *IngestionService) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(is.config)
}

func (is *IngestionService) handleGetValidationRules(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(is.validationRules)
}

func (is *IngestionService) handleQueueStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"event_queue": map[string]interface{}{
			"size":     len(is.eventQueue),
			"capacity": cap(is.eventQueue),
			"usage":    float64(len(is.eventQueue)) / float64(cap(is.eventQueue)) * 100,
		},
		"batch_queue": map[string]interface{}{
			"size":     len(is.batchQueue),
			"capacity": cap(is.batchQueue),
			"usage":    float64(len(is.batchQueue)) / float64(cap(is.batchQueue)) * 100,
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Utility functions

func initValidationRules() map[string][]ValidationRule {
	return map[string][]ValidationRule{
		"financial": {
			{Field: "amount", Required: true, Type: "number"},
			{Field: "currency", Required: true, Type: "string", Values: []string{"USD", "EUR", "GBP", "JPY"}},
			{Field: "transaction_type", Required: true, Type: "string"},
		},
		"network": {
			{Field: "protocol", Required: true, Type: "string"},
			{Field: "source_port", Required: false, Type: "number"},
			{Field: "destination_port", Required: false, Type: "number"},
		},
		"iam": {
			{Field: "action", Required: true, Type: "string"},
			{Field: "user_agent", Required: false, Type: "string", MaxLen: 500},
		},
	}
}

func generateProcessingID() string {
	return fmt.Sprintf("proc_%d_%d", time.Now().UnixNano(), os.Getpid())
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func normalizeCurrency(currency string) string {
	// Normalize currency codes to uppercase
	return strings.ToUpper(strings.TrimSpace(currency))
}

func normalizeProtocol(protocol string) string {
	return strings.ToLower(strings.TrimSpace(protocol))
}

func categorizePort(port int) string {
	if port < 1024 {
		return "well_known"
	} else if port < 49152 {
		return "registered"
	} else {
		return "dynamic"
	}
}

func isSuspiciousPort(port int) bool {
	suspiciousPorts := []int{1433, 3389, 22, 23, 135, 139, 445, 3306, 5432}
	for _, p := range suspiciousPorts {
		if port == p {
			return true
		}
	}
	return false
}

func isBot(userAgent string) bool {
	botIndicators := []string{"bot", "crawler", "spider", "scraper"}
	userAgentLower := strings.ToLower(userAgent)
	
	for _, indicator := range botIndicators {
		if strings.Contains(userAgentLower, indicator) {
			return true
		}
	}
	return false
}

func isPrivateIP(ip string) bool {
	privateRanges := []string{"192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "172.3", "127."}
	
	for _, prefix := range privateRanges {
		if strings.HasPrefix(ip, prefix) {
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
	
	// Create and start service
	service := NewIngestionService(config)
	
	if err := service.Start(); err != nil {
		log.Fatalf("Service failed to start: %v", err)
	}
}
