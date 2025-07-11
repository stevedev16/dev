import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CNNAnomalyDetector:
    """
    CNN-based anomaly detection model for network traffic and spatial patterns.
    Simulates deep learning model behavior for network security analysis.
    """
    
    def __init__(self):
        self.name = "CNN Network Anomaly Detector"
        self.model_type = "cnn"
        self.version = "1.0.0"
        self.is_loaded = False
        self.threshold = 0.7
        self.model_weights = None
        self.feature_importance = {}
        self.training_history = []
        
        # Model architecture parameters
        self.input_shape = (32, 32, 3)  # Simulated input shape for network patterns
        self.conv_layers = [64, 128, 256]
        self.dense_layers = [512, 256, 1]
        
        # Performance metrics
        self.metrics = {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.94,
            'f1_score': 0.91,
            'inference_time_ms': 0.0
        }
        
        # Feature mappings for network analysis
        self.network_features = [
            'packet_size', 'protocol_type', 'connection_duration',
            'bytes_transferred', 'port_number', 'flags_set',
            'packet_frequency', 'bandwidth_usage', 'connection_count',
            'tcp_flags', 'ip_fragmentation', 'payload_entropy'
        ]
        
        # Threat patterns
        self.threat_patterns = {
            'ddos_attack': {'weight': 0.9, 'indicators': ['high_packet_frequency', 'bandwidth_spike']},
            'port_scan': {'weight': 0.8, 'indicators': ['multiple_ports', 'connection_attempts']},
            'malware_traffic': {'weight': 0.85, 'indicators': ['unusual_payload', 'encrypted_traffic']},
            'data_exfiltration': {'weight': 0.88, 'indicators': ['large_transfers', 'unusual_destinations']},
            'brute_force': {'weight': 0.75, 'indicators': ['repeated_connections', 'auth_failures']}
        }

    async def initialize(self):
        """Initialize the CNN model"""
        try:
            logger.info("Initializing CNN anomaly detection model...")
            
            # Simulate model loading
            await asyncio.sleep(0.1)
            
            # Initialize model weights (simulated)
            self.model_weights = self._initialize_weights()
            
            # Set feature importance
            self.feature_importance = self._calculate_feature_importance()
            
            self.is_loaded = True
            logger.info("✅ CNN model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CNN model: {e}")
            raise

    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize model weights (simulated)"""
        weights = {}
        
        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            weights[f'conv_{i}_weights'] = np.random.randn(3, 3, 3 if i == 0 else self.conv_layers[i-1], filters) * 0.1
            weights[f'conv_{i}_bias'] = np.zeros(filters)
        
        # Dense layers
        for i, units in enumerate(self.dense_layers):
            input_size = 256 if i == 0 else self.dense_layers[i-1]
            weights[f'dense_{i}_weights'] = np.random.randn(input_size, units) * 0.1
            weights[f'dense_{i}_bias'] = np.zeros(units)
        
        return weights

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance for interpretability"""
        importance = {}
        
        # Network-specific feature importance
        base_importance = {
            'packet_size': 0.15,
            'protocol_type': 0.12,
            'connection_duration': 0.10,
            'bytes_transferred': 0.18,
            'port_number': 0.08,
            'flags_set': 0.07,
            'packet_frequency': 0.14,
            'bandwidth_usage': 0.16
        }
        
        # Add some randomness to simulate model learning
        for feature, base_score in base_importance.items():
            importance[feature] = base_score + np.random.normal(0, 0.02)
        
        # Normalize
        total = sum(importance.values())
        for feature in importance:
            importance[feature] /= total
        
        return importance

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make anomaly prediction using CNN model"""
        if not self.is_loaded:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Extract network features
            network_features = self._extract_network_features(features)
            
            # Simulate CNN forward pass
            anomaly_score = await self._cnn_forward_pass(network_features)
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self.threshold
            
            # Calculate confidence
            confidence = min(anomaly_score + 0.1, 1.0) if is_anomaly else max(1.0 - anomaly_score, 0.1)
            
            # Determine threat type
            threat_type = self._identify_threat_type(network_features, anomaly_score)
            
            # Generate explanations
            explanations = self._generate_explanations(network_features, anomaly_score, threat_type)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(anomaly_score)
            
            # Record inference time
            inference_time = (time.time() - start_time) * 1000
            self.metrics['inference_time_ms'] = inference_time
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'anomaly_score': anomaly_score,
                'model_type': self.model_type,
                'threat_type': threat_type,
                'risk_level': risk_level,
                'explanation': explanations
            }
            
        except Exception as e:
            logger.error(f"CNN prediction error: {e}")
            raise

    def _extract_network_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize network features"""
        network_features = {}
        
        # Extract relevant features for network analysis
        raw_data = features.get('raw_data', {})
        
        # Packet size analysis
        packet_size = raw_data.get('packet_size', features.get('packetSize', 0))
        network_features['packet_size'] = self._normalize_packet_size(packet_size)
        
        # Protocol type
        protocol = raw_data.get('protocol', features.get('protocolType', 'tcp'))
        network_features['protocol_type'] = self._encode_protocol(protocol)
        
        # Connection duration
        duration = raw_data.get('connection_duration', features.get('connectionDuration', 0))
        network_features['connection_duration'] = self._normalize_duration(duration)
        
        # Bytes transferred
        bytes_transferred = raw_data.get('bytes_transferred', features.get('bytesTransferred', 0))
        network_features['bytes_transferred'] = self._normalize_bytes(bytes_transferred)
        
        # Port number
        port = raw_data.get('port', features.get('portNumber', 80))
        network_features['port_number'] = self._normalize_port(port)
        
        # Flags
        flags = raw_data.get('flags', features.get('flagsSet', []))
        network_features['flags_set'] = self._encode_flags(flags)
        
        # Derived features
        network_features['packet_frequency'] = self._calculate_packet_frequency(features)
        network_features['bandwidth_usage'] = self._calculate_bandwidth_usage(features)
        
        return network_features

    async def _cnn_forward_pass(self, features: Dict[str, float]) -> float:
        """Simulate CNN forward pass"""
        # Simulate computation time
        await asyncio.sleep(0.001)
        
        # Convert features to tensor-like structure
        feature_vector = np.array(list(features.values()))
        
        # Simulate convolution operations
        conv_output = self._simulate_convolution(feature_vector)
        
        # Simulate pooling
        pooled_output = self._simulate_pooling(conv_output)
        
        # Simulate dense layers
        dense_output = self._simulate_dense_layers(pooled_output)
        
        # Apply sigmoid activation for anomaly score
        anomaly_score = 1 / (1 + np.exp(-dense_output))
        
        return float(anomaly_score)

    def _simulate_convolution(self, input_vector: np.ndarray) -> np.ndarray:
        """Simulate convolution operation"""
        # Pad input to simulate 2D convolution
        padded_input = np.pad(input_vector, (1, 1), mode='constant')
        
        # Simulate convolution with learned patterns
        conv_result = []
        for i in range(len(input_vector)):
            # Simulate convolution kernel
            kernel_response = np.sum(padded_input[i:i+3] * np.array([0.1, 0.5, 0.1]))
            conv_result.append(max(0, kernel_response))  # ReLU activation
        
        return np.array(conv_result)

    def _simulate_pooling(self, conv_output: np.ndarray) -> np.ndarray:
        """Simulate max pooling operation"""
        pooled_size = len(conv_output) // 2
        pooled_output = []
        
        for i in range(0, len(conv_output), 2):
            if i + 1 < len(conv_output):
                pooled_output.append(max(conv_output[i], conv_output[i+1]))
            else:
                pooled_output.append(conv_output[i])
        
        return np.array(pooled_output)

    def _simulate_dense_layers(self, input_vector: np.ndarray) -> float:
        """Simulate dense layer operations"""
        current_input = input_vector
        
        # Simulate multiple dense layers
        for i in range(3):  # 3 dense layers
            # Simulate weight multiplication
            weights = np.random.randn(len(current_input)) * 0.1
            bias = np.random.randn() * 0.01
            
            output = np.dot(current_input, weights) + bias
            
            # Apply activation (ReLU for hidden layers, linear for output)
            if i < 2:
                output = max(0, output)
                current_input = np.array([output])
            else:
                return output
        
        return 0.0

    def _identify_threat_type(self, features: Dict[str, float], anomaly_score: float) -> Optional[str]:
        """Identify specific threat type based on features"""
        if anomaly_score < 0.5:
            return None
        
        # Check for threat patterns
        best_match = None
        best_score = 0
        
        for threat_type, pattern in self.threat_patterns.items():
            score = self._calculate_threat_score(features, pattern)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = threat_type
        
        return best_match

    def _calculate_threat_score(self, features: Dict[str, float], pattern: Dict[str, Any]) -> float:
        """Calculate threat pattern match score"""
        indicators = pattern['indicators']
        weight = pattern['weight']
        
        # Simplified threat detection logic
        matches = 0
        total_indicators = len(indicators)
        
        # Check for high packet frequency (DDoS indicator)
        if 'high_packet_frequency' in indicators and features.get('packet_frequency', 0) > 0.8:
            matches += 1
        
        # Check for multiple ports (port scan indicator)
        if 'multiple_ports' in indicators and features.get('port_number', 0) > 0.9:
            matches += 1
        
        # Check for large transfers (data exfiltration)
        if 'large_transfers' in indicators and features.get('bytes_transferred', 0) > 0.8:
            matches += 1
        
        # Check for unusual payload (malware)
        if 'unusual_payload' in indicators and features.get('flags_set', 0) > 0.7:
            matches += 1
        
        # Check for repeated connections (brute force)
        if 'repeated_connections' in indicators and features.get('connection_duration', 0) < 0.3:
            matches += 1
        
        match_ratio = matches / total_indicators if total_indicators > 0 else 0
        return match_ratio * weight

    def _generate_explanations(self, features: Dict[str, float], anomaly_score: float, threat_type: Optional[str]) -> List[str]:
        """Generate human-readable explanations for the prediction"""
        explanations = []
        
        # Feature-based explanations
        if features.get('packet_size', 0) > 0.8:
            explanations.append("Unusually large packet size detected")
        
        if features.get('bytes_transferred', 0) > 0.8:
            explanations.append("High data transfer volume detected")
        
        if features.get('packet_frequency', 0) > 0.8:
            explanations.append("High packet frequency indicating potential attack")
        
        if features.get('port_number', 0) > 0.9:
            explanations.append("Access to suspicious or uncommon ports")
        
        if features.get('connection_duration', 0) < 0.2:
            explanations.append("Very short connection duration - possible scanning")
        
        # Threat-specific explanations
        if threat_type:
            threat_explanations = {
                'ddos_attack': "Pattern consistent with DDoS attack",
                'port_scan': "Pattern consistent with port scanning activity",
                'malware_traffic': "Pattern consistent with malware communication",
                'data_exfiltration': "Pattern consistent with data exfiltration",
                'brute_force': "Pattern consistent with brute force attack"
            }
            explanations.append(threat_explanations.get(threat_type, f"Threat type: {threat_type}"))
        
        # Score-based explanations
        if anomaly_score > 0.9:
            explanations.append("Very high anomaly score - immediate attention required")
        elif anomaly_score > 0.8:
            explanations.append("High anomaly score - investigation recommended")
        
        return explanations

    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """Calculate risk level based on anomaly score"""
        if anomaly_score >= 0.9:
            return 'critical'
        elif anomaly_score >= 0.8:
            return 'high'
        elif anomaly_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _normalize_packet_size(self, size: int) -> float:
        """Normalize packet size to 0-1 range"""
        # Typical packet sizes: 64-1500 bytes
        return min(max(size, 0), 1500) / 1500

    def _encode_protocol(self, protocol: str) -> float:
        """Encode protocol type as numeric value"""
        protocol_map = {
            'tcp': 0.1,
            'udp': 0.2,
            'icmp': 0.3,
            'http': 0.4,
            'https': 0.5,
            'ftp': 0.6,
            'ssh': 0.7,
            'dns': 0.8,
            'unknown': 0.9
        }
        return protocol_map.get(protocol.lower(), 0.9)

    def _normalize_duration(self, duration: int) -> float:
        """Normalize connection duration"""
        # Typical connection durations: 0-3600 seconds
        return min(max(duration, 0), 3600) / 3600

    def _normalize_bytes(self, bytes_count: int) -> float:
        """Normalize bytes transferred"""
        # Typical transfer sizes: 0-1GB
        return min(max(bytes_count, 0), 1073741824) / 1073741824

    def _normalize_port(self, port: int) -> float:
        """Normalize port number"""
        # Check if port is in suspicious range
        suspicious_ports = [1433, 3389, 22, 23, 135, 139, 445, 3306, 5432]
        if port in suspicious_ports:
            return 0.9
        elif port < 1024:  # Well-known ports
            return 0.3
        elif port < 49152:  # Registered ports
            return 0.5
        else:  # Dynamic ports
            return 0.7

    def _encode_flags(self, flags: List[str]) -> float:
        """Encode TCP flags as numeric value"""
        if not flags:
            return 0.0
        
        suspicious_flags = ['URG', 'PSH', 'RST', 'SYN', 'FIN']
        suspicious_count = sum(1 for flag in flags if flag in suspicious_flags)
        
        return min(suspicious_count / len(suspicious_flags), 1.0)

    def _calculate_packet_frequency(self, features: Dict[str, Any]) -> float:
        """Calculate packet frequency metric"""
        # Simulate packet frequency calculation
        raw_data = features.get('raw_data', {})
        packet_count = raw_data.get('packet_count', 1)
        duration = raw_data.get('connection_duration', 1)
        
        frequency = packet_count / max(duration, 1)
        return min(frequency / 100, 1.0)  # Normalize to 0-1

    def _calculate_bandwidth_usage(self, features: Dict[str, Any]) -> float:
        """Calculate bandwidth usage metric"""
        # Simulate bandwidth calculation
        raw_data = features.get('raw_data', {})
        bytes_transferred = raw_data.get('bytes_transferred', 0)
        duration = raw_data.get('connection_duration', 1)
        
        bandwidth = bytes_transferred / max(duration, 1)
        return min(bandwidth / 1000000, 1.0)  # Normalize to 0-1 (1MB/s max)

    async def retrain(self, training_data: Dict[str, Any]):
        """Retrain the model with new data"""
        logger.info("Starting CNN model retraining...")
        
        # Simulate retraining process
        await asyncio.sleep(0.5)
        
        # Update model weights (simulated)
        self.model_weights = self._initialize_weights()
        
        # Update metrics (simulated improvement)
        self.metrics['accuracy'] = min(self.metrics['accuracy'] + 0.01, 0.99)
        self.metrics['precision'] = min(self.metrics['precision'] + 0.01, 0.99)
        self.metrics['recall'] = min(self.metrics['recall'] + 0.01, 0.99)
        self.metrics['f1_score'] = min(self.metrics['f1_score'] + 0.01, 0.99)
        
        logger.info("CNN model retraining completed")

    async def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.metrics.copy()

    def get_input_features(self) -> List[str]:
        """Get list of input features"""
        return self.network_features.copy()

    def get_output_classes(self) -> List[str]:
        """Get list of output classes"""
        return ['normal', 'anomaly']

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
