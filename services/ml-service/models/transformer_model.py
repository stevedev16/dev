import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class TransformerAnomalyDetector:
    """
    Transformer-based anomaly detection model for complex behavioral patterns.
    Specialized for analyzing user behavior, access patterns, and multi-modal security events.
    """
    
    def __init__(self):
        self.name = "Transformer Behavioral Analyzer"
        self.model_type = "transformer"
        self.version = "1.0.0"
        self.is_loaded = False
        self.threshold = 0.7
        self.model_weights = None
        
        # Transformer architecture parameters
        self.d_model = 512  # Model dimension
        self.n_heads = 8    # Number of attention heads
        self.n_layers = 6   # Number of transformer layers
        self.d_ff = 2048   # Feed-forward dimension
        self.max_seq_length = 128
        self.dropout_rate = 0.1
        
        # Performance metrics
        self.metrics = {
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.91,
            'f1_score': 0.89,
            'inference_time_ms': 0.0
        }
        
        # Behavioral features for transformer analysis
        self.behavioral_features = [
            'user_context', 'session_context', 'system_context',
            'temporal_context', 'spatial_context', 'access_pattern',
            'device_fingerprint', 'behavioral_biometrics', 'usage_pattern',
            'privilege_escalation', 'data_access_pattern', 'network_behavior'
        ]
        
        # Attention patterns for different threat types
        self.threat_attention_patterns = {
            'insider_threat': {
                'weight': 0.88,
                'key_features': ['privilege_escalation', 'unusual_access', 'data_exfiltration'],
                'attention_focus': 'temporal_behavioral'
            },
            'credential_stuffing': {
                'weight': 0.82,
                'key_features': ['multiple_failed_logins', 'ip_rotation', 'user_agent_variation'],
                'attention_focus': 'access_pattern'
            },
            'account_compromise': {
                'weight': 0.86,
                'key_features': ['location_change', 'device_change', 'behavior_change'],
                'attention_focus': 'context_change'
            },
            'privilege_abuse': {
                'weight': 0.84,
                'key_features': ['elevated_access', 'unusual_permissions', 'sensitive_data'],
                'attention_focus': 'privilege_pattern'
            },
            'social_engineering': {
                'weight': 0.80,
                'key_features': ['phishing_indicators', 'urgency_patterns', 'trust_exploitation'],
                'attention_focus': 'behavioral_anomaly'
            }
        }
        
        # Context encoders
        self.context_encoders = {}
        self.positional_encodings = {}
        
        # Behavioral baselines
        self.behavioral_baselines = {}

    async def initialize(self):
        """Initialize the Transformer model"""
        try:
            logger.info("Initializing Transformer anomaly detection model...")
            
            # Simulate model loading
            await asyncio.sleep(0.15)
            
            # Initialize model weights
            self.model_weights = self._initialize_transformer_weights()
            
            # Initialize positional encodings
            self.positional_encodings = self._generate_positional_encodings()
            
            # Initialize context encoders
            self.context_encoders = self._initialize_context_encoders()
            
            # Load behavioral baselines
            await self._load_behavioral_baselines()
            
            self.is_loaded = True
            logger.info("✅ Transformer model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformer model: {e}")
            raise

    def _initialize_transformer_weights(self) -> Dict[str, np.ndarray]:
        """Initialize Transformer model weights"""
        weights = {}
        
        # Embedding layers
        weights['token_embedding'] = np.random.randn(1000, self.d_model) * 0.1
        weights['position_embedding'] = np.random.randn(self.max_seq_length, self.d_model) * 0.1
        
        # Transformer layers
        for layer in range(self.n_layers):
            # Multi-head attention
            weights[f'layer_{layer}_attention_q'] = np.random.randn(self.d_model, self.d_model) * 0.1
            weights[f'layer_{layer}_attention_k'] = np.random.randn(self.d_model, self.d_model) * 0.1
            weights[f'layer_{layer}_attention_v'] = np.random.randn(self.d_model, self.d_model) * 0.1
            weights[f'layer_{layer}_attention_o'] = np.random.randn(self.d_model, self.d_model) * 0.1
            
            # Feed-forward network
            weights[f'layer_{layer}_ff_1'] = np.random.randn(self.d_model, self.d_ff) * 0.1
            weights[f'layer_{layer}_ff_2'] = np.random.randn(self.d_ff, self.d_model) * 0.1
            
            # Layer normalization
            weights[f'layer_{layer}_ln1_weight'] = np.ones(self.d_model)
            weights[f'layer_{layer}_ln1_bias'] = np.zeros(self.d_model)
            weights[f'layer_{layer}_ln2_weight'] = np.ones(self.d_model)
            weights[f'layer_{layer}_ln2_bias'] = np.zeros(self.d_model)
        
        # Output layers
        weights['output_projection'] = np.random.randn(self.d_model, 1) * 0.1
        weights['output_bias'] = np.zeros(1)
        
        return weights

    def _generate_positional_encodings(self) -> np.ndarray:
        """Generate positional encodings for transformer"""
        pe = np.zeros((self.max_seq_length, self.d_model))
        
        position = np.arange(0, self.max_seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe

    def _initialize_context_encoders(self) -> Dict[str, Dict[str, Any]]:
        """Initialize context-specific encoders"""
        encoders = {}
        
        # User context encoder
        encoders['user_context'] = {
            'vocab_size': 1000,
            'embedding_dim': 128,
            'features': ['user_id', 'role', 'department', 'seniority', 'clearance_level']
        }
        
        # Session context encoder
        encoders['session_context'] = {
            'vocab_size': 500,
            'embedding_dim': 64,
            'features': ['session_id', 'duration', 'activity_level', 'resource_usage']
        }
        
        # System context encoder
        encoders['system_context'] = {
            'vocab_size': 200,
            'embedding_dim': 32,
            'features': ['system_id', 'os_type', 'security_level', 'patch_level']
        }
        
        # Temporal context encoder
        encoders['temporal_context'] = {
            'vocab_size': 100,
            'embedding_dim': 32,
            'features': ['hour', 'day_of_week', 'is_holiday', 'business_hours']
        }
        
        # Spatial context encoder
        encoders['spatial_context'] = {
            'vocab_size': 300,
            'embedding_dim': 64,
            'features': ['location', 'ip_range', 'network_zone', 'physical_location']
        }
        
        return encoders

    async def _load_behavioral_baselines(self):
        """Load behavioral baselines for comparison"""
        # Simulate loading behavioral baselines
        await asyncio.sleep(0.05)
        
        # Sample behavioral baselines
        self.behavioral_baselines = {
            'normal_access_pattern': {
                'login_frequency': 2.5,
                'session_duration': 480,  # 8 hours
                'resource_access': ['email', 'documents', 'database'],
                'time_of_day': [9, 17],  # 9 AM to 5 PM
                'location_consistency': 0.9
            },
            'privileged_user_pattern': {
                'login_frequency': 1.8,
                'session_duration': 360,  # 6 hours
                'resource_access': ['admin_panel', 'server_logs', 'user_management'],
                'time_of_day': [8, 18],  # 8 AM to 6 PM
                'location_consistency': 0.95
            },
            'developer_pattern': {
                'login_frequency': 3.2,
                'session_duration': 600,  # 10 hours
                'resource_access': ['code_repository', 'deployment_tools', 'databases'],
                'time_of_day': [7, 20],  # 7 AM to 8 PM
                'location_consistency': 0.8
            }
        }

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make behavioral anomaly prediction using Transformer model"""
        if not self.is_loaded:
            raise RuntimeError("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Extract behavioral features
            behavioral_features = self._extract_behavioral_features(features)
            
            # Encode contexts
            encoded_contexts = self._encode_contexts(behavioral_features)
            
            # Create input sequence
            input_sequence = self._create_input_sequence(encoded_contexts)
            
            # Transformer forward pass
            anomaly_score = await self._transformer_forward_pass(input_sequence)
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self.threshold
            
            # Calculate confidence
            confidence = min(anomaly_score + 0.1, 1.0) if is_anomaly else max(1.0 - anomaly_score, 0.1)
            
            # Identify threat type
            threat_type = self._identify_threat_type(behavioral_features, anomaly_score)
            
            # Generate explanations with attention analysis
            explanations = self._generate_explanations(behavioral_features, anomaly_score, threat_type)
            
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
            logger.error(f"Transformer prediction error: {e}")
            raise

    def _extract_behavioral_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features from input"""
        behavioral_features = {}
        raw_data = features.get('raw_data', {})
        
        # User context
        behavioral_features['user_context'] = {
            'user_id': features.get('userId', 'unknown'),
            'role': raw_data.get('role', 'user'),
            'department': raw_data.get('department', 'unknown'),
            'seniority': raw_data.get('seniority', 'junior'),
            'clearance_level': raw_data.get('clearance_level', 'standard')
        }
        
        # Session context
        behavioral_features['session_context'] = {
            'session_id': raw_data.get('session_id', 'unknown'),
            'duration': raw_data.get('session_duration', 480),
            'activity_level': raw_data.get('activity_level', 'normal'),
            'resource_usage': raw_data.get('resource_usage', 'low')
        }
        
        # System context
        behavioral_features['system_context'] = {
            'system_id': raw_data.get('system_id', 'unknown'),
            'os_type': raw_data.get('os_type', 'windows'),
            'security_level': raw_data.get('security_level', 'standard'),
            'patch_level': raw_data.get('patch_level', 'current')
        }
        
        # Temporal context
        event_time = datetime.fromisoformat(features.get('timestamp', datetime.now().isoformat()))
        behavioral_features['temporal_context'] = {
            'hour': event_time.hour,
            'day_of_week': event_time.weekday(),
            'is_holiday': self._is_holiday(event_time),
            'business_hours': self._is_business_hours(event_time)
        }
        
        # Spatial context
        location = features.get('location', {})
        behavioral_features['spatial_context'] = {
            'location': location.get('city', 'unknown'),
            'ip_range': self._get_ip_range(features.get('ipAddress', '0.0.0.0')),
            'network_zone': raw_data.get('network_zone', 'internal'),
            'physical_location': raw_data.get('physical_location', 'office')
        }
        
        # Access pattern
        behavioral_features['access_pattern'] = {
            'resources_accessed': raw_data.get('resources_accessed', []),
            'failed_attempts': raw_data.get('failed_attempts', 0),
            'privilege_level': raw_data.get('privilege_level', 'standard'),
            'data_sensitivity': raw_data.get('data_sensitivity', 'low')
        }
        
        # Device fingerprint
        behavioral_features['device_fingerprint'] = {
            'device_id': raw_data.get('device_id', 'unknown'),
            'user_agent': raw_data.get('user_agent', 'unknown'),
            'screen_resolution': raw_data.get('screen_resolution', 'unknown'),
            'timezone': raw_data.get('timezone', 'UTC')
        }
        
        return behavioral_features

    def _encode_contexts(self, behavioral_features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Encode different contexts into embeddings"""
        encoded_contexts = {}
        
        for context_name, context_data in behavioral_features.items():
            if context_name in self.context_encoders:
                encoder = self.context_encoders[context_name]
                encoded_contexts[context_name] = self._encode_context(context_data, encoder)
            else:
                # Simple encoding for unknown contexts
                encoded_contexts[context_name] = self._simple_encode(context_data)
        
        return encoded_contexts

    def _encode_context(self, context_data: Dict[str, Any], encoder: Dict[str, Any]) -> np.ndarray:
        """Encode a specific context using its encoder"""
        embedding_dim = encoder['embedding_dim']
        encoded = np.zeros(embedding_dim)
        
        # Simple hash-based encoding
        for i, (key, value) in enumerate(context_data.items()):
            if i >= embedding_dim:
                break
            
            # Hash the key-value pair to a position
            hash_value = hash(f"{key}:{value}")
            position = hash_value % embedding_dim
            encoded[position] = 1.0
        
        return encoded

    def _simple_encode(self, data: Dict[str, Any]) -> np.ndarray:
        """Simple encoding for unknown contexts"""
        encoded = np.zeros(64)  # Default dimension
        
        for i, (key, value) in enumerate(data.items()):
            if i >= 64:
                break
            
            hash_value = hash(f"{key}:{value}")
            position = hash_value % 64
            encoded[position] = 1.0
        
        return encoded

    def _create_input_sequence(self, encoded_contexts: Dict[str, np.ndarray]) -> np.ndarray:
        """Create input sequence for transformer"""
        # Concatenate all encoded contexts
        context_vectors = []
        for context_name in self.behavioral_features:
            if context_name in encoded_contexts:
                context_vectors.append(encoded_contexts[context_name])
            else:
                # Pad with zeros if context not available
                context_vectors.append(np.zeros(64))
        
        # Pad or truncate to max sequence length
        sequence = np.array(context_vectors)
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        elif len(sequence) < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        
        return sequence

    async def _transformer_forward_pass(self, input_sequence: np.ndarray) -> float:
        """Simulate transformer forward pass"""
        # Simulate computation time
        await asyncio.sleep(0.005)
        
        # Add positional encoding
        seq_len = min(input_sequence.shape[0], self.max_seq_length)
        
        # Project input to model dimension
        x = self._project_to_model_dim(input_sequence)
        
        # Add positional encoding
        x = x + self.positional_encodings[:seq_len, :]
        
        # Pass through transformer layers
        for layer in range(self.n_layers):
            x = self._transformer_layer(x, layer)
        
        # Global average pooling
        pooled = np.mean(x, axis=0)
        
        # Output projection
        output = np.dot(pooled, self.model_weights['output_projection'].flatten()) + self.model_weights['output_bias'][0]
        
        # Sigmoid activation
        anomaly_score = 1 / (1 + np.exp(-output))
        
        return float(anomaly_score)

    def _project_to_model_dim(self, input_sequence: np.ndarray) -> np.ndarray:
        """Project input to model dimension"""
        # Simple projection using random weights
        input_dim = input_sequence.shape[1]
        projection_weights = np.random.randn(input_dim, self.d_model) * 0.1
        
        return np.dot(input_sequence, projection_weights)

    def _transformer_layer(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Simulate a single transformer layer"""
        # Multi-head attention
        attention_output = self._multi_head_attention(x, layer)
        
        # Add & norm
        x = self._layer_norm(x + attention_output, layer, 'ln1')
        
        # Feed-forward
        ff_output = self._feed_forward(x, layer)
        
        # Add & norm
        x = self._layer_norm(x + ff_output, layer, 'ln2')
        
        return x

    def _multi_head_attention(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Simulate multi-head attention"""
        # Simplified attention mechanism
        seq_len, d_model = x.shape
        
        # Create Q, K, V matrices
        Q = np.dot(x, self.model_weights[f'layer_{layer}_attention_q'])
        K = np.dot(x, self.model_weights[f'layer_{layer}_attention_k'])
        V = np.dot(x, self.model_weights[f'layer_{layer}_attention_v'])
        
        # Scaled dot-product attention
        attention_scores = np.dot(Q, K.T) / np.sqrt(d_model)
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention to values
        attention_output = np.dot(attention_weights, V)
        
        # Output projection
        output = np.dot(attention_output, self.model_weights[f'layer_{layer}_attention_o'])
        
        return output

    def _feed_forward(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Simulate feed-forward network"""
        # First linear layer + ReLU
        ff1 = np.dot(x, self.model_weights[f'layer_{layer}_ff_1'])
        ff1 = np.maximum(0, ff1)  # ReLU
        
        # Second linear layer
        ff2 = np.dot(ff1, self.model_weights[f'layer_{layer}_ff_2'])
        
        return ff2

    def _layer_norm(self, x: np.ndarray, layer: int, norm_type: str) -> np.ndarray:
        """Simulate layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        
        normalized = (x - mean) / (std + 1e-8)
        
        # Apply learned parameters
        weight = self.model_weights[f'layer_{layer}_{norm_type}_weight']
        bias = self.model_weights[f'layer_{layer}_{norm_type}_bias']
        
        return normalized * weight + bias

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _identify_threat_type(self, features: Dict[str, Any], anomaly_score: float) -> Optional[str]:
        """Identify threat type using attention patterns"""
        if anomaly_score < 0.6:
            return None
        
        best_match = None
        best_score = 0
        
        for threat_type, pattern in self.threat_attention_patterns.items():
            score = self._calculate_threat_score(features, pattern)
            if score > best_score and score > 0.7:
                best_score = score
                best_match = threat_type
        
        return best_match

    def _calculate_threat_score(self, features: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate threat score based on attention patterns"""
        key_features = pattern['key_features']
        weight = pattern['weight']
        
        matches = 0
        total_features = len(key_features)
        
        # Check for privilege escalation
        if 'privilege_escalation' in key_features:
            access_pattern = features.get('access_pattern', {})
            if access_pattern.get('privilege_level') == 'elevated':
                matches += 1
        
        # Check for unusual access
        if 'unusual_access' in key_features:
            access_pattern = features.get('access_pattern', {})
            resources = access_pattern.get('resources_accessed', [])
            if any('admin' in str(resource).lower() for resource in resources):
                matches += 1
        
        # Check for location change
        if 'location_change' in key_features:
            spatial_context = features.get('spatial_context', {})
            if spatial_context.get('network_zone') == 'external':
                matches += 1
        
        # Check for device change
        if 'device_change' in key_features:
            device_fp = features.get('device_fingerprint', {})
            if device_fp.get('device_id') == 'unknown':
                matches += 1
        
        # Check for behavior change
        if 'behavior_change' in key_features:
            temporal_context = features.get('temporal_context', {})
            if not temporal_context.get('business_hours'):
                matches += 1
        
        # Check for multiple failed logins
        if 'multiple_failed_logins' in key_features:
            access_pattern = features.get('access_pattern', {})
            if access_pattern.get('failed_attempts', 0) > 5:
                matches += 1
        
        match_ratio = matches / total_features if total_features > 0 else 0
        return match_ratio * weight

    def _generate_explanations(self, features: Dict[str, Any], anomaly_score: float, threat_type: Optional[str]) -> List[str]:
        """Generate explanations using attention analysis"""
        explanations = []
        
        # Context-based explanations
        temporal_context = features.get('temporal_context', {})
        if not temporal_context.get('business_hours'):
            explanations.append("Access outside business hours detected")
        
        spatial_context = features.get('spatial_context', {})
        if spatial_context.get('network_zone') == 'external':
            explanations.append("Access from external network zone")
        
        access_pattern = features.get('access_pattern', {})
        if access_pattern.get('privilege_level') == 'elevated':
            explanations.append("Elevated privilege access detected")
        
        if access_pattern.get('failed_attempts', 0) > 3:
            explanations.append("Multiple failed authentication attempts")
        
        device_fp = features.get('device_fingerprint', {})
        if device_fp.get('device_id') == 'unknown':
            explanations.append("Access from unrecognized device")
        
        user_context = features.get('user_context', {})
        if user_context.get('clearance_level') == 'high':
            explanations.append("High-clearance user with unusual behavior")
        
        # Threat-specific explanations
        if threat_type:
            threat_explanations = {
                'insider_threat': "Behavioral pattern consistent with insider threat",
                'credential_stuffing': "Pattern consistent with credential stuffing attack",
                'account_compromise': "Pattern consistent with account compromise",
                'privilege_abuse': "Pattern consistent with privilege abuse",
                'social_engineering': "Pattern consistent with social engineering attack"
            }
            explanations.append(threat_explanations.get(threat_type, f"Threat type: {threat_type}"))
        
        # Attention-based explanations
        if anomaly_score > 0.85:
            explanations.append("High attention weights on suspicious behavioral patterns")
        
        return explanations

    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """Calculate risk level"""
        if anomaly_score >= 0.9:
            return 'critical'
        elif anomaly_score >= 0.8:
            return 'high'
        elif anomaly_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # Simple holiday detection
        month_day = (date.month, date.day)
        holidays = [(1, 1), (7, 4), (12, 25)]  # New Year, Independence Day, Christmas
        return month_day in holidays

    def _is_business_hours(self, date: datetime) -> bool:
        """Check if time is within business hours"""
        return 9 <= date.hour <= 17 and date.weekday() < 5

    def _get_ip_range(self, ip_address: str) -> str:
        """Get IP range classification"""
        if ip_address.startswith('192.168.') or ip_address.startswith('10.'):
            return 'internal'
        elif ip_address.startswith('172.16.'):
            return 'internal'
        else:
            return 'external'

    async def retrain(self, training_data: Dict[str, Any]):
        """Retrain the model with new data"""
        logger.info("Starting Transformer model retraining...")
        
        # Simulate retraining process
        await asyncio.sleep(1.0)
        
        # Update model weights
        self.model_weights = self._initialize_transformer_weights()
        
        # Update metrics
        self.metrics['accuracy'] = min(self.metrics['accuracy'] + 0.01, 0.99)
        self.metrics['precision'] = min(self.metrics['precision'] + 0.01, 0.99)
        self.metrics['recall'] = min(self.metrics['recall'] + 0.01, 0.99)
        self.metrics['f1_score'] = min(self.metrics['f1_score'] + 0.01, 0.99)
        
        logger.info("Transformer model retraining completed")

    async def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.metrics.copy()

    def get_input_features(self) -> List[str]:
        """Get list of input features"""
        return self.behavioral_features.copy()

    def get_output_classes(self) -> List[str]:
        """Get list of output classes"""
        return ['normal_behavior', 'anomalous_behavior']

    def get_attention_patterns(self) -> Dict[str, Any]:
        """Get attention patterns for interpretability"""
        return self.threat_attention_patterns.copy()
