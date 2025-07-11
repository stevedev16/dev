import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Advanced feature extraction service for multi-domain security events.
    Extracts traditional, temporal, spatial, behavioral, and graph-derived features.
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.user_profiles = {}
        self.baseline_metrics = {}
        self.graph_cache = {}
        
        # Feature extraction configurations
        self.temporal_windows = [300, 900, 3600, 86400]  # 5min, 15min, 1hour, 1day
        self.spatial_radius_km = [1, 5, 25, 100]  # Different radius for spatial analysis
        
        # Statistical baselines for normalization
        self.feature_baselines = {
            'transaction_amount': {'mean': 250, 'std': 500, 'min': 0, 'max': 100000},
            'login_frequency': {'mean': 3, 'std': 2, 'min': 0, 'max': 50},
            'session_duration': {'mean': 480, 'std': 240, 'min': 0, 'max': 1440},
            'ip_diversity': {'mean': 2, 'std': 3, 'min': 1, 'max': 20},
            'location_changes': {'mean': 1, 'std': 2, 'min': 0, 'max': 10}
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'temporal_anomaly': 0.15,
            'spatial_anomaly': 0.20,
            'behavioral_anomaly': 0.25,
            'volume_anomaly': 0.15,
            'pattern_anomaly': 0.25
        }

    async def extract_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract comprehensive features from raw event data"""
        try:
            logger.info(f"Extracting features for {source_type}:{event_type}")
            
            # Initialize feature dictionary
            features = {
                'metadata': {
                    'event_type': event_type,
                    'source_type': source_type,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'feature_version': '1.0.0'
                }
            }
            
            # Extract different feature categories
            traditional_features = await self._extract_traditional_features(raw_data, event_type, source_type)
            temporal_features = await self._extract_temporal_features(raw_data, event_type, source_type)
            spatial_features = await self._extract_spatial_features(raw_data, event_type, source_type)
            behavioral_features = await self._extract_behavioral_features(raw_data, event_type, source_type)
            network_features = await self._extract_network_features(raw_data, event_type, source_type)
            financial_features = await self._extract_financial_features(raw_data, event_type, source_type)
            
            # Combine all features
            features.update({
                'traditional': traditional_features,
                'temporal': temporal_features,
                'spatial': spatial_features,
                'behavioral': behavioral_features,
                'network': network_features,
                'financial': financial_features
            })
            
            # Calculate derived features
            derived_features = await self._calculate_derived_features(features, raw_data)
            features['derived'] = derived_features
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(features)
            features['risk_score'] = risk_score
            
            # Update caches and profiles
            await self._update_caches(raw_data, features, event_type, source_type)
            
            logger.info(f"Feature extraction completed with {len(features)} feature categories")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise

    async def _extract_traditional_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract traditional statistical features"""
        features = {}
        
        # Event frequency features
        user_id = raw_data.get('userId', raw_data.get('user_id'))
        ip_address = raw_data.get('ipAddress', raw_data.get('ip_address'))
        
        if user_id:
            features['user_event_count_1h'] = await self._get_event_count(user_id, 3600, 'user')
            features['user_event_count_24h'] = await self._get_event_count(user_id, 86400, 'user')
            features['user_unique_ips_24h'] = await self._get_unique_count(user_id, 86400, 'ip', 'user')
        
        if ip_address:
            features['ip_event_count_1h'] = await self._get_event_count(ip_address, 3600, 'ip')
            features['ip_unique_users_1h'] = await self._get_unique_count(ip_address, 3600, 'user', 'ip')
        
        # Volume-based features
        if source_type == 'network':
            features['bytes_transferred'] = raw_data.get('bytes_transferred', 0)
            features['packet_count'] = raw_data.get('packet_count', 0)
            features['connection_duration'] = raw_data.get('connection_duration', 0)
            
        elif source_type == 'financial':
            features['transaction_amount'] = raw_data.get('amount', 0)
            features['transaction_count_1h'] = await self._get_event_count(user_id, 3600, 'user')
            
        # Authentication features
        if event_type in ['login', 'login_failed', 'logout']:
            features['failed_attempts_1h'] = await self._get_failed_attempts(user_id, 3600)
            features['successful_logins_24h'] = await self._get_successful_logins(user_id, 86400)
            
        # Resource access features
        features['resource_diversity'] = len(set(raw_data.get('resources_accessed', [])))
        features['privilege_level'] = self._encode_privilege_level(raw_data.get('privilege_level', 'standard'))
        
        # Normalize features
        for key, value in features.items():
            if isinstance(value, (int, float)) and key in self.feature_baselines:
                features[f'{key}_normalized'] = self._normalize_feature(key, value)
        
        return features

    async def _extract_temporal_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract temporal pattern features"""
        features = {}
        
        # Parse timestamp
        timestamp_str = raw_data.get('timestamp', datetime.now().isoformat())
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
        
        # Basic temporal features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['day_of_month'] = timestamp.day
        features['month'] = timestamp.month
        features['is_weekend'] = timestamp.weekday() >= 5
        features['is_business_hours'] = 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5
        features['is_night_time'] = timestamp.hour < 6 or timestamp.hour > 22
        
        # Holiday detection
        features['is_holiday'] = self._is_holiday(timestamp)
        
        # Time since last event
        user_id = raw_data.get('userId', raw_data.get('user_id'))
        if user_id:
            last_event_time = await self._get_last_event_time(user_id)
            if last_event_time:
                time_diff = (timestamp - last_event_time).total_seconds()
                features['time_since_last_event'] = time_diff
                features['time_since_last_event_normalized'] = min(time_diff / 86400, 1.0)  # Normalize to days
        
        # Velocity features (events per time window)
        for window in self.temporal_windows:
            window_name = self._window_name(window)
            if user_id:
                features[f'event_velocity_{window_name}'] = await self._get_event_count(user_id, window, 'user')
        
        # Temporal pattern analysis
        features['usual_time_score'] = await self._calculate_usual_time_score(user_id, timestamp.hour)
        features['temporal_entropy'] = await self._calculate_temporal_entropy(user_id, timestamp)
        
        # Session-based temporal features
        session_id = raw_data.get('session_id')
        if session_id:
            features['session_duration'] = await self._get_session_duration(session_id)
            features['session_activity_rate'] = await self._get_session_activity_rate(session_id)
        
        return features

    async def _extract_spatial_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract spatial and geographical features"""
        features = {}
        
        # Extract location data
        location = raw_data.get('location', {})
        if not location and 'coordinates' in raw_data:
            location = raw_data['coordinates']
        
        if location and 'latitude' in location and 'longitude' in location:
            lat = location['latitude']
            lng = location['longitude']
            
            features['latitude'] = lat
            features['longitude'] = lng
            features['has_location'] = 1
            
            # Country and city analysis
            country = location.get('country', 'Unknown')
            city = location.get('city', 'Unknown')
            features['country'] = country
            features['city'] = city
            features['is_domestic'] = 1 if country in ['US', 'USA', 'United States'] else 0
            
            # High-risk location detection
            high_risk_countries = ['Unknown', 'TOR', 'VPN', 'Proxy']
            features['is_high_risk_location'] = 1 if country in high_risk_countries else 0
            
            # Distance calculations
            user_id = raw_data.get('userId', raw_data.get('user_id'))
            if user_id:
                last_location = await self._get_last_location(user_id)
                if last_location:
                    distance = self._calculate_distance(lat, lng, last_location['lat'], last_location['lng'])
                    features['distance_from_last_location'] = distance
                    features['impossible_travel'] = 1 if distance > 1000 else 0  # >1000km might be impossible
                    
                    # Travel velocity
                    last_time = await self._get_last_event_time(user_id)
                    if last_time:
                        time_diff = (datetime.now() - last_time).total_seconds() / 3600  # hours
                        if time_diff > 0:
                            travel_speed = distance / time_diff  # km/h
                            features['travel_velocity'] = travel_speed
                            features['impossible_velocity'] = 1 if travel_speed > 1000 else 0  # >1000 km/h impossible
                
                # Location diversity
                features['location_entropy'] = await self._calculate_location_entropy(user_id)
                
                # Radius-based location features
                for radius in self.spatial_radius_km:
                    nearby_events = await self._get_events_in_radius(lat, lng, radius, 86400)  # 24h
                    features[f'events_within_{radius}km_24h'] = len(nearby_events)
        else:
            features['has_location'] = 0
            features['is_domestic'] = 0
            features['is_high_risk_location'] = 0
        
        # IP geolocation features
        ip_address = raw_data.get('ipAddress', raw_data.get('ip_address'))
        if ip_address:
            features['ip_type'] = self._classify_ip_type(ip_address)
            features['is_private_ip'] = 1 if self._is_private_ip(ip_address) else 0
            features['is_vpn_proxy'] = await self._check_vpn_proxy(ip_address)
        
        return features

    async def _extract_behavioral_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract user behavioral pattern features"""
        features = {}
        
        user_id = raw_data.get('userId', raw_data.get('user_id'))
        if not user_id:
            return features
        
        # User profile features
        user_profile = await self._get_user_profile(user_id)
        if user_profile:
            features['account_age_days'] = (datetime.now() - user_profile['created_date']).days
            features['user_risk_score'] = user_profile.get('risk_score', 0.5)
            features['user_privilege_level'] = self._encode_privilege_level(user_profile.get('privilege_level', 'standard'))
        
        # Device and user agent analysis
        user_agent = raw_data.get('user_agent', raw_data.get('userAgent'))
        if user_agent:
            features['user_agent_entropy'] = self._calculate_string_entropy(user_agent)
            features['is_bot_user_agent'] = 1 if self._is_bot_user_agent(user_agent) else 0
        
        device_id = raw_data.get('device_id', raw_data.get('deviceId'))
        if device_id:
            features['is_new_device'] = 1 if await self._is_new_device(user_id, device_id) else 0
            features['device_trust_score'] = await self._get_device_trust_score(user_id, device_id)
        
        # Access pattern analysis
        resources_accessed = raw_data.get('resources_accessed', [])
        if resources_accessed:
            features['resource_count'] = len(resources_accessed)
            features['sensitive_resource_access'] = sum(1 for r in resources_accessed if self._is_sensitive_resource(r))
            features['admin_resource_access'] = sum(1 for r in resources_accessed if 'admin' in str(r).lower())
        
        # Behavioral deviation analysis
        features['behavior_deviation_score'] = await self._calculate_behavior_deviation(user_id, raw_data)
        features['access_pattern_anomaly'] = await self._calculate_access_pattern_anomaly(user_id, raw_data)
        
        # Session behavior
        session_id = raw_data.get('session_id')
        if session_id:
            features['session_anomaly_score'] = await self._calculate_session_anomaly(session_id, raw_data)
        
        # Typing pattern / biometric features (if available)
        if 'keystroke_dynamics' in raw_data:
            keystroke_data = raw_data['keystroke_dynamics']
            features['keystroke_anomaly'] = self._analyze_keystroke_pattern(user_id, keystroke_data)
        
        return features

    async def _extract_network_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract network-specific features"""
        features = {}
        
        if source_type != 'network':
            return features
        
        # Protocol analysis
        protocol = raw_data.get('protocol', '').lower()
        features['protocol_type'] = protocol
        features['is_encrypted_protocol'] = 1 if protocol in ['https', 'ssh', 'ssl', 'tls'] else 0
        features['is_suspicious_protocol'] = 1 if protocol in ['tor', 'socks', 'proxy'] else 0
        
        # Port analysis
        source_port = raw_data.get('source_port', raw_data.get('sourcePort', 0))
        dest_port = raw_data.get('destination_port', raw_data.get('destinationPort', 0))
        
        features['source_port'] = source_port
        features['destination_port'] = dest_port
        features['is_well_known_port'] = 1 if dest_port < 1024 else 0
        features['is_suspicious_port'] = 1 if dest_port in [1433, 3389, 22, 23, 135, 139, 445] else 0
        
        # Traffic volume analysis
        bytes_transferred = raw_data.get('bytes_transferred', 0)
        packet_count = raw_data.get('packet_count', 0)
        
        features['bytes_transferred'] = bytes_transferred
        features['packet_count'] = packet_count
        features['avg_packet_size'] = bytes_transferred / max(packet_count, 1)
        features['is_large_transfer'] = 1 if bytes_transferred > 10**8 else 0  # >100MB
        
        # Connection characteristics
        connection_duration = raw_data.get('connection_duration', 0)
        features['connection_duration'] = connection_duration
        features['data_rate'] = bytes_transferred / max(connection_duration, 1)  # bytes per second
        
        # TCP flags analysis
        tcp_flags = raw_data.get('tcp_flags', [])
        if tcp_flags:
            features['tcp_flag_count'] = len(tcp_flags)
            features['has_syn_flag'] = 1 if 'SYN' in tcp_flags else 0
            features['has_rst_flag'] = 1 if 'RST' in tcp_flags else 0
            features['has_fin_flag'] = 1 if 'FIN' in tcp_flags else 0
        
        # DNS analysis (if applicable)
        if 'dns_query' in raw_data:
            dns_query = raw_data['dns_query']
            features['dns_query_length'] = len(dns_query)
            features['dns_subdomain_count'] = dns_query.count('.')
            features['is_dga_domain'] = 1 if self._is_dga_domain(dns_query) else 0
        
        # HTTP analysis (if applicable)
        if protocol == 'http' or protocol == 'https':
            user_agent = raw_data.get('user_agent', '')
            features['user_agent_length'] = len(user_agent)
            features['is_bot_request'] = 1 if self._is_bot_user_agent(user_agent) else 0
            
            http_method = raw_data.get('http_method', 'GET')
            features['http_method'] = http_method
            features['is_post_request'] = 1 if http_method == 'POST' else 0
        
        return features

    async def _extract_financial_features(self, raw_data: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Extract financial transaction features"""
        features = {}
        
        if source_type != 'financial':
            return features
        
        # Transaction amount analysis
        amount = raw_data.get('amount', 0)
        currency = raw_data.get('currency', 'USD')
        
        features['transaction_amount'] = amount
        features['currency'] = currency
        features['is_large_amount'] = 1 if amount > 10000 else 0
        features['is_round_amount'] = 1 if amount % 100 == 0 and amount > 0 else 0
        
        # Account analysis
        from_account = raw_data.get('from_account', raw_data.get('fromAccount'))
        to_account = raw_data.get('to_account', raw_data.get('toAccount'))
        
        if from_account:
            features['from_account_type'] = self._classify_account_type(from_account)
            features['from_account_age'] = await self._get_account_age(from_account)
        
        if to_account:
            features['to_account_type'] = self._classify_account_type(to_account)
            features['to_account_age'] = await self._get_account_age(to_account)
        
        # Transaction type analysis
        transaction_type = raw_data.get('transaction_type', raw_data.get('transactionType', 'unknown'))
        features['transaction_type'] = transaction_type
        features['is_wire_transfer'] = 1 if transaction_type == 'wire_transfer' else 0
        features['is_cash_transaction'] = 1 if transaction_type in ['atm_withdrawal', 'cash_deposit'] else 0
        
        # Merchant analysis
        merchant_id = raw_data.get('merchant_id', raw_data.get('merchantId'))
        if merchant_id:
            features['merchant_category'] = await self._get_merchant_category(merchant_id)
            features['merchant_risk_score'] = await self._get_merchant_risk_score(merchant_id)
        
        # Velocity analysis
        user_id = raw_data.get('userId', raw_data.get('user_id'))
        if user_id:
            for window in self.temporal_windows:
                window_name = self._window_name(window)
                features[f'transaction_count_{window_name}'] = await self._get_transaction_count(user_id, window)
                features[f'transaction_volume_{window_name}'] = await self._get_transaction_volume(user_id, window)
        
        # Risk indicators
        features['cross_border'] = 1 if raw_data.get('location', {}).get('country', 'US') != 'US' else 0
        features['unusual_time'] = 1 if raw_data.get('hour', 12) < 6 or raw_data.get('hour', 12) > 22 else 0
        
        return features

    async def _calculate_derived_features(self, features: Dict[str, Any], raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from base features"""
        derived = {}
        
        traditional = features.get('traditional', {})
        temporal = features.get('temporal', {})
        spatial = features.get('spatial', {})
        behavioral = features.get('behavioral', {})
        network = features.get('network', {})
        financial = features.get('financial', {})
        
        # Cross-domain derived features
        
        # Velocity-based features
        if 'event_velocity_1h' in traditional and 'event_velocity_24h' in traditional:
            recent_velocity = traditional['event_velocity_1h']
            daily_velocity = traditional['event_velocity_24h'] / 24
            derived['velocity_acceleration'] = recent_velocity / max(daily_velocity, 1)
        
        # Spatio-temporal features
        if spatial.get('has_location') and temporal.get('time_since_last_event'):
            distance = spatial.get('distance_from_last_location', 0)
            time_diff = temporal.get('time_since_last_event', 3600) / 3600  # hours
            if time_diff > 0:
                derived['travel_velocity'] = distance / time_diff
                derived['impossible_travel_score'] = min(derived['travel_velocity'] / 1000, 1.0)  # Normalize by max reasonable speed
        
        # Behavioral consistency score
        consistency_factors = []
        if temporal.get('usual_time_score') is not None:
            consistency_factors.append(temporal['usual_time_score'])
        if spatial.get('distance_from_last_location') is not None:
            # Normalize distance (closer = more consistent)
            distance_score = 1 / (1 + spatial['distance_from_last_location'] / 100)
            consistency_factors.append(distance_score)
        if behavioral.get('behavior_deviation_score') is not None:
            consistency_factors.append(1 - behavioral['behavior_deviation_score'])
        
        if consistency_factors:
            derived['behavioral_consistency'] = np.mean(consistency_factors)
        
        # Multi-factor risk indicators
        risk_indicators = []
        
        # Temporal risk
        if temporal.get('is_night_time') or temporal.get('is_weekend'):
            risk_indicators.append('unusual_time')
        
        # Spatial risk
        if spatial.get('is_high_risk_location') or spatial.get('impossible_travel'):
            risk_indicators.append('suspicious_location')
        
        # Behavioral risk
        if behavioral.get('is_new_device') or behavioral.get('behavior_deviation_score', 0) > 0.7:
            risk_indicators.append('behavioral_anomaly')
        
        # Network risk
        if network.get('is_suspicious_protocol') or network.get('is_large_transfer'):
            risk_indicators.append('network_anomaly')
        
        # Financial risk
        if financial.get('is_large_amount') or financial.get('cross_border'):
            risk_indicators.append('financial_anomaly')
        
        derived['risk_indicators'] = risk_indicators
        derived['risk_indicator_count'] = len(risk_indicators)
        
        # Composite anomaly scores
        derived['temporal_anomaly_score'] = self._calculate_temporal_anomaly_score(temporal)
        derived['spatial_anomaly_score'] = self._calculate_spatial_anomaly_score(spatial)
        derived['behavioral_anomaly_score'] = self._calculate_behavioral_anomaly_score(behavioral)
        derived['network_anomaly_score'] = self._calculate_network_anomaly_score(network)
        derived['financial_anomaly_score'] = self._calculate_financial_anomaly_score(financial)
        
        return derived

    def _calculate_risk_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall risk score using weighted anomaly scores"""
        derived = features.get('derived', {})
        
        risk_components = {
            'temporal_anomaly': derived.get('temporal_anomaly_score', 0),
            'spatial_anomaly': derived.get('spatial_anomaly_score', 0),
            'behavioral_anomaly': derived.get('behavioral_anomaly_score', 0),
            'volume_anomaly': derived.get('network_anomaly_score', 0),
            'pattern_anomaly': derived.get('financial_anomaly_score', 0)
        }
        
        weighted_score = 0
        total_weight = 0
        
        for component, weight in self.risk_weights.items():
            if component in risk_components:
                weighted_score += risk_components[component] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            risk_score = weighted_score / total_weight
        else:
            risk_score = 0
        
        # Apply risk indicator bonus
        risk_indicator_count = derived.get('risk_indicator_count', 0)
        risk_score += min(risk_indicator_count * 0.1, 0.3)  # Max 30% bonus
        
        return min(risk_score, 1.0)

    # Helper methods for feature calculation
    
    async def _get_event_count(self, identifier: str, window_seconds: int, id_type: str) -> int:
        """Get event count for identifier in time window"""
        # This would query the database for actual event counts
        # For now, return a simulated value based on cache
        cache_key = f"{id_type}_{identifier}_{window_seconds}"
        return self.feature_cache.get(cache_key, np.random.randint(0, 10))
    
    async def _get_unique_count(self, identifier: str, window_seconds: int, count_field: str, id_type: str) -> int:
        """Get unique count of field for identifier in time window"""
        cache_key = f"{id_type}_{identifier}_{count_field}_{window_seconds}"
        return self.feature_cache.get(cache_key, np.random.randint(1, 5))
    
    async def _get_failed_attempts(self, user_id: str, window_seconds: int) -> int:
        """Get failed authentication attempts for user"""
        cache_key = f"failed_attempts_{user_id}_{window_seconds}"
        return self.feature_cache.get(cache_key, 0)
    
    async def _get_successful_logins(self, user_id: str, window_seconds: int) -> int:
        """Get successful login count for user"""
        cache_key = f"successful_logins_{user_id}_{window_seconds}"
        return self.feature_cache.get(cache_key, np.random.randint(0, 5))
    
    async def _get_last_event_time(self, user_id: str) -> Optional[datetime]:
        """Get timestamp of last event for user"""
        cache_key = f"last_event_{user_id}"
        cached_time = self.feature_cache.get(cache_key)
        if cached_time:
            return datetime.fromisoformat(cached_time)
        return None
    
    async def _get_last_location(self, user_id: str) -> Optional[Dict[str, float]]:
        """Get last known location for user"""
        cache_key = f"last_location_{user_id}"
        return self.feature_cache.get(cache_key)
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two coordinates in km"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lng/2) * math.sin(delta_lng/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature using z-score normalization"""
        if feature_name not in self.feature_baselines:
            return value
        
        baseline = self.feature_baselines[feature_name]
        normalized = (value - baseline['mean']) / baseline['std']
        return np.clip(normalized, -3, 3)  # Clip to reasonable range
    
    def _encode_privilege_level(self, privilege: str) -> float:
        """Encode privilege level as numeric value"""
        privilege_map = {
            'guest': 0.1,
            'standard': 0.3,
            'user': 0.3,
            'power_user': 0.5,
            'admin': 0.7,
            'super_admin': 0.9,
            'root': 1.0
        }
        return privilege_map.get(privilege.lower(), 0.3)
    
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday"""
        # Basic holiday detection
        month_day = (date.month, date.day)
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 28), # Thanksgiving (approximate)
        ]
        return month_day in holidays
    
    def _window_name(self, seconds: int) -> str:
        """Convert seconds to readable window name"""
        if seconds < 3600:
            return f"{seconds//60}m"
        elif seconds < 86400:
            return f"{seconds//3600}h"
        else:
            return f"{seconds//86400}d"
    
    def _calculate_string_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of string"""
        if not s:
            return 0
        
        char_counts = defaultdict(int)
        for char in s:
            char_counts[char] += 1
        
        entropy = 0
        length = len(s)
        for count in char_counts.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _is_bot_user_agent(self, user_agent: str) -> bool:
        """Check if user agent indicates bot/automated access"""
        bot_indicators = ['bot', 'crawler', 'spider', 'scraper', 'curl', 'wget', 'python']
        return any(indicator in user_agent.lower() for indicator in bot_indicators)
    
    def _classify_ip_type(self, ip: str) -> str:
        """Classify IP address type"""
        if self._is_private_ip(ip):
            return 'private'
        elif ip.startswith('127.'):
            return 'localhost'
        else:
            return 'public'
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private range"""
        private_ranges = ['192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.2', '172.3']
        return any(ip.startswith(range_prefix) for range_prefix in private_ranges)
    
    async def _check_vpn_proxy(self, ip: str) -> int:
        """Check if IP is VPN/proxy (simplified)"""
        # In real implementation, this would check against VPN/proxy databases
        return 0
    
    def _is_sensitive_resource(self, resource: str) -> bool:
        """Check if resource is sensitive"""
        sensitive_keywords = ['admin', 'config', 'secret', 'password', 'key', 'private', 'confidential']
        return any(keyword in str(resource).lower() for keyword in sensitive_keywords)
    
    def _is_dga_domain(self, domain: str) -> bool:
        """Check if domain looks like it's generated by DGA"""
        # Simple heuristic: high entropy and unusual patterns
        entropy = self._calculate_string_entropy(domain.split('.')[0])
        return entropy > 3.5  # Threshold for DGA detection
    
    def _calculate_temporal_anomaly_score(self, temporal_features: Dict[str, Any]) -> float:
        """Calculate temporal anomaly score"""
        score = 0
        
        if temporal_features.get('is_night_time'):
            score += 0.3
        if temporal_features.get('is_weekend'):
            score += 0.2
        if temporal_features.get('is_holiday'):
            score += 0.2
        if temporal_features.get('usual_time_score', 1) < 0.3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_spatial_anomaly_score(self, spatial_features: Dict[str, Any]) -> float:
        """Calculate spatial anomaly score"""
        score = 0
        
        if spatial_features.get('is_high_risk_location'):
            score += 0.4
        if spatial_features.get('impossible_travel'):
            score += 0.5
        if spatial_features.get('distance_from_last_location', 0) > 500:
            score += 0.3
        if not spatial_features.get('is_domestic'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_behavioral_anomaly_score(self, behavioral_features: Dict[str, Any]) -> float:
        """Calculate behavioral anomaly score"""
        score = 0
        
        if behavioral_features.get('is_new_device'):
            score += 0.3
        if behavioral_features.get('behavior_deviation_score', 0) > 0.7:
            score += 0.4
        if behavioral_features.get('access_pattern_anomaly', 0) > 0.7:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_network_anomaly_score(self, network_features: Dict[str, Any]) -> float:
        """Calculate network anomaly score"""
        score = 0
        
        if network_features.get('is_suspicious_protocol'):
            score += 0.3
        if network_features.get('is_large_transfer'):
            score += 0.3
        if network_features.get('is_suspicious_port'):
            score += 0.2
        if network_features.get('is_dga_domain'):
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_financial_anomaly_score(self, financial_features: Dict[str, Any]) -> float:
        """Calculate financial anomaly score"""
        score = 0
        
        if financial_features.get('is_large_amount'):
            score += 0.3
        if financial_features.get('cross_border'):
            score += 0.2
        if financial_features.get('is_wire_transfer'):
            score += 0.2
        if financial_features.get('unusual_time'):
            score += 0.2
        if financial_features.get('is_round_amount'):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _update_caches(self, raw_data: Dict[str, Any], features: Dict[str, Any], event_type: str, source_type: str):
        """Update feature caches with new data"""
        user_id = raw_data.get('userId', raw_data.get('user_id'))
        
        if user_id:
            # Update last event time
            self.feature_cache[f"last_event_{user_id}"] = datetime.now().isoformat()
            
            # Update location cache
            location = raw_data.get('location')
            if location and 'latitude' in location:
                self.feature_cache[f"last_location_{user_id}"] = {
                    'lat': location['latitude'],
                    'lng': location['longitude']
                }
    
    # Placeholder methods for complex calculations that would require database queries
    async def _calculate_usual_time_score(self, user_id: str, hour: int) -> float:
        """Calculate how usual this time is for the user"""
        return 0.8  # Placeholder
    
    async def _calculate_temporal_entropy(self, user_id: str, timestamp: datetime) -> float:
        """Calculate temporal pattern entropy for user"""
        return 0.5  # Placeholder
    
    async def _get_session_duration(self, session_id: str) -> float:
        """Get current session duration"""
        return 480.0  # Placeholder
    
    async def _get_session_activity_rate(self, session_id: str) -> float:
        """Get session activity rate"""
        return 0.5  # Placeholder
    
    async def _calculate_location_entropy(self, user_id: str) -> float:
        """Calculate location diversity entropy"""
        return 0.3  # Placeholder
    
    async def _get_events_in_radius(self, lat: float, lng: float, radius_km: float, window_seconds: int) -> List[Any]:
        """Get events within radius and time window"""
        return []  # Placeholder
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data"""
        return {
            'created_date': datetime.now() - timedelta(days=365),
            'risk_score': 0.3,
            'privilege_level': 'standard'
        }  # Placeholder
    
    async def _is_new_device(self, user_id: str, device_id: str) -> bool:
        """Check if device is new for user"""
        return False  # Placeholder
    
    async def _get_device_trust_score(self, user_id: str, device_id: str) -> float:
        """Get device trust score"""
        return 0.8  # Placeholder
    
    async def _calculate_behavior_deviation(self, user_id: str, raw_data: Dict[str, Any]) -> float:
        """Calculate behavioral deviation score"""
        return 0.2  # Placeholder
    
    async def _calculate_access_pattern_anomaly(self, user_id: str, raw_data: Dict[str, Any]) -> float:
        """Calculate access pattern anomaly score"""
        return 0.1  # Placeholder
    
    async def _calculate_session_anomaly(self, session_id: str, raw_data: Dict[str, Any]) -> float:
        """Calculate session anomaly score"""
        return 0.15  # Placeholder
    
    def _analyze_keystroke_pattern(self, user_id: str, keystroke_data: Dict[str, Any]) -> float:
        """Analyze keystroke dynamics for anomalies"""
        return 0.1  # Placeholder
    
    def _classify_account_type(self, account_id: str) -> str:
        """Classify account type"""
        return 'personal'  # Placeholder
    
    async def _get_account_age(self, account_id: str) -> int:
        """Get account age in days"""
        return 365  # Placeholder
    
    async def _get_merchant_category(self, merchant_id: str) -> str:
        """Get merchant category"""
        return 'retail'  # Placeholder
    
    async def _get_merchant_risk_score(self, merchant_id: str) -> float:
        """Get merchant risk score"""
        return 0.2  # Placeholder
    
    async def _get_transaction_count(self, user_id: str, window_seconds: int) -> int:
        """Get transaction count for user in window"""
        return np.random.randint(0, 5)  # Placeholder
    
    async def _get_transaction_volume(self, user_id: str, window_seconds: int) -> float:
        """Get transaction volume for user in window"""
        return np.random.uniform(0, 10000)  # Placeholder
