import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class InferenceService:
    """
    Central inference service that coordinates predictions across multiple ML models.
    Handles model selection, ensemble methods, and inference orchestration.
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.inference_queue = deque(maxlen=10000)
        self.inference_cache = {}
        self.model_weights = {}
        
        # Inference metrics
        self.metrics = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'avg_inference_time': 0.0,
            'cache_hit_rate': 0.0,
            'model_usage': defaultdict(int)
        }
        
        # Model selection strategy
        self.model_selection_strategy = 'best_fit'  # 'best_fit', 'ensemble', 'round_robin'
        
        # Ensemble configuration
        self.ensemble_config = {
            'method': 'weighted_average',  # 'weighted_average', 'majority_vote', 'stacking'
            'use_confidence_weighting': True,
            'min_models_for_ensemble': 2
        }
        
        # Performance tracking
        self.performance_window = deque(maxlen=1000)
        self.model_accuracy_tracker = defaultdict(lambda: deque(maxlen=100))
        
        # Cache configuration
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 10000

    async def initialize(self, models: Dict[str, Any]):
        """Initialize the inference service with available models"""
        try:
            logger.info("Initializing inference service...")
            
            self.models = models
            
            # Initialize model weights based on historical performance
            await self._initialize_model_weights()
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            logger.info(f"✅ Inference service initialized with {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference service: {e}")
            raise

    async def predict(self, model_type: str, features: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Make prediction using specified model or best available model"""
        start_time = time.time()
        inference_id = f"{int(time.time() * 1000)}_{hash(str(features)) % 10000}"
        
        try:
            logger.info(f"Starting inference {inference_id} for {model_type}")
            
            # Check cache first
            cache_key = self._generate_cache_key(model_type, features, event_type, source_type)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for inference {inference_id}")
                self.metrics['cache_hit_rate'] = (self.metrics['cache_hit_rate'] * 0.9) + (1.0 * 0.1)
                return cached_result
            
            # Validate model availability
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not available, selecting best alternative")
                model_type = await self._select_best_model(event_type, source_type)
            
            # Single model prediction
            if self.model_selection_strategy == 'best_fit' or len(self.models) == 1:
                prediction = await self._single_model_predict(model_type, features, event_type, source_type)
            
            # Ensemble prediction
            elif self.model_selection_strategy == 'ensemble':
                prediction = await self._ensemble_predict(features, event_type, source_type)
            
            # Round robin prediction
            else:
                model_type = await self._round_robin_select()
                prediction = await self._single_model_predict(model_type, features, event_type, source_type)
            
            # Post-process prediction
            prediction = await self._post_process_prediction(prediction, features, event_type, source_type)
            
            # Update metrics and tracking
            inference_time = time.time() - start_time
            await self._update_metrics(inference_id, model_type, prediction, inference_time, True)
            
            # Cache result
            await self._cache_result(cache_key, prediction)
            
            logger.info(f"Completed inference {inference_id} in {inference_time:.3f}s")
            return prediction
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Inference {inference_id} failed: {e}")
            
            # Update failure metrics
            await self._update_metrics(inference_id, model_type, None, inference_time, False)
            
            # Return fallback prediction
            return await self._fallback_prediction(features, event_type, source_type)

    async def batch_predict(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform batch prediction for multiple requests"""
        try:
            logger.info(f"Starting batch inference for {len(requests)} requests")
            
            # Group requests by model type for efficient processing
            grouped_requests = self._group_requests_by_model(requests)
            
            # Process each group concurrently
            tasks = []
            for model_type, model_requests in grouped_requests.items():
                task = self._batch_predict_single_model(model_type, model_requests)
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten and order results
            predictions = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch prediction error: {result}")
                    continue
                predictions.extend(result)
            
            logger.info(f"Completed batch inference with {len(predictions)} results")
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

    async def _single_model_predict(self, model_type: str, features: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Make prediction using a single model"""
        try:
            model = self.models[model_type]
            
            # Track model usage
            self.metrics['model_usage'][model_type] += 1
            
            # Make prediction
            prediction = await model.predict(features)
            
            # Add metadata
            prediction['model_type'] = model_type
            prediction['inference_timestamp'] = datetime.now().isoformat()
            prediction['feature_count'] = len(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Single model prediction failed for {model_type}: {e}")
            raise

    async def _ensemble_predict(self, features: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Make ensemble prediction using multiple models"""
        try:
            logger.info("Starting ensemble prediction")
            
            # Select models for ensemble
            ensemble_models = await self._select_ensemble_models(event_type, source_type)
            
            if len(ensemble_models) < self.ensemble_config['min_models_for_ensemble']:
                logger.warning("Insufficient models for ensemble, falling back to best model")
                best_model = await self._select_best_model(event_type, source_type)
                return await self._single_model_predict(best_model, features, event_type, source_type)
            
            # Get predictions from all ensemble models
            predictions = []
            tasks = []
            
            for model_type in ensemble_models:
                task = self._single_model_predict(model_type, features, event_type, source_type)
                tasks.append(task)
            
            model_predictions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed predictions
            valid_predictions = []
            for i, pred in enumerate(model_predictions):
                if not isinstance(pred, Exception):
                    valid_predictions.append(pred)
                else:
                    logger.warning(f"Model {ensemble_models[i]} failed in ensemble: {pred}")
            
            if not valid_predictions:
                raise RuntimeError("All ensemble models failed")
            
            # Combine predictions using ensemble method
            ensemble_prediction = await self._combine_predictions(valid_predictions, ensemble_models)
            
            logger.info(f"Ensemble prediction completed with {len(valid_predictions)} models")
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise

    async def _combine_predictions(self, predictions: List[Dict[str, Any]], model_types: List[str]) -> Dict[str, Any]:
        """Combine multiple predictions using ensemble method"""
        try:
            method = self.ensemble_config['method']
            
            if method == 'weighted_average':
                return await self._weighted_average_ensemble(predictions, model_types)
            elif method == 'majority_vote':
                return await self._majority_vote_ensemble(predictions, model_types)
            elif method == 'stacking':
                return await self._stacking_ensemble(predictions, model_types)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
                
        except Exception as e:
            logger.error(f"Failed to combine predictions: {e}")
            raise

    async def _weighted_average_ensemble(self, predictions: List[Dict[str, Any]], model_types: List[str]) -> Dict[str, Any]:
        """Combine predictions using weighted average"""
        try:
            total_weight = 0
            weighted_anomaly_score = 0
            weighted_confidence = 0
            
            # Collect all explanations
            all_explanations = []
            threat_types = []
            
            for i, pred in enumerate(predictions):
                model_type = model_types[i] if i < len(model_types) else pred.get('model_type', 'unknown')
                
                # Get model weight
                weight = self.model_weights.get(model_type, 1.0)
                
                # Apply confidence weighting if enabled
                if self.ensemble_config['use_confidence_weighting']:
                    confidence_weight = pred.get('confidence', 0.5)
                    weight *= confidence_weight
                
                # Accumulate weighted scores
                weighted_anomaly_score += pred.get('anomaly_score', 0) * weight
                weighted_confidence += pred.get('confidence', 0) * weight
                total_weight += weight
                
                # Collect explanations and threat types
                explanations = pred.get('explanation', [])
                if explanations:
                    all_explanations.extend(explanations)
                
                threat_type = pred.get('threat_type')
                if threat_type:
                    threat_types.append(threat_type)
            
            # Normalize by total weight
            if total_weight > 0:
                final_anomaly_score = weighted_anomaly_score / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_anomaly_score = 0
                final_confidence = 0
            
            # Determine final threat type (most common)
            final_threat_type = None
            if threat_types:
                threat_counts = defaultdict(int)
                for threat in threat_types:
                    threat_counts[threat] += 1
                final_threat_type = max(threat_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate risk level
            final_risk_level = self._calculate_risk_level(final_anomaly_score)
            
            return {
                'is_anomaly': final_anomaly_score > 0.7,
                'confidence': final_confidence,
                'anomaly_score': final_anomaly_score,
                'model_type': 'ensemble_weighted_average',
                'threat_type': final_threat_type,
                'risk_level': final_risk_level,
                'explanation': list(set(all_explanations)),  # Remove duplicates
                'ensemble_info': {
                    'models_used': model_types,
                    'model_count': len(predictions),
                    'total_weight': total_weight,
                    'individual_scores': [p.get('anomaly_score', 0) for p in predictions]
                }
            }
            
        except Exception as e:
            logger.error(f"Weighted average ensemble failed: {e}")
            raise

    async def _majority_vote_ensemble(self, predictions: List[Dict[str, Any]], model_types: List[str]) -> Dict[str, Any]:
        """Combine predictions using majority vote"""
        try:
            anomaly_votes = 0
            total_votes = len(predictions)
            
            threat_type_votes = defaultdict(int)
            risk_level_votes = defaultdict(int)
            all_explanations = []
            confidence_sum = 0
            anomaly_score_sum = 0
            
            for pred in predictions:
                # Count anomaly votes
                if pred.get('is_anomaly', False):
                    anomaly_votes += 1
                
                # Count threat type votes
                threat_type = pred.get('threat_type')
                if threat_type:
                    threat_type_votes[threat_type] += 1
                
                # Count risk level votes
                risk_level = pred.get('risk_level', 'low')
                risk_level_votes[risk_level] += 1
                
                # Accumulate scores
                confidence_sum += pred.get('confidence', 0)
                anomaly_score_sum += pred.get('anomaly_score', 0)
                
                # Collect explanations
                explanations = pred.get('explanation', [])
                all_explanations.extend(explanations)
            
            # Determine final prediction by majority vote
            is_anomaly = anomaly_votes > (total_votes / 2)
            
            # Select most voted threat type and risk level
            final_threat_type = max(threat_type_votes.items(), key=lambda x: x[1])[0] if threat_type_votes else None
            final_risk_level = max(risk_level_votes.items(), key=lambda x: x[1])[0] if risk_level_votes else 'low'
            
            # Calculate average scores
            avg_confidence = confidence_sum / total_votes
            avg_anomaly_score = anomaly_score_sum / total_votes
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': avg_confidence,
                'anomaly_score': avg_anomaly_score,
                'model_type': 'ensemble_majority_vote',
                'threat_type': final_threat_type,
                'risk_level': final_risk_level,
                'explanation': list(set(all_explanations)),
                'ensemble_info': {
                    'models_used': model_types,
                    'model_count': total_votes,
                    'anomaly_votes': anomaly_votes,
                    'vote_ratio': anomaly_votes / total_votes
                }
            }
            
        except Exception as e:
            logger.error(f"Majority vote ensemble failed: {e}")
            raise

    async def _stacking_ensemble(self, predictions: List[Dict[str, Any]], model_types: List[str]) -> Dict[str, Any]:
        """Combine predictions using stacking method"""
        try:
            # For simplicity, implement a basic stacking approach
            # In production, this would use a trained meta-model
            
            # Extract features from base model predictions
            stacking_features = []
            for pred in predictions:
                features = [
                    pred.get('anomaly_score', 0),
                    pred.get('confidence', 0),
                    1 if pred.get('is_anomaly', False) else 0,
                    self._encode_risk_level(pred.get('risk_level', 'low'))
                ]
                stacking_features.extend(features)
            
            # Simple meta-model (weighted combination with learned weights)
            meta_weights = np.array([0.8, 0.2, 0.6, 0.4] * len(predictions))
            meta_weights = meta_weights[:len(stacking_features)]
            
            meta_score = np.dot(stacking_features, meta_weights) / len(predictions)
            meta_score = np.clip(meta_score, 0, 1)
            
            # Determine final prediction
            is_anomaly = meta_score > 0.7
            confidence = min(meta_score + 0.1, 1.0)
            
            # Use weighted average for other attributes
            weighted_result = await self._weighted_average_ensemble(predictions, model_types)
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'anomaly_score': meta_score,
                'model_type': 'ensemble_stacking',
                'threat_type': weighted_result.get('threat_type'),
                'risk_level': self._calculate_risk_level(meta_score),
                'explanation': weighted_result.get('explanation', []),
                'ensemble_info': {
                    'models_used': model_types,
                    'model_count': len(predictions),
                    'meta_score': meta_score,
                    'base_predictions': [p.get('anomaly_score', 0) for p in predictions]
                }
            }
            
        except Exception as e:
            logger.error(f"Stacking ensemble failed: {e}")
            raise

    async def _select_best_model(self, event_type: str, source_type: str) -> str:
        """Select the best model for given event and source type"""
        try:
            # Model-source affinity mapping
            model_affinity = {
                'cnn': ['network', 'spatial'],
                'lstm': ['financial', 'temporal'],
                'transformer': ['iam', 'cloud', 'behavioral']
            }
            
            # Find models with affinity for this source type
            candidate_models = []
            for model_type, affinities in model_affinity.items():
                if model_type in self.models and source_type in affinities:
                    candidate_models.append(model_type)
            
            # If no specific affinity, use all available models
            if not candidate_models:
                candidate_models = list(self.models.keys())
            
            # Select based on performance
            best_model = None
            best_score = -1
            
            for model_type in candidate_models:
                # Calculate composite score: accuracy * weight * availability
                accuracy = self.model_performance.get(model_type, {}).get('accuracy', 0.5)
                weight = self.model_weights.get(model_type, 1.0)
                availability = 1.0  # Could check model health here
                
                composite_score = accuracy * weight * availability
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model = model_type
            
            return best_model or list(self.models.keys())[0]
            
        except Exception as e:
            logger.error(f"Failed to select best model: {e}")
            return list(self.models.keys())[0] if self.models else 'cnn'

    async def _select_ensemble_models(self, event_type: str, source_type: str) -> List[str]:
        """Select models for ensemble prediction"""
        try:
            # Select top performing models
            model_scores = {}
            
            for model_type in self.models.keys():
                accuracy = self.model_performance.get(model_type, {}).get('accuracy', 0.5)
                weight = self.model_weights.get(model_type, 1.0)
                model_scores[model_type] = accuracy * weight
            
            # Sort by score and select top models
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top 3 models or all if less than 3
            ensemble_size = min(3, len(sorted_models))
            selected_models = [model for model, _ in sorted_models[:ensemble_size]]
            
            return selected_models
            
        except Exception as e:
            logger.error(f"Failed to select ensemble models: {e}")
            return list(self.models.keys())

    async def _round_robin_select(self) -> str:
        """Select model using round-robin strategy"""
        # Simple round-robin implementation
        models = list(self.models.keys())
        total_usage = sum(self.metrics['model_usage'].values())
        return models[total_usage % len(models)]

    async def _post_process_prediction(self, prediction: Dict[str, Any], features: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Post-process prediction with additional logic"""
        try:
            # Add feature context
            prediction['feature_summary'] = {
                'total_features': len(features),
                'risk_indicators': features.get('derived', {}).get('risk_indicators', []),
                'risk_score': features.get('risk_score', 0)
            }
            
            # Adjust confidence based on feature quality
            feature_quality = self._assess_feature_quality(features)
            original_confidence = prediction.get('confidence', 0.5)
            adjusted_confidence = original_confidence * feature_quality
            prediction['confidence'] = adjusted_confidence
            prediction['feature_quality'] = feature_quality
            
            # Add contextual information
            prediction['context'] = {
                'event_type': event_type,
                'source_type': source_type,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Risk level validation
            anomaly_score = prediction.get('anomaly_score', 0)
            predicted_risk = prediction.get('risk_level', 'low')
            calculated_risk = self._calculate_risk_level(anomaly_score)
            
            if predicted_risk != calculated_risk:
                prediction['risk_level'] = calculated_risk
                prediction['risk_level_adjusted'] = True
            
            return prediction
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return prediction

    async def _fallback_prediction(self, features: Dict[str, Any], event_type: str, source_type: str) -> Dict[str, Any]:
        """Generate fallback prediction when all models fail"""
        try:
            logger.warning("Generating fallback prediction")
            
            # Simple rule-based fallback
            risk_indicators = features.get('derived', {}).get('risk_indicators', [])
            risk_score = features.get('risk_score', 0)
            
            # Calculate fallback anomaly score
            anomaly_score = min(len(risk_indicators) * 0.2 + risk_score * 0.5, 1.0)
            
            is_anomaly = anomaly_score > 0.6
            confidence = 0.3  # Low confidence for fallback
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'anomaly_score': anomaly_score,
                'model_type': 'fallback_rules',
                'threat_type': None,
                'risk_level': self._calculate_risk_level(anomaly_score),
                'explanation': ['Fallback rule-based detection activated', f'Risk indicators: {risk_indicators}'],
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.1,
                'anomaly_score': 0.0,
                'model_type': 'emergency_fallback',
                'risk_level': 'low',
                'explanation': ['Emergency fallback - all detection methods failed'],
                'error': True
            }

    def _assess_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess the quality of extracted features"""
        try:
            quality_score = 1.0
            
            # Check feature completeness
            required_categories = ['traditional', 'temporal', 'spatial', 'behavioral']
            present_categories = [cat for cat in required_categories if cat in features and features[cat]]
            completeness = len(present_categories) / len(required_categories)
            
            # Check for missing critical features
            critical_missing = 0
            if not features.get('traditional', {}).get('user_event_count_1h'):
                critical_missing += 0.1
            if not features.get('temporal', {}).get('hour_of_day'):
                critical_missing += 0.1
            if not features.get('spatial', {}).get('has_location'):
                critical_missing += 0.05
            
            quality_score = completeness * (1.0 - critical_missing)
            return max(quality_score, 0.1)  # Minimum quality threshold
            
        except Exception as e:
            logger.error(f"Feature quality assessment failed: {e}")
            return 0.5

    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """Calculate risk level from anomaly score"""
        if anomaly_score >= 0.9:
            return 'critical'
        elif anomaly_score >= 0.8:
            return 'high'
        elif anomaly_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _encode_risk_level(self, risk_level: str) -> float:
        """Encode risk level as numeric value"""
        risk_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        return risk_map.get(risk_level, 0.25)

    # Cache management methods
    
    def _generate_cache_key(self, model_type: str, features: Dict[str, Any], event_type: str, source_type: str) -> str:
        """Generate cache key for prediction"""
        # Create a deterministic hash of the input
        feature_str = json.dumps(features, sort_keys=True)
        cache_input = f"{model_type}:{event_type}:{source_type}:{feature_str}"
        return str(hash(cache_input))

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache"""
        try:
            if cache_key in self.inference_cache:
                cached_entry = self.inference_cache[cache_key]
                timestamp = cached_entry['timestamp']
                
                # Check if cache entry is still valid
                if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                    return cached_entry['prediction']
                else:
                    # Remove expired entry
                    del self.inference_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    async def _cache_result(self, cache_key: str, prediction: Dict[str, Any]):
        """Cache prediction result"""
        try:
            # Check cache size limit
            if len(self.inference_cache) >= self.max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.inference_cache.keys())[:100]
                for key in oldest_keys:
                    del self.inference_cache[key]
            
            self.inference_cache[cache_key] = {
                'prediction': prediction,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    # Metrics and tracking methods
    
    async def _update_metrics(self, inference_id: str, model_type: str, prediction: Optional[Dict[str, Any]], inference_time: float, success: bool):
        """Update inference metrics"""
        try:
            self.metrics['total_inferences'] += 1
            
            if success:
                self.metrics['successful_inferences'] += 1
                
                # Update average inference time
                current_avg = self.metrics['avg_inference_time']
                total_successful = self.metrics['successful_inferences']
                self.metrics['avg_inference_time'] = ((current_avg * (total_successful - 1)) + inference_time) / total_successful
                
                # Track performance by model
                if prediction and 'confidence' in prediction:
                    self.model_accuracy_tracker[model_type].append(prediction['confidence'])
            else:
                self.metrics['failed_inferences'] += 1
            
            # Add to performance window
            self.performance_window.append({
                'inference_id': inference_id,
                'model_type': model_type,
                'inference_time': inference_time,
                'success': success,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")

    async def _initialize_model_weights(self):
        """Initialize model weights based on historical performance"""
        try:
            # Default weights based on model characteristics
            default_weights = {
                'cnn': 1.0,     # Good for network and spatial patterns
                'lstm': 1.2,    # Excellent for sequential/financial patterns
                'transformer': 1.1  # Good for complex behavioral patterns
            }
            
            for model_type in self.models.keys():
                self.model_weights[model_type] = default_weights.get(model_type, 1.0)
            
            logger.info(f"Initialized model weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Model weight initialization failed: {e}")

    async def _initialize_performance_tracking(self):
        """Initialize performance tracking for all models"""
        try:
            for model_type in self.models.keys():
                # Get initial performance metrics from models
                model = self.models[model_type]
                metrics = await model.get_metrics()
                self.model_performance[model_type] = metrics
            
            logger.info(f"Initialized performance tracking for {len(self.model_performance)} models")
            
        except Exception as e:
            logger.error(f"Performance tracking initialization failed: {e}")

    def _group_requests_by_model(self, requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group batch requests by model type"""
        grouped = defaultdict(list)
        
        for request in requests:
            model_type = request.get('model_type', 'cnn')
            grouped[model_type].append(request)
        
        return dict(grouped)

    async def _batch_predict_single_model(self, model_type: str, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform batch prediction for single model"""
        try:
            results = []
            
            # Process requests concurrently (with limit to avoid overwhelming)
            semaphore = asyncio.Semaphore(10)  # Limit concurrent predictions
            
            async def predict_single(request):
                async with semaphore:
                    return await self.predict(
                        model_type,
                        request['features'],
                        request['event_type'],
                        request['source_type']
                    )
            
            tasks = [predict_single(req) for req in requests]
            predictions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for i, prediction in enumerate(predictions):
                if isinstance(prediction, Exception):
                    logger.error(f"Batch prediction failed for request {i}: {prediction}")
                    fallback = await self._fallback_prediction(
                        requests[i]['features'],
                        requests[i]['event_type'],
                        requests[i]['source_type']
                    )
                    results.append(fallback)
                else:
                    results.append(prediction)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed for model {model_type}: {e}")
            return []

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive inference service metrics"""
        try:
            # Calculate derived metrics
            total_inferences = self.metrics['total_inferences']
            success_rate = (self.metrics['successful_inferences'] / max(total_inferences, 1)) * 100
            
            # Recent performance (last 100 inferences)
            recent_performance = list(self.performance_window)[-100:]
            recent_success_rate = (sum(1 for p in recent_performance if p['success']) / max(len(recent_performance), 1)) * 100
            
            # Model-specific metrics
            model_metrics = {}
            for model_type, accuracies in self.model_accuracy_tracker.items():
                if accuracies:
                    model_metrics[model_type] = {
                        'avg_confidence': np.mean(accuracies),
                        'usage_count': self.metrics['model_usage'][model_type],
                        'weight': self.model_weights.get(model_type, 1.0)
                    }
            
            return {
                'inference_metrics': self.metrics,
                'success_rate': success_rate,
                'recent_success_rate': recent_success_rate,
                'cache_size': len(self.inference_cache),
                'model_metrics': model_metrics,
                'ensemble_config': self.ensemble_config,
                'performance_window_size': len(self.performance_window)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of inference service"""
        try:
            health_status = {
                'service_status': 'healthy',
                'models_loaded': len(self.models),
                'total_inferences': self.metrics['total_inferences'],
                'cache_size': len(self.inference_cache),
                'timestamp': datetime.now().isoformat()
            }
            
            # Check individual model health
            model_health = {}
            for model_type, model in self.models.items():
                try:
                    # Simple health check - try to get model metrics
                    await model.get_metrics()
                    model_health[model_type] = 'healthy'
                except Exception as e:
                    model_health[model_type] = f'unhealthy: {str(e)}'
                    health_status['service_status'] = 'degraded'
            
            health_status['model_health'] = model_health
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'service_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
