#!/usr/bin/env python3
"""
Real-Time Network Traffic Anomaly Detection Simulator
=====================================================

This script simulates real-world network traffic monitoring and applies
the trained anomaly detection models for live threat detection.

Features:
- Real-time traffic simulation
- Live model inference
- Alert system for detected anomalies
- Performance monitoring
- Configurable threat scenarios
"""

import torch
import numpy as np
import pandas as pd
import time
import threading
import queue
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class NetworkTrafficSimulator:
    """Simulates real network traffic with various attack patterns"""
    
    def __init__(self):
        self.normal_patterns = {
            'http_requests': {'mean': 50, 'std': 15},
            'file_transfers': {'mean': 20, 'std': 8},
            'email_traffic': {'mean': 30, 'std': 10},
            'web_browsing': {'mean': 40, 'std': 12}
        }
        
        self.attack_patterns = {
            'ddos': {'multiplier': 10, 'duration': 30},
            'port_scan': {'rapid_connections': True, 'ports': 100},
            'botnet': {'periodic': True, 'interval': 60},
            'data_exfiltration': {'large_uploads': True, 'size_multiplier': 5}
        }
    
    def generate_normal_traffic(self):
        """Generate normal network traffic features"""
        features = []
        
        # Simulate 53 network features (matching our trained model)
        for i in range(53):
            if i < 10:  # Flow duration and packet counts
                value = np.random.normal(100, 30)
            elif i < 20:  # Byte counts and rates
                value = np.random.normal(1500, 500)
            elif i < 30:  # Timing features
                value = np.random.normal(0.1, 0.05)
            elif i < 40:  # Flag counts
                value = np.random.randint(0, 10)
            else:  # Protocol and port features
                value = np.random.normal(50, 15)
            
            features.append(max(0, value))  # Ensure non-negative
        
        return np.array(features)
    
    def inject_attack(self, attack_type, normal_features):
        """Inject attack patterns into normal traffic"""
        features = normal_features.copy()
        
        if attack_type == 'ddos':
            # Increase packet counts and rates dramatically
            features[0:5] *= np.random.uniform(5, 15)  # Packet counts
            features[10:15] *= np.random.uniform(8, 20)  # Byte rates
            
        elif attack_type == 'port_scan':
            # Rapid small packets to many destinations
            features[0] *= 0.1  # Very small packets
            features[5:10] *= np.random.uniform(10, 50)  # Many connections
            
        elif attack_type == 'botnet':
            # Periodic communication patterns
            features[20:25] = np.random.uniform(0.8, 1.2, 5)  # Regular timing
            features[30:35] *= 2  # Increased flag activity
            
        elif attack_type == 'data_exfiltration':
            # Large outbound transfers
            features[15:20] *= np.random.uniform(5, 15)  # Large uploads
            features[25:30] *= 0.5  # Longer duration
        
        return features

class RealTimeAnomalyDetector:
    """Real-time anomaly detection using trained models"""
    
    def __init__(self):
        self.models = {}
        self.scaler_params = None
        self.alert_threshold = 0.7
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load LSTM model
            self.models['lstm'] = torch.load('final_optimized_lstm_model.pth', 
                                           map_location='cpu')
            print("✅ LSTM model loaded successfully")
            
            # Load CNN model
            self.models['cnn'] = torch.load('final_fixed_cnn_model.pth', 
                                          map_location='cpu')
            print("✅ CNN model loaded successfully")
            
            # Load Autoencoder model
            self.models['autoencoder'] = torch.load('final_fixed_autoencoder_model.pth', 
                                                   map_location='cpu')
            print("✅ Autoencoder model loaded successfully")
            
        except FileNotFoundError as e:
            print(f"⚠️  Model file not found: {e}")
            print("🔄 Creating mock models for demonstration...")
            self.create_mock_models()
    
    def create_mock_models(self):
        """Create mock models for demonstration purposes"""
        class MockModel:
            def eval(self): pass
            def __call__(self, x):
                # Return realistic anomaly scores
                batch_size = x.shape[0]
                scores = torch.sigmoid(torch.randn(batch_size, 2))
                return scores
        
        self.models = {
            'lstm': MockModel(),
            'cnn': MockModel(),
            'autoencoder': MockModel()
        }
        print("🎭 Mock models created for demonstration")
    
    def preprocess_features(self, features):
        """Preprocess features for model input"""
        # Normalize features (simple min-max scaling)
        features = np.clip(features, 0, 10000)  # Clip extreme values
        features = features / 10000.0  # Scale to 0-1 range
        
        # Convert to tensor
        return torch.FloatTensor(features).unsqueeze(0)
    
    def predict_anomaly(self, features):
        """Predict if traffic is anomalous"""
        input_tensor = self.preprocess_features(features)
        predictions = {}
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
                # Extract anomaly probability
                if output.shape[1] == 2:  # Binary classification
                    anomaly_prob = torch.softmax(output, dim=1)[0, 1].item()
                else:
                    anomaly_prob = torch.sigmoid(output)[0].item()
                
                predictions[model_name] = anomaly_prob
        
        # Ensemble prediction (average of all models)
        ensemble_score = np.mean(list(predictions.values()))
        is_anomaly = ensemble_score > self.alert_threshold
        
        return {
            'ensemble_score': ensemble_score,
            'is_anomaly': is_anomaly,
            'individual_scores': predictions,
            'confidence': max(predictions.values())
        }

class RealTimeMonitor:
    """Real-time monitoring and visualization"""
    
    def __init__(self):
        self.traffic_simulator = NetworkTrafficSimulator()
        self.anomaly_detector = RealTimeAnomalyDetector()
        self.data_queue = queue.Queue(maxsize=1000)
        self.alert_queue = queue.Queue()
        
        # Data storage for visualization
        self.timestamps = deque(maxlen=100)
        self.anomaly_scores = deque(maxlen=100)
        self.traffic_volume = deque(maxlen=100)
        self.alerts = deque(maxlen=20)
        
        # Statistics
        self.total_packets = 0
        self.total_anomalies = 0
        self.start_time = time.time()
        
    def generate_traffic_stream(self):
        """Generate continuous traffic stream"""
        attack_schedule = [
            {'time': 30, 'type': 'ddos', 'duration': 15},
            {'time': 80, 'type': 'port_scan', 'duration': 10},
            {'time': 130, 'type': 'botnet', 'duration': 20},
            {'time': 200, 'type': 'data_exfiltration', 'duration': 12}
        ]
        
        start_time = time.time()
        attack_index = 0
        
        while True:
            current_time = time.time() - start_time
            
            # Check if we should inject an attack
            inject_attack = False
            attack_type = None
            
            if attack_index < len(attack_schedule):
                attack = attack_schedule[attack_index]
                if current_time >= attack['time'] and current_time <= attack['time'] + attack['duration']:
                    inject_attack = True
                    attack_type = attack['type']
                elif current_time > attack['time'] + attack['duration']:
                    attack_index += 1
            
            # Generate traffic
            if inject_attack:
                normal_features = self.traffic_simulator.generate_normal_traffic()
                features = self.traffic_simulator.inject_attack(attack_type, normal_features)
                print(f"🚨 Injecting {attack_type} attack at {current_time:.1f}s")
            else:
                features = self.traffic_simulator.generate_normal_traffic()
            
            # Add timestamp and metadata
            traffic_data = {
                'timestamp': datetime.now(),
                'features': features,
                'attack_injected': inject_attack,
                'attack_type': attack_type if inject_attack else None,
                'traffic_volume': np.sum(features[:10])  # Sum of flow features
            }
            
            # Add to queue
            if not self.data_queue.full():
                self.data_queue.put(traffic_data)
            
            # Sleep to simulate real-time (process 10 flows per second)
            time.sleep(0.1)
    
    def process_traffic(self):
        """Process traffic and detect anomalies"""
        while True:
            try:
                traffic_data = self.data_queue.get(timeout=1)
                
                # Run anomaly detection
                result = self.anomaly_detector.predict_anomaly(traffic_data['features'])
                
                # Update statistics
                self.total_packets += 1
                if result['is_anomaly']:
                    self.total_anomalies += 1
                
                # Store data for visualization
                self.timestamps.append(traffic_data['timestamp'])
                self.anomaly_scores.append(result['ensemble_score'])
                self.traffic_volume.append(traffic_data['traffic_volume'])
                
                # Generate alert if anomaly detected
                if result['is_anomaly']:
                    alert = {
                        'timestamp': traffic_data['timestamp'],
                        'score': result['ensemble_score'],
                        'confidence': result['confidence'],
                        'actual_attack': traffic_data['attack_injected'],
                        'attack_type': traffic_data['attack_type'],
                        'individual_scores': result['individual_scores']
                    }
                    
                    self.alerts.append(alert)
                    if not self.alert_queue.full():
                        self.alert_queue.put(alert)
                    
                    # Print real-time alert
                    status = "✅ TRUE POSITIVE" if traffic_data['attack_injected'] else "❌ FALSE POSITIVE"
                    print(f"\n🚨 ANOMALY DETECTED! {status}")
                    print(f"   Time: {alert['timestamp'].strftime('%H:%M:%S')}")
                    print(f"   Score: {alert['score']:.3f}")
                    print(f"   Confidence: {alert['confidence']:.3f}")
                    if traffic_data['attack_injected']:
                        print(f"   Attack Type: {traffic_data['attack_type']}")
                    print(f"   Model Scores: {result['individual_scores']}")
                
            except queue.Empty:
                continue
    
    def print_statistics(self):
        """Print real-time statistics"""
        while True:
            time.sleep(10)  # Update every 10 seconds
            
            if self.total_packets > 0:
                runtime = time.time() - self.start_time
                detection_rate = (self.total_anomalies / self.total_packets) * 100
                processing_rate = self.total_packets / runtime
                
                print(f"\n📊 REAL-TIME STATISTICS ({runtime:.1f}s)")
                print(f"   Total Packets: {self.total_packets}")
                print(f"   Anomalies Detected: {self.total_anomalies}")
                print(f"   Detection Rate: {detection_rate:.2f}%")
                print(f"   Processing Rate: {processing_rate:.1f} packets/sec")
                print(f"   Recent Alerts: {len(self.alerts)}")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print("🚀 Starting Real-Time Anomaly Detection System")
        print("=" * 60)
        
        # Start background threads
        traffic_thread = threading.Thread(target=self.generate_traffic_stream, daemon=True)
        processing_thread = threading.Thread(target=self.process_traffic, daemon=True)
        stats_thread = threading.Thread(target=self.print_statistics, daemon=True)
        
        traffic_thread.start()
        processing_thread.start()
        stats_thread.start()
        
        print("✅ All systems online!")
        print("🎯 Monitoring network traffic...")
        print("🚨 Attacks will be injected at scheduled intervals")
        print("💡 Press Ctrl+C to stop monitoring\n")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping monitoring...")
            self.print_final_report()
    
    def print_final_report(self):
        """Print final performance report"""
        runtime = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("📋 FINAL PERFORMANCE REPORT")
        print("="*60)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total Packets Processed: {self.total_packets}")
        print(f"Total Anomalies Detected: {self.total_anomalies}")
        print(f"Average Processing Rate: {self.total_packets/runtime:.1f} packets/sec")
        
        if self.alerts:
            print(f"\n🚨 ALERT SUMMARY:")
            true_positives = sum(1 for alert in self.alerts if alert['actual_attack'])
            false_positives = len(self.alerts) - true_positives
            
            print(f"   Total Alerts: {len(self.alerts)}")
            print(f"   True Positives: {true_positives}")
            print(f"   False Positives: {false_positives}")
            
            if len(self.alerts) > 0:
                precision = true_positives / len(self.alerts)
                print(f"   Precision: {precision:.3f}")
        
        print("\n✅ Real-time testing completed successfully!")

def main():
    """Main function to run real-time testing"""
    print("🌟 Enhanced Multi-Dataset Anomaly Detection System")
    print("🔬 Real-Time Network Traffic Analysis")
    print("=" * 60)
    
    # Initialize monitor
    monitor = RealTimeMonitor()
    
    # Start monitoring
    monitor.start_monitoring()

if __name__ == "__main__":
    main()