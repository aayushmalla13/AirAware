"""
Health Advisory Agent for AirAware

This agent provides intelligent health recommendations based on air quality data,
user profiles, and medical guidelines.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentConfig, AgentStatus


@dataclass
class HealthAgentConfig(AgentConfig):
    """Configuration for Health Advisory Agent"""
    # Health advisory settings
    health_guidelines_file: str = "data/health_guidelines.json"
    user_profiles_file: str = "data/user_profiles.json"
    
    # Advisory thresholds
    sensitive_group_threshold: float = 35.0  # μg/m³
    unhealthy_threshold: float = 55.0  # μg/m³
    hazardous_threshold: float = 150.0  # μg/m³
    
    # Advisory settings
    include_medical_disclaimer: bool = True
    personalized_recommendations: bool = True
    emergency_alert_threshold: float = 200.0  # μg/m³
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=lambda: {
        "update_interval": 300,  # 5 minutes
        "cache_duration": 600,   # 10 minutes
        "max_recommendations": 10
    })


@dataclass
class UserProfile:
    """User health profile"""
    user_id: str
    age_group: str  # "child", "adult", "elderly"
    health_conditions: List[str]  # ["asthma", "heart_disease", "diabetes"]
    sensitivity_level: str  # "low", "moderate", "high"
    activity_level: str  # "sedentary", "moderate", "active"
    location: str
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthRecommendation:
    """Health recommendation structure"""
    recommendation_type: str  # "prevention", "treatment", "emergency"
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    actions: List[str]
    medical_advice: Optional[str] = None
    valid_until: Optional[datetime] = None


class HealthAdvisoryAgent(BaseAgent):
    """Intelligent health advisory agent"""
    
    def __init__(self, config: HealthAgentConfig):
        super().__init__(config)
        self.health_guidelines = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.recommendation_cache: Dict[str, Dict[str, Any]] = {}
        self.last_guidelines_update: Optional[datetime] = None
        
        # Load health guidelines and user profiles
        self._load_health_guidelines()
        self._load_user_profiles()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute health advisory logic"""
        try:
            # Extract context data
            pm25_data = context.get("pm25_data", {})
            user_id = context.get("user_id")
            location = context.get("location", "unknown")
            forecast_horizon = context.get("forecast_horizon", 24)
            
            # Get current and forecasted PM2.5 levels
            current_pm25 = pm25_data.get("current", 0.0)
            forecast_pm25 = pm25_data.get("forecast", [])
            
            # Generate health recommendations
            recommendations = await self._generate_recommendations(
                current_pm25, forecast_pm25, user_id, location, forecast_horizon
            )
            
            # Generate health summary
            health_summary = self._generate_health_summary(current_pm25, forecast_pm25)
            
            # Check for emergency conditions
            emergency_alerts = self._check_emergency_conditions(current_pm25, forecast_pm25)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "health_summary": health_summary,
                "recommendations": [rec.__dict__ for rec in recommendations],
                "emergency_alerts": emergency_alerts,
                "user_id": user_id,
                "location": location,
                "pm25_levels": {
                    "current": current_pm25,
                    "forecast": forecast_pm25
                }
            }
            
        except Exception as e:
            self.logger.error(f"Health advisory execution failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check for the agent"""
        try:
            # Check if health guidelines are loaded
            if not self.health_guidelines:
                self.logger.warning("Health guidelines not loaded")
                return False
            
            # Check if user profiles are accessible
            if not self.user_profiles:
                self.logger.warning("No user profiles loaded")
            
            # Update health check timestamp
            self.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _generate_recommendations(
        self, 
        current_pm25: float, 
        forecast_pm25: List[float], 
        user_id: Optional[str], 
        location: str,
        forecast_horizon: int
    ) -> List[HealthRecommendation]:
        """Generate personalized health recommendations"""
        recommendations = []
        
        # Get user profile if available
        user_profile = self.user_profiles.get(user_id) if user_id else None
        
        # Generate recommendations based on current conditions
        current_recs = self._get_current_recommendations(current_pm25, user_profile)
        recommendations.extend(current_recs)
        
        # Generate recommendations based on forecast
        forecast_recs = self._get_forecast_recommendations(forecast_pm25, user_profile, forecast_horizon)
        recommendations.extend(forecast_recs)
        
        # Add personalized recommendations
        if user_profile and self.config.personalized_recommendations:
            personalized_recs = self._get_personalized_recommendations(user_profile, current_pm25)
            recommendations.extend(personalized_recs)
        
        # Sort by priority and limit number
        recommendations.sort(key=lambda x: self._get_priority_score(x.priority))
        return recommendations[:self.config.custom_settings["max_recommendations"]]
    
    def _get_current_recommendations(
        self, 
        pm25: float, 
        user_profile: Optional[UserProfile]
    ) -> List[HealthRecommendation]:
        """Get recommendations for current PM2.5 level"""
        recommendations = []
        
        if pm25 <= 12:
            # Good air quality
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="low",
                title="Good Air Quality",
                description="Air quality is satisfactory. Enjoy outdoor activities safely.",
                actions=[
                    "✅ Safe for outdoor activities",
                    "✅ Good time for outdoor exercise",
                    "✅ Windows can be opened for ventilation"
                ]
            ))
            
        elif pm25 <= 35:
            # Moderate air quality
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="medium",
                title="Moderate Air Quality",
                description="Air quality is acceptable for most people, but sensitive individuals should take precautions.",
                actions=[
                    "⚠️ Sensitive individuals should reduce outdoor activities",
                    "😷 Consider wearing a mask if you have respiratory issues",
                    "🪟 Keep windows closed if you're sensitive to air pollution"
                ]
            ))
            
        elif pm25 <= 55:
            # Unhealthy for sensitive groups
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="high",
                title="Unhealthy for Sensitive Groups",
                description="Members of sensitive groups may experience health effects.",
                actions=[
                    "🚫 Sensitive individuals should avoid outdoor activities",
                    "😷 Wear N95 masks when going outside",
                    "🏠 Stay indoors with windows and doors closed",
                    "💨 Use air purifiers if available"
                ]
            ))
            
        elif pm25 <= 150:
            # Unhealthy for everyone
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="high",
                title="Unhealthy Air Quality",
                description="Everyone may begin to experience health effects.",
                actions=[
                    "🚫 Avoid all outdoor activities",
                    "😷 Wear N95 or better masks if you must go outside",
                    "🏠 Stay indoors with windows and doors closed",
                    "💨 Use air purifiers and avoid activities that increase indoor pollution"
                ]
            ))
            
        else:
            # Hazardous conditions
            recommendations.append(HealthRecommendation(
                recommendation_type="emergency",
                priority="critical",
                title="Hazardous Air Quality",
                description="Health warnings of emergency conditions. The entire population is likely to be affected.",
                actions=[
                    "🚨 EMERGENCY: Stay indoors at all times",
                    "😷 Wear N95 or better masks if you must go outside",
                    "🏠 Seal windows and doors, use air purifiers",
                    "🚑 Seek medical attention if you experience breathing difficulties",
                    "📞 Follow local health department emergency guidelines"
                ],
                medical_advice="Seek immediate medical attention if you experience severe breathing difficulties, chest pain, or other serious symptoms."
            ))
        
        # Add user-specific recommendations
        if user_profile:
            user_specific = self._get_user_specific_recommendations(pm25, user_profile)
            recommendations.extend(user_specific)
        
        return recommendations
    
    def _get_forecast_recommendations(
        self, 
        forecast_pm25: List[float], 
        user_profile: Optional[UserProfile],
        forecast_horizon: int
    ) -> List[HealthRecommendation]:
        """Get recommendations based on forecast"""
        recommendations = []
        
        if not forecast_pm25:
            return recommendations
        
        # Find peak PM2.5 in forecast
        peak_pm25 = max(forecast_pm25)
        peak_hour = forecast_pm25.index(peak_pm25)
        
        # Check if forecast shows worsening conditions
        if peak_pm25 > 55:  # Unhealthy threshold
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="high",
                title="Forecasted Poor Air Quality",
                description=f"Air quality is forecasted to reach unhealthy levels (peak: {peak_pm25:.1f} μg/m³) in {peak_hour} hours.",
                actions=[
                    "📅 Plan indoor activities for the forecasted period",
                    "🛒 Stock up on necessary supplies",
                    "💨 Ensure air purifiers are working",
                    "😷 Have masks ready for essential outdoor activities"
                ],
                valid_until=datetime.now() + timedelta(hours=forecast_horizon)
            ))
        
        # Check for improving conditions
        if forecast_pm25[-1] < forecast_pm25[0] * 0.8:  # 20% improvement
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="low",
                title="Improving Air Quality Forecast",
                description="Air quality is forecasted to improve over the next 24 hours.",
                actions=[
                    "📈 Conditions expected to improve",
                    "⏰ Consider rescheduling outdoor activities for later",
                    "🪟 You may be able to open windows later in the day"
                ],
                valid_until=datetime.now() + timedelta(hours=forecast_horizon)
            ))
        
        return recommendations
    
    def _get_personalized_recommendations(
        self, 
        user_profile: UserProfile, 
        current_pm25: float
    ) -> List[HealthRecommendation]:
        """Get personalized recommendations based on user profile"""
        recommendations = []
        
        # Age-specific recommendations
        if user_profile.age_group == "child":
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="high",
                title="Child Safety Recommendations",
                description="Children are more vulnerable to air pollution effects.",
                actions=[
                    "👶 Keep children indoors during poor air quality",
                    "🏫 Check with school about outdoor activities",
                    "💨 Use child-safe air purifiers in bedrooms",
                    "🍎 Ensure good nutrition to support immune system"
                ]
            ))
        
        elif user_profile.age_group == "elderly":
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="high",
                title="Elderly Health Considerations",
                description="Older adults may be more sensitive to air pollution.",
                actions=[
                    "👴 Avoid outdoor activities during poor air quality",
                    "💊 Ensure medications are readily available",
                    "🏥 Have emergency contacts ready",
                    "💨 Use air purifiers in living areas"
                ]
            ))
        
        # Health condition-specific recommendations
        for condition in user_profile.health_conditions:
            if condition == "asthma":
                recommendations.append(HealthRecommendation(
                    recommendation_type="treatment",
                    priority="high",
                    title="Asthma Management",
                    description="Extra precautions needed for asthma management during poor air quality.",
                    actions=[
                        "💨 Keep rescue inhaler readily available",
                        "😷 Wear mask when going outside",
                        "🏠 Stay indoors with air purifier running",
                        "📞 Contact doctor if symptoms worsen"
                    ],
                    medical_advice="Monitor asthma symptoms closely and seek medical attention if breathing becomes difficult."
                ))
            
            elif condition == "heart_disease":
                recommendations.append(HealthRecommendation(
                    recommendation_type="treatment",
                    priority="high",
                    title="Heart Disease Precautions",
                    description="Air pollution can worsen heart disease symptoms.",
                    actions=[
                        "❤️ Monitor heart rate and blood pressure",
                        "🚫 Avoid strenuous activities",
                        "💊 Take medications as prescribed",
                        "📞 Contact doctor if chest pain or shortness of breath occurs"
                    ],
                    medical_advice="Seek immediate medical attention if you experience chest pain, shortness of breath, or other heart-related symptoms."
                ))
        
        return recommendations
    
    def _get_user_specific_recommendations(
        self, 
        pm25: float, 
        user_profile: UserProfile
    ) -> List[HealthRecommendation]:
        """Get user-specific recommendations based on sensitivity and activity level"""
        recommendations = []
        
        # Sensitivity-based recommendations
        if user_profile.sensitivity_level == "high":
            # Lower thresholds for sensitive users
            if pm25 > 25:  # Lower threshold for sensitive users
                recommendations.append(HealthRecommendation(
                    recommendation_type="prevention",
                    priority="high",
                    title="High Sensitivity Alert",
                    description="You are highly sensitive to air pollution. Take extra precautions.",
                    actions=[
                        "😷 Wear mask even at moderate pollution levels",
                        "🏠 Stay indoors more than usual",
                        "💨 Use air purifiers continuously",
                        "📱 Monitor air quality frequently"
                    ]
                ))
        
        # Activity level recommendations
        if user_profile.activity_level == "active":
            recommendations.append(HealthRecommendation(
                recommendation_type="prevention",
                priority="medium",
                title="Active Lifestyle Adjustments",
                description="Adjust your exercise routine based on air quality.",
                actions=[
                    "🏃 Move workouts indoors during poor air quality",
                    "🧘 Consider yoga or indoor exercises",
                    "⏰ Exercise during early morning when air quality is better",
                    "💨 Use gyms with good air filtration"
                ]
            ))
        
        return recommendations
    
    def _generate_health_summary(self, current_pm25: float, forecast_pm25: List[float]) -> Dict[str, Any]:
        """Generate health summary"""
        # Determine current health risk level
        if current_pm25 <= 12:
            risk_level = "low"
            risk_description = "Good air quality - safe for all activities"
        elif current_pm25 <= 35:
            risk_level = "moderate"
            risk_description = "Moderate air quality - sensitive individuals should take precautions"
        elif current_pm25 <= 55:
            risk_level = "high"
            risk_description = "Unhealthy for sensitive groups - limit outdoor activities"
        elif current_pm25 <= 150:
            risk_level = "very_high"
            risk_description = "Unhealthy for everyone - avoid outdoor activities"
        else:
            risk_level = "critical"
            risk_description = "Hazardous conditions - emergency precautions needed"
        
        # Calculate forecast trends
        forecast_trend = "stable"
        if forecast_pm25:
            if forecast_pm25[-1] > forecast_pm25[0] * 1.2:
                forecast_trend = "worsening"
            elif forecast_pm25[-1] < forecast_pm25[0] * 0.8:
                forecast_trend = "improving"
        
        return {
            "current_risk_level": risk_level,
            "risk_description": risk_description,
            "current_pm25": current_pm25,
            "forecast_trend": forecast_trend,
            "peak_forecast_pm25": max(forecast_pm25) if forecast_pm25 else current_pm25,
            "recommendations_count": 0,  # Will be updated by caller
            "last_updated": datetime.now().isoformat()
        }
    
    def _check_emergency_conditions(self, current_pm25: float, forecast_pm25: List[float]) -> List[Dict[str, Any]]:
        """Check for emergency conditions"""
        alerts = []
        
        if current_pm25 >= self.config.emergency_alert_threshold:
            alerts.append({
                "type": "emergency",
                "severity": "critical",
                "title": "Emergency Air Quality Alert",
                "message": f"PM2.5 levels are at hazardous levels ({current_pm25:.1f} μg/m³). Take immediate protective measures.",
                "actions": [
                    "Stay indoors immediately",
                    "Use air purifiers",
                    "Wear N95 masks if going outside",
                    "Seek medical attention if experiencing breathing difficulties"
                ],
                "timestamp": datetime.now().isoformat()
            })
        
        # Check forecast for emergency conditions
        if forecast_pm25:
            peak_pm25 = max(forecast_pm25)
            if peak_pm25 >= self.config.emergency_alert_threshold:
                peak_hour = forecast_pm25.index(peak_pm25)
                alerts.append({
                    "type": "forecast_emergency",
                    "severity": "high",
                    "title": "Forecasted Emergency Conditions",
                    "message": f"PM2.5 levels are forecasted to reach emergency levels ({peak_pm25:.1f} μg/m³) in {peak_hour} hours.",
                    "actions": [
                        "Prepare for emergency conditions",
                        "Stock up on supplies",
                        "Ensure air purifiers are working",
                        "Have emergency contacts ready"
                    ],
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def _get_priority_score(self, priority: str) -> int:
        """Get priority score for sorting"""
        priority_scores = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return priority_scores.get(priority, 0)
    
    def _load_health_guidelines(self):
        """Load health guidelines from file"""
        try:
            guidelines_path = Path(self.config.health_guidelines_file)
            if guidelines_path.exists():
                with open(guidelines_path, 'r') as f:
                    self.health_guidelines = json.load(f)
                self.last_guidelines_update = datetime.now()
                self.logger.info("Health guidelines loaded successfully")
            else:
                # Create default guidelines
                self.health_guidelines = self._create_default_guidelines()
                self._save_health_guidelines()
                self.logger.info("Created default health guidelines")
        except Exception as e:
            self.logger.error(f"Failed to load health guidelines: {e}")
            self.health_guidelines = self._create_default_guidelines()
    
    def _load_user_profiles(self):
        """Load user profiles from file"""
        try:
            profiles_path = Path(self.config.user_profiles_file)
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                
                self.user_profiles = {}
                for user_id, profile_data in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile(**profile_data)
                
                self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            else:
                self.user_profiles = {}
                self.logger.info("No user profiles file found, starting with empty profiles")
        except Exception as e:
            self.logger.error(f"Failed to load user profiles: {e}")
            self.user_profiles = {}
    
    def _create_default_guidelines(self) -> Dict[str, Any]:
        """Create default health guidelines"""
        return {
            "pm25_thresholds": {
                "good": 12,
                "moderate": 35,
                "unhealthy_sensitive": 55,
                "unhealthy": 150,
                "hazardous": 200
            },
            "health_effects": {
                "short_term": [
                    "Eye, nose, throat irritation",
                    "Coughing, sneezing, runny nose",
                    "Shortness of breath"
                ],
                "long_term": [
                    "Reduced lung function",
                    "Chronic bronchitis",
                    "Aggravated asthma",
                    "Heart disease",
                    "Lung cancer"
                ]
            },
            "vulnerable_groups": [
                "Children",
                "Elderly",
                "People with heart or lung disease",
                "Pregnant women",
                "People with diabetes"
            ],
            "protective_measures": {
                "indoor": [
                    "Use air purifiers with HEPA filters",
                    "Keep windows and doors closed",
                    "Avoid activities that create indoor pollution",
                    "Use exhaust fans when cooking"
                ],
                "outdoor": [
                    "Wear N95 or better masks",
                    "Avoid outdoor exercise",
                    "Stay indoors during peak pollution hours",
                    "Use public transportation instead of driving"
                ]
            }
        }
    
    def _save_health_guidelines(self):
        """Save health guidelines to file"""
        try:
            guidelines_path = Path(self.config.health_guidelines_file)
            guidelines_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(guidelines_path, 'w') as f:
                json.dump(self.health_guidelines, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save health guidelines: {e}")
    
    def add_user_profile(self, user_profile: UserProfile):
        """Add or update user profile"""
        self.user_profiles[user_profile.user_id] = user_profile
        self._save_user_profiles()
    
    def _save_user_profiles(self):
        """Save user profiles to file"""
        try:
            profiles_path = Path(self.config.user_profiles_file)
            profiles_path.parent.mkdir(parents=True, exist_ok=True)
            
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = profile.__dict__
            
            with open(profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save user profiles: {e}")
