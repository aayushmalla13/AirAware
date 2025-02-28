"""
Alert and Notification Agent for AirAware

This agent manages alerts, notifications, and communication with users.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from .base_agent import BaseAgent, AgentConfig, AgentStatus


@dataclass
class NotificationAgentConfig(AgentConfig):
    """Configuration for Notification Agent"""
    # Notification settings
    notification_history_file: str = "data/notification_history.json"
    user_preferences_file: str = "data/user_preferences.json"
    
    # Alert thresholds
    pm25_warning_threshold: float = 35.0  # μg/m³
    pm25_critical_threshold: float = 55.0  # μg/m³
    pm25_emergency_threshold: float = 150.0  # μg/m³
    
    # Notification channels
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = False
    webhook_enabled: bool = False
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    from_email: str = "airaware@example.com"
    
    # Rate limiting
    max_notifications_per_hour: int = 10
    max_notifications_per_day: int = 50
    notification_cooldown_minutes: int = 30
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=lambda: {
        "update_interval": 300,  # 5 minutes
        "notification_cache_duration": 3600,  # 1 hour
        "max_retry_attempts": 3,
        "retry_delay_seconds": 60
    })


@dataclass
class UserPreferences:
    """User notification preferences"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    
    # Alert preferences
    pm25_warning_enabled: bool = True
    pm25_critical_enabled: bool = True
    pm25_emergency_enabled: bool = True
    
    # Notification channels
    email_notifications: bool = True
    sms_notifications: bool = False
    push_notifications: bool = False
    
    # Timing preferences
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 7     # 7 AM
    timezone: str = "UTC"
    
    # Frequency limits
    max_notifications_per_hour: int = 5
    max_notifications_per_day: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "phone": self.phone,
            "push_token": self.push_token,
            "pm25_warning_enabled": self.pm25_warning_enabled,
            "pm25_critical_enabled": self.pm25_critical_enabled,
            "pm25_emergency_enabled": self.pm25_emergency_enabled,
            "email_notifications": self.email_notifications,
            "sms_notifications": self.sms_notifications,
            "push_notifications": self.push_notifications,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "timezone": self.timezone,
            "max_notifications_per_hour": self.max_notifications_per_hour,
            "max_notifications_per_day": self.max_notifications_per_day
        }


@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    user_id: str
    station_id: str
    alert_type: str  # "warning", "critical", "emergency"
    severity: str    # "low", "medium", "high", "critical"
    title: str
    message: str
    pm25_level: float
    threshold: float
    timestamp: datetime
    expires_at: Optional[datetime] = None
    acknowledged: bool = False
    channels: List[str] = field(default_factory=list)  # ["email", "sms", "push"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "station_id": self.station_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "pm25_level": self.pm25_level,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "acknowledged": self.acknowledged,
            "channels": self.channels
        }


@dataclass
class Notification:
    """Notification structure"""
    notification_id: str
    user_id: str
    channel: str  # "email", "sms", "push", "webhook"
    subject: str
    content: str
    alert_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # "pending", "sent", "failed", "delivered"
    retry_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "channel": self.channel,
            "subject": self.subject,
            "content": self.content,
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


class NotificationAgent(BaseAgent):
    """Intelligent alert and notification agent"""
    
    def __init__(self, config: NotificationAgentConfig):
        super().__init__(config)
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_history: List[Notification] = []
        self.rate_limits: Dict[str, Dict[str, int]] = {}  # user_id -> {hour: count, day: count}
        self.last_notification_update: Optional[datetime] = None
        
        # Load user preferences and notification history
        self._load_user_preferences()
        self._load_notification_history()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification logic"""
        try:
            # Extract context data
            pm25_data = context.get("pm25_data", {})
            station_id = context.get("station_id")
            user_id = context.get("user_id")
            forecast_data = context.get("forecast_data", [])
            
            # Check for alert conditions
            alerts = await self._check_alert_conditions(
                pm25_data, station_id, user_id, forecast_data
            )
            
            # Process alerts and send notifications
            notifications_sent = []
            for alert in alerts:
                notifications = await self._process_alert(alert)
                notifications_sent.extend(notifications)
            
            # Clean up expired alerts
            expired_alerts = self._cleanup_expired_alerts()
            
            # Update rate limits
            self._update_rate_limits()
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "alerts_generated": len(alerts),
                "notifications_sent": len(notifications_sent),
                "expired_alerts": len(expired_alerts),
                "alerts": [alert.to_dict() for alert in alerts],
                "notifications": [notif.to_dict() for notif in notifications_sent]
            }
            
        except Exception as e:
            self.logger.error(f"Notification execution failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check for the agent"""
        try:
            # Check if user preferences are loaded
            if not self.user_preferences:
                self.logger.warning("No user preferences loaded")
            
            # Check email configuration if enabled
            if self.config.email_enabled:
                if not self.config.smtp_username or not self.config.smtp_password:
                    self.logger.warning("Email configuration incomplete")
            
            # Update health check timestamp
            self.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _check_alert_conditions(
        self,
        pm25_data: Dict[str, Any],
        station_id: str,
        user_id: Optional[str],
        forecast_data: List[Dict[str, Any]]
    ) -> List[Alert]:
        """Check for alert conditions and generate alerts"""
        alerts = []
        
        if not pm25_data:
            return alerts
        
        current_pm25 = pm25_data.get("current", 0.0)
        
        # Check current conditions
        if current_pm25 >= self.config.pm25_emergency_threshold:
            alert = self._create_alert(
                user_id, station_id, "emergency", "critical",
                "Emergency Air Quality Alert",
                f"PM2.5 levels are at hazardous levels ({current_pm25:.1f} μg/m³). Take immediate protective measures.",
                current_pm25, self.config.pm25_emergency_threshold
            )
            alerts.append(alert)
        
        elif current_pm25 >= self.config.pm25_critical_threshold:
            alert = self._create_alert(
                user_id, station_id, "critical", "high",
                "Critical Air Quality Alert",
                f"PM2.5 levels are unhealthy ({current_pm25:.1f} μg/m³). Avoid outdoor activities.",
                current_pm25, self.config.pm25_critical_threshold
            )
            alerts.append(alert)
        
        elif current_pm25 >= self.config.pm25_warning_threshold:
            alert = self._create_alert(
                user_id, station_id, "warning", "medium",
                "Air Quality Warning",
                f"PM2.5 levels are moderate ({current_pm25:.1f} μg/m³). Sensitive individuals should take precautions.",
                current_pm25, self.config.pm25_warning_threshold
            )
            alerts.append(alert)
        
        # Check forecast conditions
        if forecast_data:
            peak_pm25 = max([f.get("pm25_mean", 0) for f in forecast_data])
            peak_hour = next(i for i, f in enumerate(forecast_data) if f.get("pm25_mean") == peak_pm25)
            
            if peak_pm25 >= self.config.pm25_emergency_threshold:
                alert = self._create_alert(
                    user_id, station_id, "forecast_emergency", "high",
                    "Forecasted Emergency Conditions",
                    f"PM2.5 levels are forecasted to reach emergency levels ({peak_pm25:.1f} μg/m³) in {peak_hour} hours.",
                    peak_pm25, self.config.pm25_emergency_threshold
                )
                alerts.append(alert)
            
            elif peak_pm25 >= self.config.pm25_critical_threshold:
                alert = self._create_alert(
                    user_id, station_id, "forecast_critical", "medium",
                    "Forecasted Critical Conditions",
                    f"PM2.5 levels are forecasted to reach unhealthy levels ({peak_pm25:.1f} μg/m³) in {peak_hour} hours.",
                    peak_pm25, self.config.pm25_critical_threshold
                )
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(
        self,
        user_id: Optional[str],
        station_id: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        pm25_level: float,
        threshold: float
    ) -> Alert:
        """Create a new alert"""
        alert_id = f"alert_{station_id}_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine expiration time based on severity
        if severity == "critical":
            expires_at = datetime.now() + timedelta(hours=24)
        elif severity == "high":
            expires_at = datetime.now() + timedelta(hours=12)
        else:
            expires_at = datetime.now() + timedelta(hours=6)
        
        # Determine notification channels
        channels = []
        if user_id and user_id in self.user_preferences:
            prefs = self.user_preferences[user_id]
            if prefs.email_notifications:
                channels.append("email")
            if prefs.sms_notifications:
                channels.append("sms")
            if prefs.push_notifications:
                channels.append("push")
        else:
            # Default channels
            if self.config.email_enabled:
                channels.append("email")
            if self.config.sms_enabled:
                channels.append("sms")
            if self.config.push_enabled:
                channels.append("push")
        
        alert = Alert(
            alert_id=alert_id,
            user_id=user_id or "system",
            station_id=station_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            pm25_level=pm25_level,
            threshold=threshold,
            timestamp=datetime.now(),
            expires_at=expires_at,
            channels=channels
        )
        
        # Store active alert
        self.active_alerts[alert_id] = alert
        
        return alert
    
    async def _process_alert(self, alert: Alert) -> List[Notification]:
        """Process an alert and send notifications"""
        notifications = []
        
        # Check if user has preferences
        user_prefs = self.user_preferences.get(alert.user_id)
        if not user_prefs:
            self.logger.warning(f"No preferences found for user {alert.user_id}")
            return notifications
        
        # Check rate limits
        if not self._check_rate_limits(alert.user_id):
            self.logger.warning(f"Rate limit exceeded for user {alert.user_id}")
            return notifications
        
        # Check quiet hours
        if self._is_quiet_hours(user_prefs):
            self.logger.info(f"Quiet hours active for user {alert.user_id}, skipping notification")
            return notifications
        
        # Send notifications through each channel
        for channel in alert.channels:
            if channel == "email" and user_prefs.email_notifications:
                notification = await self._send_email_notification(alert, user_prefs)
                if notification:
                    notifications.append(notification)
            
            elif channel == "sms" and user_prefs.sms_notifications:
                notification = await self._send_sms_notification(alert, user_prefs)
                if notification:
                    notifications.append(notification)
            
            elif channel == "push" and user_prefs.push_notifications:
                notification = await self._send_push_notification(alert, user_prefs)
                if notification:
                    notifications.append(notification)
        
        # Update rate limits
        self._increment_rate_limits(alert.user_id)
        
        return notifications
    
    async def _send_email_notification(self, alert: Alert, user_prefs: UserPreferences) -> Optional[Notification]:
        """Send email notification"""
        if not user_prefs.email or not self.config.email_enabled:
            return None
        
        try:
            # Create email content
            subject = f"AirAware Alert: {alert.title}"
            content = self._create_email_content(alert)
            
            # Create notification record
            notification = Notification(
                notification_id=f"email_{alert.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=alert.user_id,
                channel="email",
                subject=subject,
                content=content,
                alert_id=alert.alert_id
            )
            
            # Send email
            if self._send_email(user_prefs.email, subject, content):
                notification.status = "sent"
                self.logger.info(f"Email notification sent to {user_prefs.email}")
            else:
                notification.status = "failed"
                notification.error_message = "Failed to send email"
                self.logger.error(f"Failed to send email to {user_prefs.email}")
            
            # Store notification
            self.notification_history.append(notification)
            return notification
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return None
    
    async def _send_sms_notification(self, alert: Alert, user_prefs: UserPreferences) -> Optional[Notification]:
        """Send SMS notification"""
        if not user_prefs.phone or not self.config.sms_enabled:
            return None
        
        try:
            # Create SMS content
            content = f"AirAware Alert: {alert.title}\n{alert.message}"
            
            # Create notification record
            notification = Notification(
                notification_id=f"sms_{alert.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=alert.user_id,
                channel="sms",
                subject="AirAware Alert",
                content=content,
                alert_id=alert.alert_id
            )
            
            # Send SMS (placeholder - would integrate with SMS service)
            if self._send_sms(user_prefs.phone, content):
                notification.status = "sent"
                self.logger.info(f"SMS notification sent to {user_prefs.phone}")
            else:
                notification.status = "failed"
                notification.error_message = "Failed to send SMS"
                self.logger.error(f"Failed to send SMS to {user_prefs.phone}")
            
            # Store notification
            self.notification_history.append(notification)
            return notification
            
        except Exception as e:
            self.logger.error(f"SMS notification failed: {e}")
            return None
    
    async def _send_push_notification(self, alert: Alert, user_prefs: UserPreferences) -> Optional[Notification]:
        """Send push notification"""
        if not user_prefs.push_token or not self.config.push_enabled:
            return None
        
        try:
            # Create push content
            content = f"AirAware Alert: {alert.title}\n{alert.message}"
            
            # Create notification record
            notification = Notification(
                notification_id=f"push_{alert.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=alert.user_id,
                channel="push",
                subject=alert.title,
                content=content,
                alert_id=alert.alert_id
            )
            
            # Send push notification (placeholder - would integrate with push service)
            if self._send_push(user_prefs.push_token, alert.title, content):
                notification.status = "sent"
                self.logger.info(f"Push notification sent to {user_prefs.push_token}")
            else:
                notification.status = "failed"
                notification.error_message = "Failed to send push notification"
                self.logger.error(f"Failed to send push notification to {user_prefs.push_token}")
            
            # Store notification
            self.notification_history.append(notification)
            return notification
            
        except Exception as e:
            self.logger.error(f"Push notification failed: {e}")
            return None
    
    def _create_email_content(self, alert: Alert) -> str:
        """Create email content for alert"""
        content = f"""
        <html>
        <body>
            <h2>{alert.title}</h2>
            <p><strong>Station:</strong> {alert.station_id}</p>
            <p><strong>PM2.5 Level:</strong> {alert.pm25_level:.1f} μg/m³</p>
            <p><strong>Threshold:</strong> {alert.threshold:.1f} μg/m³</p>
            <p><strong>Severity:</strong> {alert.severity.title()}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Message:</h3>
            <p>{alert.message}</p>
            
            <h3>Health Recommendations:</h3>
            <ul>
                <li>Stay indoors during poor air quality</li>
                <li>Use air purifiers if available</li>
                <li>Wear N95 masks when going outside</li>
                <li>Monitor air quality regularly</li>
            </ul>
            
            <p><em>This is an automated message from AirAware. Please do not reply to this email.</em></p>
        </body>
        </html>
        """
        return content
    
    def _send_email(self, to_email: str, subject: str, content: str) -> bool:
        """Send email notification"""
        try:
            if not self.config.smtp_username or not self.config.smtp_password:
                self.logger.warning("SMTP credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add HTML content
            html_part = MIMEText(content, 'html')
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _send_sms(self, phone: str, content: str) -> bool:
        """Send SMS notification (placeholder)"""
        # This would integrate with an SMS service like Twilio
        self.logger.info(f"SMS placeholder: Sending to {phone}: {content}")
        return True
    
    def _send_push(self, push_token: str, title: str, content: str) -> bool:
        """Send push notification (placeholder)"""
        # This would integrate with a push notification service
        self.logger.info(f"Push placeholder: Sending to {push_token}: {title} - {content}")
        return True
    
    def _check_rate_limits(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        if user_id not in self.rate_limits:
            return True
        
        limits = self.rate_limits[user_id]
        current_hour = datetime.now().hour
        current_day = datetime.now().day
        
        # Check hourly limit
        if limits.get("hour", 0) >= self.config.max_notifications_per_hour:
            return False
        
        # Check daily limit
        if limits.get("day", 0) >= self.config.max_notifications_per_day:
            return False
        
        return True
    
    def _increment_rate_limits(self, user_id: str):
        """Increment rate limits for user"""
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {"hour": 0, "day": 0}
        
        self.rate_limits[user_id]["hour"] += 1
        self.rate_limits[user_id]["day"] += 1
    
    def _update_rate_limits(self):
        """Update rate limits (reset hourly counters)"""
        current_hour = datetime.now().hour
        
        for user_id in self.rate_limits:
            # Reset hourly counter if hour has changed
            if "last_hour" not in self.rate_limits[user_id] or self.rate_limits[user_id]["last_hour"] != current_hour:
                self.rate_limits[user_id]["hour"] = 0
                self.rate_limits[user_id]["last_hour"] = current_hour
    
    def _is_quiet_hours(self, user_prefs: UserPreferences) -> bool:
        """Check if current time is within user's quiet hours"""
        current_hour = datetime.now().hour
        
        if user_prefs.quiet_hours_start <= user_prefs.quiet_hours_end:
            # Same day quiet hours (e.g., 22:00 to 07:00)
            return user_prefs.quiet_hours_start <= current_hour < user_prefs.quiet_hours_end
        else:
            # Overnight quiet hours (e.g., 22:00 to 07:00)
            return current_hour >= user_prefs.quiet_hours_start or current_hour < user_prefs.quiet_hours_end
    
    def _cleanup_expired_alerts(self) -> List[str]:
        """Clean up expired alerts"""
        expired_alert_ids = []
        current_time = datetime.now()
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.expires_at and alert.expires_at < current_time:
                expired_alert_ids.append(alert_id)
                del self.active_alerts[alert_id]
        
        return expired_alert_ids
    
    def _load_user_preferences(self):
        """Load user preferences from file"""
        try:
            prefs_path = Path(self.config.user_preferences_file)
            if prefs_path.exists():
                with open(prefs_path, 'r') as f:
                    data = json.load(f)
                
                self.user_preferences = {}
                for user_id, prefs_data in data.items():
                    self.user_preferences[user_id] = UserPreferences(**prefs_data)
                
                self.logger.info(f"Loaded {len(self.user_preferences)} user preferences")
            else:
                self.user_preferences = {}
                self.logger.info("No user preferences file found, starting with empty preferences")
        except Exception as e:
            self.logger.error(f"Failed to load user preferences: {e}")
            self.user_preferences = {}
    
    def _load_notification_history(self):
        """Load notification history from file"""
        try:
            history_path = Path(self.config.notification_history_file)
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                
                self.notification_history = []
                for item in data:
                    notification = Notification(
                        notification_id=item["notification_id"],
                        user_id=item["user_id"],
                        channel=item["channel"],
                        subject=item["subject"],
                        content=item["content"],
                        alert_id=item.get("alert_id"),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        status=item["status"],
                        retry_count=item.get("retry_count", 0),
                        error_message=item.get("error_message")
                    )
                    self.notification_history.append(notification)
                
                self.logger.info(f"Loaded {len(self.notification_history)} notification records")
            else:
                self.notification_history = []
                self.logger.info("No notification history file found, starting with empty history")
        except Exception as e:
            self.logger.error(f"Failed to load notification history: {e}")
            self.notification_history = []
    
    def _save_user_preferences(self):
        """Save user preferences to file"""
        try:
            prefs_path = Path(self.config.user_preferences_file)
            prefs_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {user_id: prefs.to_dict() for user_id, prefs in self.user_preferences.items()}
            
            with open(prefs_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save user preferences: {e}")
    
    def _save_notification_history(self):
        """Save notification history to file"""
        try:
            history_path = Path(self.config.notification_history_file)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [notif.to_dict() for notif in self.notification_history]
            
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save notification history: {e}")
    
    def add_user_preferences(self, user_prefs: UserPreferences):
        """Add or update user preferences"""
        self.user_preferences[user_prefs.user_id] = user_prefs
        self._save_user_preferences()
    
    def get_notification_summary(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get notification summary for a user"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent notifications
        recent_notifications = [
            n for n in self.notification_history 
            if n.user_id == user_id and n.timestamp >= cutoff_time
        ]
        
        if not recent_notifications:
            return {"status": "no_data", "message": "No recent notifications"}
        
        # Calculate summary statistics
        total_notifications = len(recent_notifications)
        successful_notifications = len([n for n in recent_notifications if n.status == "sent"])
        failed_notifications = len([n for n in recent_notifications if n.status == "failed"])
        
        # Group by channel
        channel_counts = {}
        for notif in recent_notifications:
            channel_counts[notif.channel] = channel_counts.get(notif.channel, 0) + 1
        
        return {
            "status": "success",
            "user_id": user_id,
            "time_period_hours": hours,
            "total_notifications": total_notifications,
            "successful_notifications": successful_notifications,
            "failed_notifications": failed_notifications,
            "success_rate": successful_notifications / total_notifications if total_notifications > 0 else 0,
            "channel_counts": channel_counts,
            "last_notification": max([n.timestamp for n in recent_notifications]).isoformat()
        }
