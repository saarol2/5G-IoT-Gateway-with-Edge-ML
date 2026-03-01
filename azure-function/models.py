from datetime import datetime, timezone
from database import db

class Gateway(db.Model):
    __tablename__ = "gateways"

    id = db.Column(db.Integer, primary_key=True)
    gateway_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    devices = db.relationship("Device", back_populates="gateway", cascade="all, delete-orphan")

class Device(db.Model):
    __tablename__ = "devices"

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    gateway_id = db.Column(db.Integer, db.ForeignKey("gateways.id", ondelete="CASCADE"), nullable=False)

    gateway = db.relationship("Gateway", back_populates="devices")

    readings = db.relationship("Reading", back_populates="device", cascade="all, delete-orphan")

    predictions = db.relationship("Prediction", back_populates="device", cascade="all, delete-orphan")

class Reading(db.Model):
    __tablename__ = "readings"

    id = db.Column(db.Integer, primary_key=True)

    device_id = db.Column(db.Integer, db.ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)

    pc1 = db.Column(db.Float, nullable=False)
    pc2 = db.Column(db.Float, nullable=False)

    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    device = db.relationship("Device", back_populates="readings")

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)

    device_id = db.Column(db.Integer, db.ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)

    probability = db.Column(db.Float, nullable=False)
    anomaly = db.Column(db.Boolean, nullable=False)

    # source being edge or cloud
    source = db.Column(db.String(16), nullable=False)

    timestamp = db.Column(db.DateTime, nullable=False)

    device = db.relationship("Device", back_populates="predictions")