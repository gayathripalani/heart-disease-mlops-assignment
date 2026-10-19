from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PatientFeatures(BaseModel):
  age: int = Field(ge=18, le=120, description="Age in years")
  sex: int = Field(ge=0, le=1, description="1 = male, 0 = female")
  cp: int = Field(ge=1, le=4, description="Chest pain type")
  trestbps: int = Field(ge=50, le=300, description="Resting blood pressure (mm Hg)")
  chol: int = Field(ge=50, le=700, description="Serum cholesterol (mg/dl)")
  fbs: int = Field(ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
  restecg: int = Field(ge=0, le=2, description="Resting ECG result")
  thalach: int = Field(ge=40, le=250, description="Max heart rate achieved")
  exang: int = Field(ge=0, le=1, description="Exercise induced angina")
  oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
  slope: int = Field(ge=1, le=3, description="Slope of peak exercise ST segment")
  ca: Optional[float] = Field(default=None, ge=0, le=3, description="Major vessels (0-3)")
  thal: Optional[float] = Field(default=None, description="3 normal, 6 fixed, 7 reversible")

  @field_validator("thal")
  @classmethod
  def _thal_values(cls, v):
    if v is not None and v not in (3, 6, 7):
      raise ValueError("thal must be 3, 6 or 7")
    return v


class Prediction(BaseModel):
  prediction: int
  confidence: float
