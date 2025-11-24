# 🌍 GeoGuard - Integrated Disaster Detection System

A comprehensive AI-powered disaster monitoring and detection system that combines multiple environmental hazard detection capabilities with real-time weather data analysis.

## 🎯 Features

### Three Detection Modules

1. **☀️ Heatwave Prediction**
   - ML-based temperature prediction using LightGBM
   - Real-time weather data integration
   - Heatwave risk assessment
   - Historical temperature analysis

2. **🔥 Forest Fire Detection**
   - Dual AI models: Custom CNN and MobileNetV2
   - Image-based fire detection
   - Fire Spread Likelihood Score calculation
   - Weather-based risk factors (wind, humidity, temperature)
   - Comprehensive fire risk analysis

3. **⛰️ Landslide Detection**
   - U-Net deep learning segmentation
   - Upload images OR fetch real-time satellite data
   - Sentinel-2 and DEM data integration
   - Risk heatmap visualization
   - Emergency response protocols

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment
cd d:\Mini_Project\miniproject
.\virtual_env\Scripts\Activate.ps1

# Install required packages (if not already installed)
pip install streamlit pandas numpy tensorflow joblib requests pillow fpdf python-dotenv sentinelhub
```

### 2. Configure OpenWeatherMap API Key

1. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Open `integrated_app.py`
3. Replace `YOUR_API_KEY_HERE`:
   ```python
   OPENWEATHER_API_KEY = "your_actual_api_key_here"
   ```

### 3. (Optional) Configure Sentinel Hub for Satellite Data

For real-time satellite data in the Landslide Detection module:

1. Create a free account at [Sentinel Hub](https://www.sentinel-hub.com/)
2. Create OAuth credentials
3. Create file `streamlit_app/.env`:
   ```
   SENTINELHUB_CLIENT_ID=your_client_id
   SENTINELHUB_CLIENT_SECRET=your_client_secret
   ```

### 4. Run the Application

```bash
streamlit run integrated_app.py
```

## 📁 Required Files Structure

```
miniproject/
├── integrated_app.py                          # Main application
├── models/
│   ├── best_regressor_LightGBM.joblib        # Heatwave prediction model
│   ├── imputer.joblib                         # Data imputer
│   └── scaler.joblib                          # Feature scaler
├── forestfire/
│   └── mobilenet_fire_final.h5               # Fire detection model (MobileNetV2)
├── fire_detection_cnnn_final.h5              # Fire detection model (CNN)
└── streamlit_app/
    ├── best_model.keras                       # Landslide detection model (U-Net)
    └── .env                                   # Sentinel Hub credentials (optional)
```

## 🎨 User Interface

### Home Page
- Overview of all three modules
- Key features and capabilities
- Quick statistics dashboard
- Getting started guide

### Heatwave Prediction
- Live weather data widget
- Auto-populated parameters from weather API
- Manual parameter adjustment
- Temperature prediction with risk assessment
- PDF report generation

### Forest Fire Detection
- **Fire Spread Likelihood Score** (0-100)
  - Wind factor: Higher wind = faster spread
  - Humidity factor: Lower humidity = easier ignition
  - Temperature factor: Higher temp = increased risk
- Model selection (CNN or MobileNetV2)
- Image upload and analysis
- Comprehensive detection reports with weather data
- PDF and text report downloads

### Landslide Detection
- **Two data input methods:**
  1. Upload satellite/aerial images
  2. Fetch real-time Sentinel-2 satellite data
- Risk heatmap visualization
- Detection overlay display
- Emergency response protocols
- SMS and mapping integration
- PDF and text report generation
- Alert history tracking

## 📊 Fire Spread Risk Calculation

The system calculates a **Fire Spread Likelihood Score** based on:

```
Score = Wind Factor (40 pts) + Humidity Factor (35 pts) + Temperature Factor (25 pts)

Risk Levels:
- 70-100: CRITICAL (Red)
- 50-69:  HIGH (Orange)
- 30-49:  MODERATE (Yellow)
- 0-29:   LOW (Green)
```

### Factors:
- **Wind**: Higher wind speeds increase fire spread rate
- **Humidity**: Lower humidity makes vegetation more flammable
- **Temperature**: Higher temperatures dry out fuel sources

## 📄 Report Generation

All three modules support comprehensive report generation:

### PDF Reports Include:
- Detection timestamp and results
- Input parameters and weather conditions
- Risk assessment and scores
- Recommendations and action items
- Emergency contact information (landslide)

### Text Reports Include:
- All PDF content in plain text format
- Easy to share via email/SMS
- Compatible with all systems

## 🌐 Live Weather Integration

- Sidebar weather widget
- Auto-fetches: Temperature, Humidity, Wind Speed & Direction, Cloud Cover
- Auto-populates prediction parameters
- Used for fire spread risk calculation
- City-based location search

## 🚨 Emergency Features

### Landslide Alerts:
- Severity classification (Critical/High/Moderate)
- Emergency contact numbers
- Google Maps integration
- SMS alert message generation
- Evacuation recommendations

### Fire Detection:
- Immediate fire detection alerts
- Spread risk warnings
- Weather-based precautions

### Heatwave Warnings:
- Temperature threshold alerts (>37°C)
- Risk classification
- Safety recommendations

## 🔧 Troubleshooting

### Common Issues:

1. **Weather data not loading:**
   - Check API key is configured correctly
   - Verify internet connection
   - Try a different city name

2. **Models not loading:**
   - Verify all model files are in correct locations
   - Check file paths match directory structure
   - Ensure TensorFlow version compatibility

3. **Satellite data fetch fails:**
   - Configure Sentinel Hub credentials in `.env`
   - Check coordinates are valid (latitude: -90 to 90, longitude: -180 to 180)
   - Verify internet connection

4. **PDF generation errors:**
   - Ensure `fpdf` is installed: `pip install fpdf`
   - Check file write permissions

## 📈 Usage Workflow

1. **Start Application**
   ```bash
   streamlit run integrated_app.py
   ```

2. **Fetch Weather Data** (Sidebar)
   - Enter city name
   - Click "Fetch Weather Data"
   - View current conditions

3. **Select Module** (Sidebar)
   - Choose disaster type to monitor

4. **Run Analysis**
   - Upload image or fetch satellite data
   - Adjust parameters if needed
   - Click analyze button

5. **Review Results**
   - Check risk scores and alerts
   - View visualizations
   - Read recommendations

6. **Generate Reports**
   - Download PDF reports
   - Download text reports
   - Share with authorities

## 🎓 Technical Details

### Machine Learning Models:
- **Heatwave**: LightGBM Regressor with 19 features
- **Fire Detection**: 
  - Custom CNN (128x128 input)
  - MobileNetV2 (224x224 input)
- **Landslide**: U-Net Segmentation (128x128x6 input)

### Data Sources:
- **OpenWeatherMap API**: Real-time weather data
- **Sentinel Hub**: Satellite imagery (Sentinel-2, DEM)
- **User Uploads**: Custom images

### Technologies:
- **Framework**: Streamlit
- **ML/DL**: TensorFlow, Keras, scikit-learn
- **Data Processing**: NumPy, Pandas
- **Image Processing**: PIL
- **Geospatial**: SentinelHub API

## 📞 Support

For issues or questions:
1. Check model files are present
2. Verify API keys are configured
3. Review console for error messages
4. Ensure all dependencies are installed

## 🌟 Credits

Developed as an integrated disaster detection and monitoring system combining multiple AI-powered environmental hazard detection capabilities.

---

**⚠️ Important**: This system is for monitoring and early warning purposes. Always contact local emergency services (911, 112, 100) for actual emergencies.
