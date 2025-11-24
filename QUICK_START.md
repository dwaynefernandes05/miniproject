# 🚀 Quick Start Guide - EcoCast

## ⚡ 3-Minute Setup

### Step 1: Configure API Key (Required)
Open `integrated_app.py` and find line 55:
```python
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"
```
Replace with your actual key from [openweathermap.org](https://openweathermap.org/api)

### Step 2: Run the App
```bash
# Make sure you're in the project directory
cd d:\Mini_Project\miniproject

# Activate virtual environment (if not already active)
.\virtual_env\Scripts\Activate.ps1

# Run the application
streamlit run integrated_app.py
```

### Step 3: Use the System

#### 🌤️ Get Live Weather Data
1. In the **sidebar**, enter a city name (e.g., "Mumbai")
2. Click **"🔄 Fetch Weather Data"**
3. View current conditions in the sidebar

#### ☀️ Heatwave Prediction
1. Select **"☀️ Heatwave Prediction"** from sidebar
2. Weather data auto-populates parameters
3. Adjust any parameters if needed
4. Click **"🌡️ Predict Next-Day Temperature"**
5. Download PDF report

#### 🔥 Forest Fire Detection
1. Select **"🔥 Forest Fire Detection"** from sidebar
2. View **Fire Spread Likelihood Score** (if weather data loaded)
3. Choose model: MobileNetV2 (satellite) or CNN (ground/aerial)
4. Upload an image
5. Click **"🔍 Detect Fire"**
6. Download comprehensive PDF/Text report

#### ⛰️ Landslide Detection
1. Select **"⛰️ Landslide Detection"** from sidebar
2. Choose data source:
   - **📤 Upload Image**: Upload your own satellite/aerial photo
   - **🛰️ Real-time Satellite**: Fetch live Sentinel-2 data (requires Sentinel Hub credentials)
3. Enter coordinates and location name
4. Click **"🔍 Analyze for Landslide Risk"**
5. View risk heatmap and detection overlay
6. Download PDF/Text report

## 🎯 Key Features You'll Love

### 🌐 Live Weather Integration
- Real-time temperature, humidity, wind speed
- Auto-populated parameters
- Fire spread risk calculation

### 🔥 Fire Spread Score
Automatically calculated from:
- **Wind Speed**: Higher = faster spread
- **Humidity**: Lower = easier ignition  
- **Temperature**: Higher = more risk

Displays as: **Score/100** with risk level (LOW, MODERATE, HIGH, CRITICAL)

### 📄 Comprehensive Reports
Every module generates:
- **PDF Reports**: Professional formatted documents
- **Text Reports**: Plain text for easy sharing
- Weather conditions and analysis
- Risk scores and recommendations
- Emergency contacts (landslide)

### 🛰️ Satellite Data (Landslide Only)
- Fetch live Sentinel-2 imagery
- Digital Elevation Model (DEM) data
- True-color RGB visualization
- Risk heatmap overlay

## 📊 Understanding Results

### Heatwave:
- **> 37°C**: Heatwave warning (RED)
- **≤ 37°C**: Normal conditions (GREEN)

### Fire Detection:
- **≥ 50% probability**: Fire detected (RED)
- **< 50% probability**: No fire (GREEN)
- Plus spread risk score based on weather

### Landslide:
- **Coverage ≥ 0.5%**: Alert triggered
- **Severity levels**: Critical (>5%), High (>1%), Moderate (<1%)

## 🎨 UI Navigation

```
Sidebar (Left)
├── 🧭 Navigation Menu
│   ├── 🏠 Home (Overview)
│   ├── ☀️ Heatwave Prediction
│   ├── 🔥 Forest Fire Detection
│   └── ⛰️ Landslide Detection
└── 🌤️ Live Weather Data
    ├── City input
    ├── Fetch button
    └── Current conditions

Main Area (Right)
├── Module-specific interface
├── Results and visualizations
├── Reports section
└── Download buttons
```

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Weather not loading | Check API key, try different city |
| Model error | Verify all model files are present |
| Satellite fetch fails | Configure Sentinel Hub in `.env` |
| PDF won't download | Ensure `fpdf` installed |

## 💡 Pro Tips

1. **Fetch weather first** - It auto-populates many parameters
2. **Use MobileNetV2** - Better for satellite imagery
3. **Try both landslide modes** - Upload for quick check, Satellite for detailed analysis
4. **Download reports** - Share with authorities or keep records
5. **Check fire spread score** - High score = immediate action needed

## 📱 Emergency Numbers

In case of actual emergencies:
- **US**: 911
- **EU**: 112  
- **India**: 100 (Police), 102 (Ambulance)

---

**Ready to start?** Run `streamlit run integrated_app.py` and open the app in your browser! 🚀
