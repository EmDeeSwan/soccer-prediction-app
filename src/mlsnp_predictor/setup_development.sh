#!/bin/bash
echo "🚀 Setting up MLS Next Pro Predictor with Python 3.12 + Railway Database"
echo "========================================================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Current Python version: $python_version"

# Verify Python 3.12 compatibility
if [[ "$python_version" =~ "Python 3.12" ]]; then
    echo "✅ Python 3.12 detected - proceeding with AutoGluon setup"
elif [[ "$python_version" =~ "Python 3.11" ]]; then
    echo "⚠️  Python 3.11 detected - AutoGluon should work, but 3.12 is recommended"
else
    echo "⚠️  Python version may not be optimal for AutoGluon. Recommend Python 3.11+"
fi

# 1. Create .env file with Railway connection
echo ""
echo "📝 Creating .env file with Railway database connection..."
cat > .env << EOL
# Railway Database Configuration
DATABASE_URL=postgresql://postgres:<your_password_here>@<your_host>.proxy.rlwy.net:<port>/railway

# Authentication Keys (generate secure keys for production)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Environment
ENVIRONMENT=development
RAILWAY_ENVIRONMENT=development

# Optional OAuth (leave blank for now, configure later if needed)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=

# AutoML Configuration
USE_AUTOML=true
MODEL_CACHE_DIR=models/
EOL

# 2. Create necessary directories
echo "📁 Creating required directories..."
mkdir -p output/archive
mkdir -p models
mkdir -p data
mkdir -p logs

# 3. Install core dependencies
echo ""
echo "📦 Installing core Python dependencies..."
pip install -r requirements.txt

# 4. Install AutoML dependencies with fallback options
echo ""
echo "🤖 Installing AutoML libraries compatible with Python 3.12..."

# Primary option: AutoGluon (best for Python 3.12)
echo "Installing AutoGluon (primary ML library)..."
pip install autogluon

# Check if AutoGluon installed successfully
autogluon_status=$?

if [ $autogluon_status -eq 0 ]; then
    echo "✅ AutoGluon installed successfully!"
else
    echo "❌ AutoGluon installation failed, installing fallback libraries..."
    # Fallback: Sklearn + supporting libraries
    pip install scikit-learn optuna lightgbm xgboost catboost
    echo "✅ Fallback ML libraries installed (sklearn, optuna, lightgbm, xgboost, catboost)"
fi

# 5. Install additional ML/data science libraries
echo ""
echo "📊 Installing additional data science libraries..."
pip install \
    rapidfuzz \
    plotly \
    seaborn \
    joblib \
    python-dateutil

# 6. Test all installations
echo ""
echo "🧪 Testing installations..."
python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")
print("=" * 50)

# Test AutoML libraries
try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon installed successfully!")
    automl_available = True
except ImportError:
    print("❌ AutoGluon not available")
    automl_available = False

if not automl_available:
    try:
        import sklearn
        import optuna
        import lightgbm
        import xgboost
        print("✅ Sklearn + ML libraries installed as fallback!")
    except ImportError as e:
        print(f"❌ Fallback ML libraries missing: {e}")

# Test core dependencies
try:
    import pandas
    import numpy
    import fastapi
    import databases
    import asyncpg
    import bcrypt
    import jwt
    print("✅ Core API dependencies installed!")
except ImportError as e:
    print(f"❌ Missing core dependency: {e}")

# Test soccer-specific libraries
try:
    import itscalledsoccer
    import rapidfuzz
    print("✅ Soccer analytics libraries installed!")
except ImportError as e:
    print(f"❌ Missing soccer library: {e}")

# Test additional data science libraries
try:
    import plotly
    import seaborn
    print("✅ Visualization libraries installed!")
except ImportError as e:
    print(f"⚠️  Some visualization libraries missing: {e}")
EOF

# 7. Test Railway database connection
echo ""
echo "🔍 Testing Railway database connection..."
python3 << 'EOF'
import os
import asyncpg
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_connection():
    try:
        # Get connection string
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            print("❌ DATABASE_URL not found in .env file!")
            print("   Please update .env with your Railway database credentials")
            return
            
        if '<your_password_here>' in db_url or '<your_host>' in db_url:
            print("❌ Please update .env with actual Railway database credentials!")
            print("   Current DATABASE_URL contains placeholder values")
            return
        
        # Test connection
        print("Attempting to connect to Railway PostgreSQL...")
        conn = await asyncpg.connect(db_url)
        version = await conn.fetchval('SELECT version()')
        
        # Test basic query
        current_time = await conn.fetchval('SELECT NOW()')
        await conn.close()
        
        print("✅ Successfully connected to Railway PostgreSQL!")
        print(f"   Database version: {version.split(',')[0]}")
        print(f"   Current server time: {current_time}")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Troubleshooting tips:")
        print("   1. Check your DATABASE_URL in .env file")
        print("   2. Ensure your Railway database is running")
        print("   3. Verify network connectivity")
        print("   4. Check if your IP is whitelisted (if required)")

asyncio.run(test_connection())
EOF

# 8. Create sample data file if it doesn't exist
echo ""
echo "📄 Checking for team data file..."
if [ ! -f "data/asa_mls_next_pro_teams.json" ]; then
    echo "⚠️  Team data file not found. The scraper will need this file."
    echo "   Sample team data should be available in the project repository."
else
    echo "✅ Team data file found!"
fi

# 9. Final setup summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Installation Summary:"
echo "   • Python dependencies: Installed"
echo "   • AutoML libraries: $([ $autogluon_status -eq 0 ] && echo "AutoGluon (primary)" || echo "Sklearn + extras (fallback)")"
echo "   • Database configuration: Ready for Railway"
echo "   • Directory structure: Created"
echo ""
echo "📋 Next Steps:"
echo "1. Update your .env file with actual Railway database credentials:"
echo "   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@YOUR_HOST.proxy.rlwy.net:PORT/railway"
echo ""
echo "2. Test the database connection:"
echo "   python3 -c \"import asyncio; from src.common.database import connect; asyncio.run(connect())\""
echo ""
echo "3. Start the development server:"
echo "   uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "4. Access the API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "5. Load historical data (admin required):"
echo "   POST /data/load-historical/2025"
echo ""
echo "🔧 Configuration Notes:"
echo "   • AutoML models will be saved to: models/"
echo "   • Logs and output files: output/"
echo "   • Team data should be in: data/asa_mls_next_pro_teams.json"
echo ""
echo "Ready to predict!"
