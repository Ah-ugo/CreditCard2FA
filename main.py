from logging import exception

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson import ObjectId
import pyotp
import qrcode
import io
import base64
import os
import random
from fastapi.responses import StreamingResponse
import csv
import uuid
import numpy as np
from faker import Faker
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Initialize Faker for realistic data generation
fake = Faker()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["fraud_detection"]

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3400

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Fraud detection model setup
FRAUD_MODEL_PATH = "fraud_detection_model.joblib"


def initialize_fraud_model():
    """Initialize or load the fraud detection model"""
    if Path(FRAUD_MODEL_PATH).exists():
        return joblib.load(FRAUD_MODEL_PATH)
    else:
        # Create a simple Isolation Forest model as a placeholder
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        joblib.dump(model, FRAUD_MODEL_PATH)
        return model


fraud_model = initialize_fraud_model()

app = FastAPI(title="Credit Card Fraud Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: str
    is_2fa_enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    requires_2fa: bool = False


class TokenData(BaseModel):
    email: Optional[str] = None
    is_2fa_verified: bool = False


class TwoFactorSetup(BaseModel):
    qr_code: str
    secret: str


class TwoFactorVerify(BaseModel):
    code: str
    secret: Optional[str] = None


class TransactionBase(BaseModel):
    amount: float
    merchant: str
    location: str
    card_number: str = Field(..., pattern="^[0-9]{16}$")
    cvv: str = Field(..., pattern="^[0-9]{3,4}$")
    expiry_date: str = Field(..., pattern="^(0[1-9]|1[0-2])/[0-9]{2}$")


class TransactionCreate(TransactionBase):
    pass


class Transaction(BaseModel):
    id: str
    amount: float
    merchant: str
    location: str
    timestamp: datetime
    is_fraudulent: bool
    risk_score: int
    user_id: str

    class Config:
        from_attributes = True


class PaginatedTransactions(BaseModel):
    items: List[Transaction]
    total: int
    page: int
    size: int
    total_pages: int


class DashboardStats(BaseModel):
    totalTransactions: int
    fraudulentTransactions: int
    totalAmount: float
    riskScore: int


# Helper functions
def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(email):
    user_data = db.users.find_one({"email": email})
    if user_data:
        user_data["id"] = str(user_data["_id"])
        return User(**user_data)
    return None


def authenticate_user(email, password):
    user_data = db.users.find_one({"email": email})
    if not user_data:
        return False
    if not verify_password(password, user_data["password"]):
        return False
    user_data["id"] = str(user_data["_id"])
    return User(**user_data)


# Enhanced helper functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    # Ensure is_2fa_verified is always included
    if "is_2fa_verified" not in to_encode:
        to_encode["is_2fa_verified"] = False
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception

        # Get user from database to check 2FA status
        user = get_user(email)
        if user is None:
            raise credentials_exception
            # If 2FA is enabled but token isn't verified
        if user.is_2fa_enabled and not payload.get("is_2fa_verified", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Two-factor authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user
    except JWTError:
        raise credentials_exception


# Enhanced fraud detection algorithm
def detect_fraud(transaction: TransactionCreate, user_id: str) -> tuple:
    """Enhanced fraud detection with machine learning and rule-based checks"""
    # Get user's transaction history
    user_transactions = list(db.transactions.find({"user_id": user_id}).sort("timestamp", -1).limit(100))

    # Feature engineering for ML model
    features = {
        'amount': transaction.amount,
        'hour_of_day': datetime.utcnow().hour,
        'day_of_week': datetime.utcnow().weekday(),
        'avg_user_amount': 0,
        'location_change': 0,
        'frequency': 0,
        'amount_diff': 0,
        'merchant_risk': random.uniform(0, 1)  # Placeholder - would come from merchant risk DB
    }

    # Calculate historical features if user has previous transactions
    if user_transactions:
        # Calculate average transaction amount
        avg_amount = sum(t['amount'] for t in user_transactions) / len(user_transactions)
        features['avg_user_amount'] = avg_amount
        features['amount_diff'] = abs(transaction.amount - avg_amount) / avg_amount if avg_amount > 0 else 0

        # Check for location changes
        last_location = user_transactions[0]['location']
        features['location_change'] = 1 if transaction.location != last_location else 0

        # Check transaction frequency (transactions per hour)
        recent_hours = 6
        recent_count = sum(1 for t in user_transactions
                           if (datetime.utcnow() - t['timestamp']).total_seconds() < recent_hours * 3600)
        features['frequency'] = recent_count / recent_hours

    # Convert features to array for ML model
    feature_values = np.array([[features['amount'],
                                features['amount_diff'],
                                features['frequency'],
                                features['location_change'],
                                features['merchant_risk']]])

    # Get ML prediction (-1 for outlier/anomaly, 1 for normal)
    ml_score = fraud_model.decision_function(feature_values)[0]
    ml_prediction = fraud_model.predict(feature_values)[0]

    # Rule-based checks
    risk_score = 0

    # Amount-based rules
    if transaction.amount > 5000:
        risk_score += 30  # Very large transaction
    elif transaction.amount > 1000:
        risk_score += 15
    elif transaction.amount < 1:
        risk_score += 10  # Micro transactions can be suspicious

    # Location-based rules
    if features['location_change'] == 1:
        risk_score += 20

    # Frequency-based rules
    if features['frequency'] > 5:  # More than 5 transactions per hour
        risk_score += 25

    # Time-based rules (late night/early morning)
    if 0 <= features['hour_of_day'] <= 4:
        risk_score += 15

    # Merchant-based rules
    if features['merchant_risk'] > 0.8:
        risk_score += 30

    # Combine ML score with rule-based score (ML score ranges typically -0.5 to 0.5)
    ml_contribution = (0.5 - ml_score) * 100  # Convert to 0-100 scale
    combined_score = (risk_score * 0.4) + (ml_contribution * 0.6)

    # Cap at 100
    final_score = min(100, combined_score)

    # Determine if fraudulent (threshold at 75)
    is_fraudulent = final_score >= 75

    return is_fraudulent, int(final_score)


# Routes
@app.post("/api/auth/register", response_model=Token)
async def register_user(user: UserCreate):
    db_user = db.users.find_one({"email": user.email})
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    user_data = user.dict()
    user_data["password"] = hashed_password
    user_data["is_2fa_enabled"] = False
    user_data["created_at"] = datetime.utcnow()

    result = db.users.insert_one(user_data)
    user_id = str(result.inserted_id)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "is_2fa_verified": False},  # Explicitly set to False
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "requires_2fa": user.is_2fa_enabled
    }

@app.get("/api/auth/setup-2fa", response_model=TwoFactorSetup)
async def setup_2fa(current_user: User = Depends(get_current_user)):
    # Generate a random secret key
    secret = pyotp.random_base32()

    # Create a TOTP object
    totp = pyotp.TOTP(secret)

    # Generate the provisioning URI for the QR code
    provisioning_uri = totp.provisioning_uri(
        name=current_user.email,
        issuer_name="SecureCard Fraud Detection"
    )

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(provisioning_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert image to base64 string
    buffered = io.BytesIO()
    img.save(buffered)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Return the QR code as an HTML img tag and the secret
    return {
        "qr_code": f'<img src="data:image/png;base64,{img_str}" alt="2FA QR Code" />',
        "secret": secret
    }


@app.post("/api/auth/verify-2fa-setup")
async def verify_2fa_setup(verify_data: TwoFactorVerify, current_user: User = Depends(get_current_user)):
    # Create a TOTP object with the secret
    totp = pyotp.TOTP(verify_data.secret)

    # Verify the code
    if not totp.verify(verify_data.code):
        raise HTTPException(status_code=400, detail="Invalid verification code")

    # Enable 2FA for the user
    db.users.update_one(
        {"email": current_user.email},
        {"$set": {"is_2fa_enabled": True, "totp_secret": verify_data.secret}}
    )

    # Create a new token with 2FA verified
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.email, "is_2fa_verified": True},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/auth/verify-2fa")
async def verify_2fa(
    verify_data: TwoFactorVerify,
    token: str = Depends(oauth2_scheme)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_data = db.users.find_one({"email": email})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

        # Use the secret from the request if provided (for setup), otherwise from DB
    secret = verify_data.secret if verify_data.secret else user_data.get("totp_secret")
    if not secret:
        raise HTTPException(status_code=400, detail="2FA not configured")

    totp = pyotp.TOTP(secret)
    if not totp.verify(verify_data.code):
        raise HTTPException(status_code=400, detail="Invalid verification code")

    # Create new token with verified status
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email, "is_2fa_verified": True},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/api/transactions", response_model=Transaction)
async def create_transaction(transaction: TransactionCreate, current_user: User = Depends(get_current_user)):
    user_id = current_user.id

    # Detect if transaction is fraudulent
    is_fraudulent, risk_score = detect_fraud(transaction, user_id)

    # Create transaction object
    transaction_data = {
        "amount": transaction.amount,
        "merchant": transaction.merchant,
        "location": transaction.location,
        "timestamp": datetime.utcnow(),
        "is_fraudulent": is_fraudulent,
        "risk_score": risk_score,
        "user_id": user_id
    }

    # Insert into database
    result = db.transactions.insert_one(transaction_data)
    transaction_id = str(result.inserted_id)

    # Return transaction with ID
    return {
        "id": transaction_id,
        **transaction_data
    }


@app.get("/api/transactions", response_model=PaginatedTransactions)
async def get_transactions(
        page: int = 1,
        size: int = 10,
        filter: str = "all",
        search: str = "",
        current_user: User = Depends(get_current_user)
):
    user_id = current_user.id

    # Build query
    query = {"user_id": user_id}

    # Apply filter
    if filter == "fraudulent":
        query["is_fraudulent"] = True
    elif filter == "legitimate":
        query["is_fraudulent"] = False

    # Apply search
    if search:
        query["$or"] = [
            {"merchant": {"$regex": search, "$options": "i"}},
            {"amount": {"$regex": search, "$options": "i"}} if search.replace(".", "", 1).isdigit() else {"amount": 0}
        ]

    # Count total matching documents
    total = db.transactions.count_documents(query)

    # Calculate pagination
    total_pages = (total + size - 1) // size
    skip = (page - 1) * size

    # Get paginated transactions
    transactions = list(
        db.transactions.find(query)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(size)
    )

    # Convert ObjectId to string
    for transaction in transactions:
        transaction["id"] = str(transaction["_id"])
        del transaction["_id"]

    return {
        "items": transactions,
        "total": total,
        "page": page,
        "size": size,
        "total_pages": total_pages
    }


@app.get("/api/transactions/recent", response_model=List[Transaction])
async def get_recent_transactions(current_user: User = Depends(get_current_user)):
    user_id = current_user.id

    # Get recent transactions
    transactions = list(
        db.transactions.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(5)
    )

    # Convert ObjectId to string
    for transaction in transactions:
        transaction["id"] = str(transaction["_id"])
        del transaction["_id"]

    return transactions


@app.get("/api/transactions/export")
async def export_transactions(
        filter: str = "all",
        search: str = "",
        current_user: User = Depends(get_current_user)
):
    user_id = current_user.id

    # Build query
    query = {"user_id": user_id}

    # Apply filter
    if filter == "fraudulent":
        query["is_fraudulent"] = True
    elif filter == "legitimate":
        query["is_fraudulent"] = False

    # Apply search
    if search:
        query["$or"] = [
            {"merchant": {"$regex": search, "$options": "i"}},
            {"amount": {"$regex": search, "$options": "i"}} if search.replace(".", "", 1).isdigit() else {"amount": 0}
        ]

    # Get transactions
    transactions = list(
        db.transactions.find(query)
        .sort("timestamp", -1)
    )

    # Convert ObjectId to string
    for transaction in transactions:
        transaction["id"] = str(transaction["_id"])
        del transaction["_id"]

    # Create CSV file
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Transaction ID",
        "Date",
        "Amount",
        "Merchant",
        "Location",
        "Status",
        "Risk Score"
    ])

    # Write data
    for transaction in transactions:
        writer.writerow([
            transaction["id"],
            transaction["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            f"${transaction['amount']:.2f}",
            transaction["merchant"],
            transaction["location"],
            "Fraudulent" if transaction["is_fraudulent"] else "Legitimate",
            f"{transaction['risk_score']}%"
        ])

    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=transactions.csv"

    return response


@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    user_id = current_user.id

    # Get total transactions
    total_transactions = db.transactions.count_documents({"user_id": user_id})

    # Get fraudulent transactions
    fraudulent_transactions = db.transactions.count_documents({
        "user_id": user_id,
        "is_fraudulent": True
    })

    # Get total amount
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": None, "total": {"$sum": "$amount"}}}
    ]
    result = list(db.transactions.aggregate(pipeline))
    total_amount = result[0]["total"] if result else 0

    # Calculate risk score (average of all transaction risk scores)
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": None, "avg_risk": {"$avg": "$risk_score"}}}
    ]
    result = list(db.transactions.aggregate(pipeline))
    risk_score = int(result[0]["avg_risk"]) if result else 0

    return {
        "totalTransactions": total_transactions,
        "fraudulentTransactions": fraudulent_transactions,
        "totalAmount": total_amount,
        "riskScore": risk_score
    }


# Enhanced seed data function
@app.post("/api/seed-data")
async def seed_data(current_user: User = Depends(get_current_user)):
    user_id = current_user.id

    # Check if user already has transactions
    if db.transactions.count_documents({"user_id": user_id}) > 0:
        return {"message": "Data already seeded for this user"}

    # Generate realistic merchant categories with different risk profiles
    merchant_categories = {
        "retail": {
            "names": ["Walmart", "Target", "Best Buy", "Home Depot", "Costco"],
            "risk_factor": 0.1
        },
        "tech": {
            "names": ["Apple Store", "Microsoft", "Amazon", "Google", "Dell"],
            "risk_factor": 0.2
        },
        "food": {
            "names": ["McDonald's", "Starbucks", "Subway", "Chipotle", "Pizza Hut"],
            "risk_factor": 0.05
        },
        "travel": {
            "names": ["Expedia", "Booking.com", "Delta", "United", "Airbnb"],
            "risk_factor": 0.3
        },
        "entertainment": {
            "names": ["Netflix", "Spotify", "AMC Theaters", "Disney+", "HBO Max"],
            "risk_factor": 0.15
        },
        "high_risk": {
            "names": ["Online Casino", "Crypto Exchange", "Foreign Marketplace", "Wire Transfer", "Auction Site"],
            "risk_factor": 0.7
        }
    }

    # Generate realistic locations based on user patterns
    def generate_locations(base_city=None, num_locations=5):
        if not base_city:
            base_city = fake.city() + ", " + fake.state_abbr()

        locations = [base_city]
        for _ in range(num_locations - 1):
            if random.random() < 0.7:  # 70% chance to stay in same state
                locations.append(fake.city() + ", " + base_city.split(", ")[1])
            else:
                locations.append(fake.city() + ", " + fake.state_abbr())
        return locations

    base_city = fake.city() + ", " + fake.state_abbr()
    locations = generate_locations(base_city)

    # Generate transactions with realistic patterns
    transactions = []
    for i in range(100):  # Generate 100 transactions
        # Choose a merchant category weighted by risk
        category = random.choices(
            list(merchant_categories.keys()),
            weights=[0.3, 0.2, 0.25, 0.1, 0.1, 0.05]  # Weighted probabilities
        )[0]

        merchant = random.choice(merchant_categories[category]["names"])
        base_risk = merchant_categories[category]["risk_factor"]

        # Generate amount based on merchant category
        if category == "retail":
            amount = round(random.uniform(10, 500), 2)
        elif category == "tech":
            amount = round(random.uniform(50, 2000), 2)
        elif category == "food":
            amount = round(random.uniform(5, 50), 2)
        elif category == "travel":
            amount = round(random.uniform(100, 5000), 2)
        elif category == "entertainment":
            amount = round(random.uniform(10, 30), 2)
        else:  # high_risk
            amount = round(random.uniform(100, 10000), 2)

        # 80% chance to use home location, 20% for other locations
        location = base_city if random.random() < 0.8 else random.choice(locations[1:])

        # Create timestamp with realistic patterns (clustering during daytime)
        day_offset = random.randint(0, 30)
        if random.random() < 0.7:  # 70% chance daytime transaction (9am-9pm)
            hour = random.randint(9, 21)
        else:
            hour = random.randint(0, 23)

        minute = random.randint(0, 59)
        timestamp = datetime.utcnow() - timedelta(days=day_offset, hours=hour, minutes=minute)

        # Determine if fraudulent (based on merchant risk and amount)
        is_fraudulent = False
        risk_score = random.randint(0, 100)

        # High risk merchants have higher chance of fraud
        if base_risk > 0.5 and random.random() < 0.4:
            is_fraudulent = True
            risk_score = random.randint(70, 100)
        elif amount > 3000 and random.random() < 0.3:
            is_fraudulent = True
            risk_score = random.randint(70, 100)
        elif location != base_city and random.random() < 0.2:
            is_fraudulent = True
            risk_score = random.randint(70, 100)
        else:
            risk_score = random.randint(0, 69)

        transactions.append({
            "amount": amount,
            "merchant": merchant,
            "location": location,
            "timestamp": timestamp,
            "is_fraudulent": is_fraudulent,
            "risk_score": risk_score,
            "user_id": user_id,
            "_id": ObjectId(),
            "category": category  # Adding category for potential analytics
        })

    # Insert transactions
    if transactions:
        db.transactions.insert_many(transactions)

    # Update the fraud model with the new data
    X = np.array([[t['amount'], t['risk_score'] / 100, 1 if t['is_fraudulent'] else 0]
                  for t in transactions])
    fraud_model.fit(X)
    joblib.dump(fraud_model, FRAUD_MODEL_PATH)

    return {"message": f"Successfully seeded {len(transactions)} realistic transactions"}