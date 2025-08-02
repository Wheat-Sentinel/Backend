# Wheat Disease Detection API

A Node.js Express API backend for the Wheat Disease Detection system. This API handles user authentication, detection data storage, notifications, and integration with the Python inference engine.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory (one level up) with:
   ```env
   MONGODB_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   NODE_ENV=development
   PORT=3000
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Start production server:**
   ```bash
   npm start
   ```

## 📁 Project Structure

```
api/
├── server.js              # Main Express server
├── package.json           # Dependencies and scripts
├── vercel.json           # Vercel deployment config
│
├── config/
│   └── database.js       # MongoDB connection setup
│
├── middleware/
│   └── auth.js          # JWT authentication middleware
│
├── models/
│   ├── User.js          # User data model
│   └── Detection.js     # Detection result model
│
└── routes/
    ├── auth.js          # Authentication endpoints
    ├── detections.js    # Detection CRUD operations
    └── notifications.js # Push notification handling
```

## 🔌 API Endpoints

### Authentication Routes (`/api/auth`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | Register new user | ❌ |
| POST | `/login` | User login | ❌ |
| GET | `/profile` | Get user profile | ✅ |
| PUT | `/profile` | Update user profile | ✅ |

### Detection Routes (`/api/detections`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Get all detections | ✅ |
| POST | `/` | Create new detection | ✅ |
| GET | `/:id` | Get specific detection | ✅ |
| PUT | `/:id` | Update detection | ✅ |
| DELETE | `/:id` | Delete detection | ✅ |
| GET | `/user/:userId` | Get user's detections | ✅ |

### Notification Routes (`/api/notifications`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/send` | Send push notification | ✅ |
| POST | `/register-token` | Register device token | ✅ |
| GET | `/history` | Get notification history | ✅ |

## 📊 Data Models

### User Model
```javascript
{
  _id: ObjectId,
  username: String,
  email: String,
  password: String (hashed),
  pushToken: String,
  createdAt: Date,
  updatedAt: Date
}
```

### Detection Model
```javascript
{
  _id: ObjectId,
  userId: ObjectId,
  imageUrl: String,
  detections: [{
    class: String,
    confidence: Number,
    bbox: [Number] // [x, y, width, height]
  }],
  location: {
    latitude: Number,
    longitude: Number
  },
  timestamp: Date,
  processed: Boolean
}
```

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication:

1. **Registration/Login:** Client receives JWT token
2. **Protected Routes:** Include `Authorization: Bearer <token>` header
3. **Token Validation:** Middleware validates token on protected routes

Example request:
```javascript
fetch('/api/detections', {
  headers: {
    'Authorization': 'Bearer your-jwt-token',
    'Content-Type': 'application/json'
  }
})
```

## 🌐 Deployment

### Vercel (Recommended)

This API is configured for Vercel deployment:

1. **Connect your repository** to Vercel
2. **Set environment variables** in Vercel Dashboard:
   - `MONGODB_URI`
   - `JWT_SECRET`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
3. **Deploy** automatically on push to main branch

### Manual Deployment

1. **Build and start:**
   ```bash
   npm install --production
   npm start
   ```

2. **Environment:** Set `NODE_ENV=production`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB connection string | ✅ |
| `JWT_SECRET` | Secret key for JWT tokens | ✅ |
| `SUPABASE_URL` | Supabase project URL | ✅ |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | ✅ |
| `NODE_ENV` | Environment (development/production) | ❌ |
| `PORT` | Server port (default: 3000) | ❌ |

### CORS Configuration

The API is configured to accept requests from:
- `http://localhost:*` (development)
- `https://your-frontend-domain.com` (production)

## 🧪 Testing

### Manual Testing

Test the API using tools like Postman or curl:

```bash
# Register a new user
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"password123"}'

# Login
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

### Health Check

```bash
curl http://localhost:3000/health
```

## 🔍 Integration with Python App

The Python inference engine communicates with this API to:

1. **Store Detection Results:** POST to `/api/detections`
2. **Send Notifications:** POST to `/api/notifications/send`
3. **User Management:** Authentication and user data

Example integration from Python:
```python
import requests

# Store detection result
response = requests.post('http://localhost:3000/api/detections', 
  headers={'Authorization': f'Bearer {token}'},
  json={
    'imageUrl': 'path/to/image.jpg',
    'detections': [{'class': 'rust', 'confidence': 0.95}],
    'location': {'latitude': 40.7128, 'longitude': -74.0060}
  }
)
```

## 📝 Development Notes

- **Database:** Uses MongoDB for flexible document storage
- **File Storage:** Images stored in Supabase Storage
- **Security:** Passwords hashed with bcryptjs
- **CORS:** Configured for cross-origin requests
- **Error Handling:** Comprehensive error responses
- **Logging:** Console logging for debugging


## 📄 License

This project is part of the Wheat Disease Detection System.
