console.log("Server starting...");
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));

// Load breed labels from file
let labels = [];

function loadLabels() {
    try {
        const labelsPath = path.join(__dirname, 'labels.txt');
        const labelsData = fs.readFileSync(labelsPath, 'utf8');
        labels = labelsData.split('\n').filter(label => label.trim() !== '');
        console.log(`Loaded ${labels.length} breed labels:`, labels);
    } catch (error) {
        console.log('Could not load labels.txt, using default breeds');
        labels = ['Holstein', 'Angus', 'Hereford', 'Jersey', 'Brahman'];
    }
}

// Simulate model prediction (you can replace this with actual TensorFlow Lite processing later)
function simulateModelPrediction() {
    const randomIndex = Math.floor(Math.random() * labels.length);
    const confidence = 0.80 + Math.random() * 0.19; // 80-99% confidence
    
    return {
        breed: labels[randomIndex],
        confidence: confidence
    };
}

// Image preprocessing (resize and validate)
async function processImage(imageBuffer) {
    try {
        // Resize image to standard size and convert to JPEG
        const processedBuffer = await sharp(imageBuffer)
            .resize(224, 224)
            .jpeg({ quality: 80 })
            .toBuffer();
        
        return processedBuffer;
    } catch (error) {
        throw new Error('Failed to process image: ' + error.message);
    }
}

// Main classification endpoint
app.post('/api/classify', async (req, res) => {
    try {
        console.log('📸 Received classification request');
        
        // Check if image data exists
        if (!req.body.image) {
            return res.status(400).json({ error: 'No image data provided' });
        }
        
        // Decode base64 image
        const imageBuffer = Buffer.from(req.body.image, 'base64');
        console.log(`📊 Image size: ${imageBuffer.length} bytes`);
        
        // Process the image
        await processImage(imageBuffer);
        console.log('✅ Image processed successfully');
        
        // Simulate processing time (remove this in production)
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Get prediction from model (simulated for now)
        const prediction = simulateModelPrediction();
        
        console.log(`🐄 Prediction: ${prediction.breed} (${(prediction.confidence * 100).toFixed(2)}%)`);
        
        // Send response
        res.json({
            breed: prediction.breed,
            confidence: prediction.confidence,
            processingMethod: 'Cloud Model',
            timestamp: new Date().toISOString(),
            message: 'Classification successful'
        });
        
    } catch (error) {
        console.error('❌ Classification error:', error.message);
        res.status(500).json({ 
            error: 'Classification failed', 
            message: error.message 
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'Server is running! 🚀',
        labels: labels.length,
        availableBreeds: labels,
        timestamp: new Date().toISOString()
    });
});

// Test endpoint
app.get('/', (req, res) => {
    res.send(`
        <h1>🐄 Cow Classifier Backend</h1>
        <p>✅ Server is running!</p>
        <p>📱 Ready to receive requests from Android app</p>
        <p>🔗 <a href="/api/health">Check Health Status</a></p>
    `);
});

// Start server
app.listen(PORT, "0.0.0.0", () => {
    console.log('🚀 Cow Classifier Backend Server Started!');
    console.log(`📡 Server running at: http://localhost:${PORT}`);
    console.log(`🌍 Network access: http://YOUR-IP-ADDRESS:${PORT}`);
    console.log('📱 Ready to receive requests from Android app!');
    console.log('---');
    loadLabels();
});
