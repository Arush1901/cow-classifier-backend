console.log("Server starting...");
const express = require('express');
const tf = require('@tensorflow/tfjs');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const bodyParser = require('body-parser');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

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

// Load TFLite model
let model;
async function loadModel() {
  try {
    model = await tf.loadGraphModel('file://model.tflite');
    console.log("✅ TFLite model loaded!");
  } catch (err) {
    console.error("❌ Failed to load TFLite model:", err.message);
  }
}

// Predict function
async function predict(imageBuffer) {
  const input = tf.node
    .decodeImage(imageBuffer, 3) // decode into RGB
    .resizeNearestNeighbor([224, 224])
    .expandDims(0)
    .toFloat()
    .div(255.0);

  const output = model.predict(input);
  const data = output.dataSync();

  const maxIndex = data.indexOf(Math.max(...data));
  return {
    breed: labels[maxIndex] || "Unknown",
    confidence: data[maxIndex]
  };
}

// Image preprocessing
async function processImage(imageBuffer) {
  return await sharp(imageBuffer)
    .resize(224, 224)
    .jpeg({ quality: 80 })
    .toBuffer();
}

// Classification endpoint
app.post('/api/classify', async (req, res) => {
  try {
    console.log('📸 Received classification request');
    if (!req.body.image) {
      return res.status(400).json({ error: 'No image data provided' });
    }

    const imageBuffer = Buffer.from(req.body.image, 'base64');
    console.log(`📊 Image size: ${imageBuffer.length} bytes`);

    const processedBuffer = await processImage(imageBuffer);
    console.log('✅ Image processed successfully');

    const prediction = await predict(processedBuffer);

    console.log(`🐄 Prediction: ${prediction.breed} (${(prediction.confidence * 100).toFixed(2)}%)`);

    res.json({
      breed: prediction.breed,
      confidence: prediction.confidence,
      processingMethod: 'TFLite',
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

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'Server is running! 🚀',
    labels: labels.length,
    availableBreeds: labels,
    timestamp: new Date().toISOString()
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.send(`
    <h1>🐄 Cow Classifier Backend</h1>
    <p>✅ Server is running!</p>
    <p>📱 Ready to receive requests from Android app</p>
    <p>🔗 <a href="/api/health">Check Health Status</a></p>
  `);
});

// Start server
app.listen(PORT, "0.0.0.0", async () => {
  console.log('🚀 Cow Classifier Backend Server Started!');
  console.log(`📡 Running on port: ${PORT}`);
  console.log('---');
  loadLabels();
  await loadModel();
});
