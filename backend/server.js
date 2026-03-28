const express = require ("express");
const axios = require("axios");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

//Route to check server
app.get("/",(req,res) => {
    res.send("Backend is running");
});

//Main route → talks to ML API
app.post("/predict-fraud", async (req, res) => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        features: req.body.features,
      });
  
      res.json(response.data);
    } catch (error) {
      console.error(error.message);
      res.status(500).json({ error: "Prediction failed" });
    }
  });

  await axios.post("http://127.0.0.1:5000/predict", {
    features: req.body.features
  }, {
    headers: {
      "Content-Type": "application/json"
    }
  });
  
  app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
  });
