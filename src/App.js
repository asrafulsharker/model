import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import { loadLayersModel, IOHandler } from '@tensorflow/tfjs-layers';

import modelJson from './model (1).json';

const customIOHandler: IOHandler = {
  async load() {
    return {
      modelTopology: modelJson.modelTopology,
      format: modelJson.format,
    };
  },
  async save() {
    return {};
  },
};

function App() {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [model, setModel] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [results, setResults] = useState([]);
  const [history, setHistory] = useState([]);

  const imageRef = useRef();
  const textInputRef = useRef();
  const fileInputRef = useRef();

  const loadModel = async () => {
    try {
      const loadedModel = await loadLayersModel(customIOHandler);
      setModel(loadedModel);
      setIsModelLoading(false);
    } catch (error) {
      console.error(error);
      setIsModelLoading(false);
    }
  };

  const uploadImage = (e) => {
    const { files } = e.target;
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0]);
      setImageURL(url);
    } else {
      setImageURL(null);
    }
  };

  const identify = async () => {
    textInputRef.current.value = '';
    if (model && imageURL) {
      const img = new Image();
      img.src = imageURL;

      // Wait for the image to load
      await img.decode();

      // Preprocess the image
      const tensorImg = tf.browser.fromPixels(img).resizeNearestNeighbor([240, 240]).toFloat();
      const offset = tf.scalar(127.5);
      const normalized = tensorImg.sub(offset).div(offset).expandDims();

      // Make the prediction
      const prediction = await model.predict(normalized).array();
      setResults(prediction[0]);

      // Clean up the tensor
      tensorImg.dispose();
    }
  };

  const handleOnChange = (e) => {
    setImageURL(e.target.value);
    setResults([]);
  };

  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  useEffect(() => {
    loadModel();
  }, []);

  useEffect(() => {
    if (imageURL) {
      setHistory([imageURL, ...history]);
    }
  }, [imageURL]);

  if (isModelLoading) {
    return <h2>Model Loading...</h2>;
  }

  return (
    <div className="App">
      <h1 className="header">Image Identification</h1>
      <div className="inputHolder">
        <input type="file" accept="image/*" capture="camera" className="uploadInput" onChange={uploadImage} ref={fileInputRef} />
        <button className="uploadImage" onClick={triggerUpload}>
          Upload Image
        </button>
        <span className="or">OR</span>
        <input type="text" placeholder="Paste image URL" ref={textInputRef} onChange={handleOnChange} />
      </div>
      <div className="mainWrapper">
        <div className="mainContent">
          <div className="imageHolder">
            {imageURL && <img src={imageURL} alt="Upload Preview" crossOrigin="anonymous" ref={imageRef} />}
          </div>
          {results.length > 0 && (
            <div className="resultsHolder">
              {results.map((result, index) => {
                return (
                  <div className="result" key={index}>
                    <span className="name">{`Class ${index + 1}`}</span>
                    <span className="confidence">Confidence level: {(result * 100).toFixed(2)}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
        {imageURL && <button className="button" onClick={identify}>Identify Image</button>}
      </div>
      {history.length > 0 && (
        <div className="recentPredictions">
          <h2>Recent Images</h2>
          <div className="recentImages">
            {history.map((image, index) => {
              return (
                <div className="recentPrediction" key={`${image}${index}`}>
                  <img src={image} alt="Recent Prediction" onClick={() => setImageURL(image)} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
