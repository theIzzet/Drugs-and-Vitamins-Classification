import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPrediction(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Lütfen tahmin için bir resim dosyası seçiniz.");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {

        const errorData = await response.json();
        throw new Error(errorData.detail || `API'den hata döndü: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error("Tahmin hatası:", err);
      setError(`Tahmin sırasında bir hata oluştu: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>💊 İlaç/Vitamin Sınıflandırma</h1>
        <p>Görüntü İşleme Transfer Öğrenimi (MobileNetV2) ve FastAPI</p>
      </header>

      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          id="file-upload"
          disabled={loading}
        />
        <label htmlFor="file-upload" className="custom-file-upload">
          {file ? file.name : 'Resim Seçin (.png, .jpg)'}
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? 'Tahmin Ediliyor...' : 'Tahmin Yap'}
        </button>
      </div>

      {error && <p className="error-message"> {error}</p>}

      <div className="results-container">
        <div className="image-preview">
          <h2>Seçilen Görüntü</h2>
          {preview ? (
            <img src={preview} alt="Önizleme" style={{ maxWidth: '100%', maxHeight: '250px', objectFit: 'contain' }} />
          ) : (
            <div className="placeholder">Görüntü Yok</div>
          )}
        </div>

        <div className="prediction-results">
          <h2>Tahmin Sonucu</h2>
          {prediction ? (
            <div>
              <p className="prediction-text">
                Tahmin Edilen İlaç: <strong>{prediction.predicted_class}</strong>
              </p>
              <p className="confidence-text">
                Güven: <strong>{(prediction.confidence * 100).toFixed(2)}%</strong>
              </p>

              {/* Olasılıkları listeleyelim */}
              {prediction.all_probabilities && (
                <div className="probabilities">
                  <h3>Tüm Olasılıklar:</h3>
                  <ul>
                    {Object.entries(prediction.all_probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([className, probability]) => (
                        <li key={className} style={{ fontWeight: className === prediction.predicted_class ? 'bold' : 'normal' }}>
                          {className}: {(probability * 100).toFixed(2)}%
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p>Lütfen bir resim yükleyin ve 'Tahmin Yap' butonuna tıklayın.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;