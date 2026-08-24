const { useState, useRef, useEffect, useCallback } = React;

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const API_URL = "http://localhost:8000/predict";

function sanitize(name) {
  return name.replace(/[^a-z0-9]/gi, "").toLowerCase();
}

/* ===== NAVBAR ===== */
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);
  const closeMenu = () => setMenuOpen(false);
  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""} ${menuOpen ? "open" : ""}`}>
      <a href="#" className="navbar-logo">
        <img src="../web_img/valo_logo.png" alt="VALORANT" className="navbar-logo-icon" />
        <img src="../web_img/valorant_logo_text.avif" alt="VALORANT" className="navbar-logo-text" />
      </a>
      <button
        className={`navbar-hamburger ${menuOpen ? "open" : ""}`}
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle navigation menu"
        aria-expanded={menuOpen}
      >
        <span /><span /><span />
      </button>
      <ul className="navbar-links">
        <li><a href="#hero" onClick={closeMenu}>Home</a></li>
        <li><a href="#classifier" className="active" onClick={closeMenu}>Classifier</a></li>
        <li><a href="#powered" onClick={closeMenu}>Technology</a></li>
        <li><a href="#announcements" onClick={closeMenu}>Updates</a></li>
      </ul>
    </nav>
  );
}

/* ===== HERO ===== */
function Hero() {
  return (
    <section className="hero" id="hero">
      <video className="hero-video" autoPlay muted loop playsInline>
        <source src="../Animated/LNY_Sage-3_NightMarket.mp4" type="video/mp4" />
      </video>
      <div className="hero-overlay" />
      <div className="hero-content">
        <p className="hero-eyebrow">AI-Powered Identification</p>
        <h1 className="hero-title">
          Identify Your
          <span>Skin Collection</span>
        </h1>
        <p className="hero-subtitle">
          Drop any weapon skin image and our trained neural network will instantly identify bundle it belongs to.
        </p>
        <div className="hero-actions">
          <a href="#classifier" className="btn btn-primary">
            Start Classifying
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 17L17 7M17 7H7M17 7V17"/></svg>
          </a>
          <a href="#powered" className="btn btn-secondary">Learn More</a>
        </div>
      </div>
    </section>
  );
}

/* ===== CLASSIFIER ===== */
function Classifier() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const lastFileRef = useRef(null);
  const sectionRef = useRef(null);

  const classify = useCallback(async (file) => {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    setResult(null);
    setLoading(true);
    lastFileRef.current = file;

    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(API_URL, { method: "POST", body: fd, signal: controller.signal });
      const d = await r.json();
      if (!r.ok) {
        setError(d.detail || "Prediction failed.");
      } else if (d.error) {
        setError(d.error);
      } else {
        setResult(d);
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setError("Could not reach the API. Check your connection and try again.");
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Invalid file type. Please upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (f.size > MAX_FILE_SIZE) {
      setError(`File too large (${(f.size / (1024 * 1024)).toFixed(1)}MB). Max size: 10MB.`);
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
    classify(f);
  }, [preview, classify]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleRetry = useCallback(() => {
    if (lastFileRef.current) classify(lastFileRef.current);
  }, [classify]);

  const handleClear = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    lastFileRef.current = null;
    sectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [preview]);

  const pct = (v) => (v * 100).toFixed(1);

  return (
    <section ref={sectionRef} className="classifier-section" id="classifier">
      <div className="classifier-layout">
        <div className="classifier-info">
          <p className="section-eyebrow">Skin Classifier</p>
          <h2 className="section-title">Identify Your Gun Skin</h2>
          <div className="section-divider" />
          <p className="classifier-desc">
            Our EfficientNet-B0 model has been trained on thousands of weapon skin images to accurately classify Reaver and Prime collections with high confidence.
          </p>
          <div className="classifier-steps">
            <div className="step"><div className="step-number">01</div><div className="step-text"><h4>Upload Image</h4><p>Drag and drop or click to upload any weapon skin screenshot</p></div></div>
            <div className="step"><div className="step-number">02</div><div className="step-text"><h4>AI Analysis</h4><p>Our neural network analyzes visual features in real-time</p></div></div>
            <div className="step"><div className="step-number">03</div><div className="step-text"><h4>Get Results</h4><p>See the predicted skin collection with confidence scores</p></div></div>
          </div>
        </div>

        <div className="classifier-card">
          <div className="classifier-card-bg">
            <img src="../web_img/9146891368df21c1cae7d528829cddde64783034-3440x1020.avif" alt="" />
          </div>

          {!preview && !loading && !result && (
            <div
              className={`dropzone ${dragOver ? "dragover" : ""}`}
              onDrop={onDrop}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); } }}
              role="button"
              tabIndex={0}
              aria-label="Upload skin image"
            >
              <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => handleFile(e.target.files[0])} style={{ display: "none" }} />
              <div className="dropzone-icon">
                <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <p className="dropzone-title">Upload Skin Image</p>
              <p className="dropzone-subtitle"><span>Click to browse</span> or drag and drop</p>
            </div>
          )}

          {preview && (
            <div className="preview-container">
              <p className="preview-label">Uploaded Image</p>
              <img className="preview-image" src={preview} alt="Uploaded skin" />
            </div>
          )}
          {loading && (
            <div className="loading-container">
              <div className="loading-spinner" />
              <p className="loading-text">Analyzing Skin...</p>
            </div>
          )}
          {error && (
            <div className="error-container">
              <div className="error-icon"><svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
              <div>
                <p className="error-text">{error}</p>
                <div className="error-actions">
                  {lastFileRef.current && <button className="btn-sm btn-retry" onClick={handleRetry}>Retry</button>}
                  <button className="btn-sm btn-clear" onClick={handleClear}>Clear</button>
                </div>
              </div>
            </div>
          )}
          {result && (
            <div className="result-container">
              <div className="result-header"><span className="result-label">Prediction</span><span className={`result-class ${sanitize(result.predicted)}`}>{result.predicted}</span></div>
              {Object.entries(result.scores).sort((a, b) => b[1] - a[1]).map(([name, score]) => (
                <div className="confidence-row" key={name}>
                  <div className="confidence-header"><span className={`confidence-name ${sanitize(name)}`}>{name}</span><span className="confidence-value">{pct(score)}%</span></div>
                  <div className="confidence-track"><div className={`confidence-fill ${sanitize(name)}`} style={{ width: `${pct(score)}%` }} /></div>
                </div>
              ))}
              <div className="result-actions">
                <button className="btn-sm btn-clear" onClick={handleClear}>Upload Another</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

/* ===== POWERED BY ===== */
function PoweredBy() {
  const cards = [
    { video: '../Animated/LNY_Sage-2_ShoppingA.mp4', title: 'EfficientNet-B0', desc: "Google's EfficientNet architecture optimized for mobile and edge deployment with maximum accuracy." },
    { video: '../Animated/LNY_Sage-2_ShoppingB.mp4', title: 'Transfer Learning', desc: 'Pre-trained on ImageNet and fine-tuned on thousands of VALORANT weapon skin images.' },
    { video: '../Animated/LNY_Sage-2_ShoppingC.mp4', title: 'Real-Time Inference', desc: 'Optimized pipeline delivers instant predictions with high confidence scores.' },
  ];
  return (
    <section className="powered-section" id="powered">
      <p className="section-eyebrow">Technology</p>
      <h2 className="section-title">Powered By</h2>
      <div className="section-divider" />
      <div className="powered-grid">
        {cards.map((c, i) => (
          <div className="powered-card" key={i}>
            <video className="powered-card-video" autoPlay muted loop playsInline>
              <source src={c.video} type="video/mp4" />
            </video>
            <div className="powered-card-body">
              <h3 className="powered-card-title">{c.title}</h3>
              <p className="powered-card-desc">{c.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ===== ANNOUNCEMENTS ===== */
function Announcements() {
  return (
    <section className="announce-section" id="announcements">
      <div className="section-header" style={{ textAlign: 'center' }}>
        <p className="section-eyebrow">Version</p>
        <h2 className="section-title">v1.1.0</h2>
        <div className="section-divider" />
      </div>
      <div className="announce-showcase">
        <div className="announce-video-wrap">
          <video className="announce-video" autoPlay muted loop playsInline>
            <source src="../Animated/LNY_Sage-5_CallBack.mp4" type="video/mp4" />
          </video>
          <div className="announce-video-overlay" />
        </div>
        <div className="announce-content">
          <div className="announce-badge">
            <div className="announce-badge-dot" />
            <span>Latest Release</span>
          </div>
          <h3 className="announce-version-title">What's New</h3>
          <ul className="announce-list">
            <li>
              <span className="announce-list-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>Added 7 new bundles (Mystbloom, Elderflame, Glitchpop, Nebula, Oni, Prism, Sovereign)</span>
            </li>
            <li>
              <span className="announce-list-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
              <span>Security hardening and mobile responsiveness fixes</span>
            </li>
          </ul>
          <div className="announce-changelog">
            <span className="announce-date">26/07/2026</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ===== FOOTER ===== */
function Footer() {
  return (
    <footer className="footer">
      <div className="footer-top">
        <div className="footer-brand">
          <img src="../web_img/valo_logo.png" alt="VALORANT" className="footer-logo-icon" />
          <img src="../web_img/valorant_logo_text.avif" alt="VALORANT" className="footer-logo-text" />
          <span className="footer-text">Skin Classifier</span>
        </div>
        <div className="footer-social">
          <a href="#" aria-label="Instagram">
            <svg viewBox="0 0 24 24"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/></svg>
          </a>
          <a href="#" aria-label="LinkedIn">
            <svg viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
          </a>
        </div>
      </div>
      <div className="footer-bottom">
        <p className="footer-credits">
          All images and assets are from the <span>official VALORANT website</span> (playvalorant.com).<br />
          Valorant and all related properties are trademarks of <span>Riot Games, Inc.</span>
        </p>
        <p className="footer-creator">
          Created by <span>Lance Christian C. Crucis</span>
        </p>
      </div>
    </footer>
  );
}

/* ===== APP ===== */
function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <Classifier />
      <PoweredBy />
      <Announcements />
      <Footer />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
