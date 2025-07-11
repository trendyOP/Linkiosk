import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles.css';
import App from './App';

/* 로그로 로드 확인 */
console.log('🟢 index.js loaded: mounting <App />');

const container = document.getElementById('root');
if (!container) {
  throw new Error('#root element not found. Make sure public/index.html contains <div id="root"></div>');
}

const root = ReactDOM.createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);