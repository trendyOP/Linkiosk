import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ModeProvider } from './context/ModeContext';
import ModeGate from './components/ModeGate';
import TopBar from './components/TopBar';
import StoreOrPack from './pages/StoreOrPack';
import Menu from './pages/Menu';
import Cart from './pages/Cart';

export default function App() {
  return (
    <ModeProvider>
      <TopBar />
      <Routes>
        {/* 1) Choose mode first */}
        <Route path="/mode" element={<ModeGate />} />
        {/* 2) Then store/pack */}
        <Route path="/start" element={<StoreOrPack />} />
        {/* 3) Menu & Cart */}
        <Route path="/menu" element={<Menu />} />
        <Route path="/cart" element={<Cart />} />
        {/* Default route -> mode gate */}
        <Route path="*" element={<Navigate to="/mode" replace />} />
      </Routes>
    </ModeProvider>
  );
}
