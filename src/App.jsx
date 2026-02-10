import React from 'react';
import Hero from './components/Hero';
import TheShift from './components/TheShift';
import FactoryNervousSystem from './components/FactoryNervousSystem';
import DashboardParadox from './components/DashboardParadox';
import GoverningHybrid from './components/GoverningHybrid';

function App() {
  return (
    <div className="min-h-screen bg-black text-white selection:bg-blue-500/30">
      <main>
        <Hero />
        <TheShift />
        <FactoryNervousSystem />
        <DashboardParadox />
        <GoverningHybrid />
      </main>

      <footer className="py-12 text-center text-gray-600 text-sm">
        <p>© 2026 Emerging Markets Group 7. All rights reserved.</p>
        <p className="mt-2">Designed with Algorithmic Precision.</p>
      </footer>
    </div>
  );
}

export default App;
