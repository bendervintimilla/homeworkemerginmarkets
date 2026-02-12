import React from 'react';
import Hero from './components/Hero';
import ThreatProfile from './components/TheShift';
import AttackArsenal from './components/FactoryNervousSystem';
import KillChainWalkthrough from './components/DashboardParadox';
import StrategicDefense from './components/GoverningHybrid';

function App() {
  return (
    <div className="min-h-screen bg-black text-white selection:bg-red-500/30">
      <main>
        <Hero />
        <ThreatProfile />
        <AttackArsenal />
        <KillChainWalkthrough />
        <StrategicDefense />
      </main>

      <footer className="py-12 text-center text-gray-600 text-sm">
        <p>© 2026 Cybersecurity Group 7. All rights reserved.</p>
        <p className="mt-2">Andres · Mattia · Joyce · Juan David · Pablo</p>
      </footer>
    </div>
  );
}

export default App;
