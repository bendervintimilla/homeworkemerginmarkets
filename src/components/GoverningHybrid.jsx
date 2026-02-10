import React from 'react';
import { motion } from 'framer-motion';
import { Shield, GitMerge, Power } from 'lucide-react';

const Pillar = ({ title, icon: Icon, desc, color, delay, number }) => (
    <motion.div
        initial={{ y: 60, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
        className="flex-1 relative group cursor-pointer"
    >
        <div className={`absolute -top-px left-0 right-0 h-[3px] ${color} rounded-full opacity-80 group-hover:opacity-100 group-hover:shadow-[0_0_20px_currentColor] transition-all duration-500`}></div>
        <div className="bg-white/[0.03] backdrop-blur-sm border border-white/[0.06] rounded-2xl p-8 h-full flex flex-col items-start text-left group-hover:bg-white/[0.06] group-hover:border-white/10 transition-all duration-500 group-hover:-translate-y-2">
            <span className="text-[80px] font-bold leading-none text-white/[0.04] absolute top-4 right-6 select-none">{number}</span>
            <div className={`p-3 rounded-2xl bg-white/[0.05] mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <Icon size={24} className="text-gray-300" />
            </div>
            <h3 className="text-xl font-semibold mb-3 text-white tracking-tight">{title}</h3>
            <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
        </div>
    </motion.div>
);

const GoverningHybrid = () => {
    return (
        <section className="py-32 px-4 relative overflow-hidden">
            <div className="max-w-6xl mx-auto">
                {/* Title */}
                <motion.div
                    className="text-center mb-12"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <h2 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight">Governing the Hybrid Line</h2>
                    <p className="text-lg text-gray-500 max-w-xl mx-auto">Resilient orchestration requires restoring the Human‑in‑the‑Loop.</p>
                </motion.div>

                {/* Hero image — tight spacing to cards */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 1 }}
                    className="mb-10 relative rounded-3xl overflow-hidden border border-white/[0.08] max-w-5xl mx-auto group"
                >
                    <img
                        src="/assets/hybrid_governance.png"
                        alt="Human and AI collaboration"
                        className="w-full h-72 md:h-96 object-cover transform group-hover:scale-105 transition-transform duration-[3s]"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/20"></div>
                    <div className="absolute bottom-0 left-0 right-0 p-8 z-10">
                        <p className="text-sm font-mono text-gray-400 tracking-widest uppercase">The Solution</p>
                        <p className="text-2xl md:text-3xl font-bold text-white mt-1">Human + Machine. Together.</p>
                    </div>
                </motion.div>

                {/* Cards — directly below image */}
                <div className="flex flex-col md:flex-row gap-5 max-w-5xl mx-auto">
                    <Pillar
                        number="01"
                        title="Tiered Stop Authority"
                        desc="Restore immediate 'Safety Stops' without bureaucracy. Workers must feel safe to halt the line."
                        icon={Shield}
                        color="bg-red-500"
                        delay={0}
                    />
                    <Pillar
                        number="02"
                        title="Hybrid Ground-Truth"
                        desc="Mandate physical cross-checks against digital signals. The dashboard is not the territory."
                        icon={GitMerge}
                        color="bg-blue-500"
                        delay={0.15}
                    />
                    <Pillar
                        number="03"
                        title="Empowered Override"
                        desc="Train intuition over anomaly scores. If it looks wrong, STOP it. Validate later."
                        icon={Power}
                        color="bg-orange-500"
                        delay={0.3}
                    />
                </div>

                {/* CTA */}
                <div className="mt-20 text-center">
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        whileInView={{ scale: 1, opacity: 1 }}
                        viewport={{ once: true }}
                        className="inline-block p-[1.5px] rounded-full bg-gradient-to-r from-orange-500 via-purple-500 to-blue-500"
                    >
                        <div className="bg-black rounded-full px-10 py-5">
                            <span className="text-xl md:text-2xl font-semibold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
                                Replace Blind Trust with Verified Reality
                            </span>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default GoverningHybrid;
