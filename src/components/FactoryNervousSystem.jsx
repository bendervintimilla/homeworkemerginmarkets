import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const FactoryNervousSystem = () => {
    const containerRef = useRef(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start end", "end start"]
    });

    const opacity = useTransform(scrollYProgress, [0, 0.3], [0, 1]);

    return (
        <section ref={containerRef} className="py-32 relative overflow-hidden bg-black">
            <div className="max-w-7xl mx-auto px-4 mb-20 relative z-20 text-center">
                <motion.h2
                    style={{ opacity }}
                    className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-orange-400 via-white to-blue-500"
                >
                    Rewiring the Factory Nervous System
                </motion.h2>
                <p className="text-xl text-gray-400 max-w-3xl mx-auto">Transforming decision architecture from the shop floor to the system layer.</p>
            </div>

            {/* Two-card layout with unique images */}
            <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 md:grid-cols-2 gap-8">

                {/* Left Card: Local Trust */}
                <motion.div
                    className="relative rounded-[32px] overflow-hidden border border-white/10 group cursor-pointer"
                    initial={{ x: -80, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ duration: 1, ease: "anticipate" }}
                    viewport={{ once: true }}
                    whileHover={{ y: -8 }}
                >
                    <div className="aspect-square relative overflow-hidden">
                        <img
                            src="/assets/local_trust.png"
                            alt="Local Trust — Human Decision Making"
                            className="absolute inset-0 w-full h-full object-cover transition-transform duration-[3s] group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent"></div>
                    </div>

                    <div className="absolute bottom-0 left-0 right-0 p-10">
                        <h3 className="text-4xl md:text-5xl font-bold text-orange-500 mb-6 tracking-tight">LOCAL TRUST</h3>
                        <div className="space-y-4 text-sm font-mono text-gray-300">
                            <div className="flex justify-between border-b border-white/10 pb-3">
                                <span className="text-gray-500">Authority</span>
                                <span className="text-orange-400 font-bold">Immediate Stoppage</span>
                            </div>
                            <div className="flex justify-between border-b border-white/10 pb-3">
                                <span className="text-gray-500">Decision Speed</span>
                                <span>Real-time Negotiation</span>
                            </div>
                            <div className="flex justify-between border-b border-white/10 pb-3">
                                <span className="text-gray-500">Recovery</span>
                                <span>Improvised Workarounds</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Knowledge</span>
                                <span>Tacit / Experiential</span>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Right Card: System Trust */}
                <motion.div
                    className="relative rounded-[32px] overflow-hidden border border-white/10 group cursor-pointer"
                    initial={{ x: 80, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ duration: 1, ease: "anticipate" }}
                    viewport={{ once: true }}
                    whileHover={{ y: -8 }}
                >
                    <div className="aspect-square relative overflow-hidden">
                        <img
                            src="/assets/system_trust.png"
                            alt="System Trust — Algorithmic Control"
                            className="absolute inset-0 w-full h-full object-cover transition-transform duration-[3s] group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-blue-950 via-blue-950/50 to-transparent"></div>
                    </div>

                    <div className="absolute bottom-0 left-0 right-0 p-10">
                        <h3 className="text-4xl md:text-5xl font-bold text-blue-400 mb-6 tracking-tight">SYSTEM TRUST</h3>
                        <div className="space-y-4 text-sm font-mono text-gray-300">
                            <div className="flex justify-between border-b border-blue-500/20 pb-3">
                                <span className="text-gray-500">Authority</span>
                                <span className="text-blue-300 font-bold">Permission-based</span>
                            </div>
                            <div className="flex justify-between border-b border-blue-500/20 pb-3">
                                <span className="text-gray-500">Decision Speed</span>
                                <span>Pre-computed Optimization</span>
                            </div>
                            <div className="flex justify-between border-b border-blue-500/20 pb-3">
                                <span className="text-gray-500">Recovery</span>
                                <span>Formalized Playbooks</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500">Knowledge</span>
                                <span>Explicit / Data-driven</span>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default FactoryNervousSystem;
