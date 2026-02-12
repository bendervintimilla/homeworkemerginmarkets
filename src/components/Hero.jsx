import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const Hero = () => {
    const { scrollY } = useScroll();
    const y1 = useTransform(scrollY, [0, 500], [0, 200]);
    const opacity = useTransform(scrollY, [0, 300], [1, 0]);

    return (
        <section className="relative min-h-screen w-full flex flex-col items-center justify-center overflow-hidden py-20">
            {/* Background Grid - Apple Style */}
            <div className="absolute inset-0 z-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:100px_100px] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_50%,#000_70%,transparent_100%)]"></div>

            <div className="z-10 text-center px-4 max-w-6xl mx-auto flex flex-col items-center">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, ease: "easeOut" }}
                >
                    <h2 className="text-xl md:text-2xl font-medium tracking-[0.2em] text-red-500 uppercase mb-6">
                        Cybersecurity Group 7
                    </h2>
                </motion.div>

                <motion.h1
                    className="text-6xl md:text-8xl font-bold tracking-tight mb-8 leading-tight"
                    style={{ y: y1, opacity }}
                >
                    <span className="text-gradient block">APT35.</span>
                    <span className="text-3xl md:text-5xl text-gray-500 font-normal mt-4 block">
                        Charming Kitten — <br />
                        <span className="text-white">State-Sponsored Cyber Espionage</span>
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5, duration: 1 }}
                    className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mt-4 mb-4 leading-relaxed"
                >
                    An Iranian state-sponsored threat group conducting long-running operations targeting U.S. and Middle Eastern military, diplomatic, and government entities.
                </motion.p>

                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.7, duration: 1 }}
                    className="text-sm text-gray-600 font-mono mb-12"
                >
                    Andres · Mattia · Joyce · Juan David · Pablo
                </motion.p>

                {/* Dynamic Image Reveal */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 40 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ delay: 0.8, duration: 1.2, ease: "circOut" }}
                    className="relative z-10 w-full max-w-4xl"
                >
                    <div className="relative rounded-2xl overflow-hidden shadow-[0_0_120px_rgba(239,68,68,0.15)] border border-white/10 group">
                        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent z-10 opacity-60"></div>
                        <img
                            src="/assets/hero_apt35.png"
                            alt="APT35 Charming Kitten"
                            className="w-full h-auto object-cover transform group-hover:scale-105 transition-transform duration-[2s]"
                        />
                    </div>
                </motion.div>
            </div>

            {/* Scroll Indicator */}
            <motion.div
                className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20"
                animate={{ y: [0, 10, 0] }}
                transition={{ repeat: Infinity, duration: 2 }}
            >
                <div className="w-6 h-10 border border-gray-600 rounded-full flex justify-center p-2 backdrop-blur-sm">
                    <div className="w-1 h-2 bg-white rounded-full"></div>
                </div>
            </motion.div>
        </section>
    );
};

export default Hero;
