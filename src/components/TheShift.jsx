import React from 'react';
import { motion } from 'framer-motion';
import { Network, Cpu, ArrowRight } from 'lucide-react';

const Card = ({ title, subtitle, icon: Icon, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
        className="flex-1 glass-panel p-10 rounded-3xl relative overflow-hidden group hover:border-[#2997ff]/30 transition-colors duration-500"
    >
        <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity duration-500">
            <Icon size={120} strokeWidth={1} />
        </div>

        <div className="relative z-10">
            <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-[#2997ff]/10 group-hover:text-[#2997ff] transition-colors duration-500">
                <Icon size={28} />
            </div>
            <h3 className="text-3xl font-semibold mb-2">{title}</h3>
            <p className="text-gray-400 text-lg leading-relaxed">{subtitle}</p>
        </div>
    </motion.div>
);

const TheShift = () => {
    return (
        <section className="py-32 px-4 relative">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="text-center mb-24"
                >
                    <h2 className="text-4xl md:text-5xl font-bold mb-6">The Definition</h2>
                    <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                        A fundamental transformation in how value is created and coordinated across distributed systems.
                    </p>
                </motion.div>

                <div className="flex flex-col md:flex-row gap-8 items-stretch">
                    <Card
                        title="Local Execution"
                        subtitle="Alignment of actions, timing, and dependencies via social contracts and tacit adjustments."
                        icon={Network}
                        delay={0}
                    />

                    <div className="flex items-center justify-center py-4 md:py-0">
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            whileInView={{ scale: 1, opacity: 1 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                            className="p-4 rounded-full bg-[#2997ff]/10 text-[#2997ff]"
                        >
                            <ArrowRight size={32} />
                        </motion.div>
                    </div>

                    <Card
                        title="Algorithmic Orchestration"
                        subtitle="Moving from tacit adjustments to system-mediated, rule-based algorithms. Validating logic over doing work."
                        icon={Cpu}
                        delay={0.2}
                    />
                </div>
            </div>
        </section>
    );
};

export default TheShift;
