import React from 'react';
import { motion } from 'framer-motion';
import { Crosshair, Globe, ArrowRight } from 'lucide-react';

const Card = ({ title, subtitle, details, icon: Icon, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
        className="flex-1 glass-panel p-10 rounded-3xl relative overflow-hidden group hover:border-red-500/30 transition-colors duration-500"
    >
        <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity duration-500">
            <Icon size={120} strokeWidth={1} />
        </div>

        <div className="relative z-10">
            <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-red-500/10 group-hover:text-red-400 transition-colors duration-500">
                <Icon size={28} />
            </div>
            <h3 className="text-3xl font-semibold mb-2">{title}</h3>
            <p className="text-gray-400 text-lg leading-relaxed mb-6">{subtitle}</p>
            <div className="space-y-3 text-sm font-mono">
                {details.map((item, i) => (
                    <div key={i} className="flex items-center gap-3 text-gray-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></span>
                        <span>{item}</span>
                    </div>
                ))}
            </div>
        </div>
    </motion.div>
);

const ThreatProfile = () => {
    return (
        <section className="py-32 px-4 relative">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="text-center mb-24"
                >
                    <h2 className="text-4xl md:text-5xl font-bold mb-6">Threat Profile</h2>
                    <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                        A persistent, well-resourced threat actor with deep ties to Iranian intelligence operations.
                    </p>
                </motion.div>

                <div className="flex flex-col md:flex-row gap-8 items-stretch">
                    <Card
                        title="Origin & Attribution"
                        subtitle="Iranian state-sponsored group assessed to be affiliated with the Islamic Revolutionary Guard Corps (IRGC)."
                        details={[
                            "Also known as: Charming Kitten",
                            "Alias: Phosphorus / TA453",
                            "Alias: Magic Hound / Newscaster",
                            "Active since: ~2014",
                        ]}
                        icon={Crosshair}
                        delay={0}
                    />

                    <div className="flex items-center justify-center py-4 md:py-0">
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            whileInView={{ scale: 1, opacity: 1 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                            className="p-4 rounded-full bg-red-500/10 text-red-400"
                        >
                            <ArrowRight size={32} />
                        </motion.div>
                    </div>

                    <Card
                        title="Targets & Motivation"
                        subtitle="Strategic intelligence collection against high-value sectors — primarily in the U.S. and Middle East."
                        details={[
                            "Government & military officials",
                            "Defense & energy sectors",
                            "Media & journalism",
                            "Academia & dissidents",
                        ]}
                        icon={Globe}
                        delay={0.2}
                    />
                </div>
            </div>
        </section>
    );
};

export default ThreatProfile;
