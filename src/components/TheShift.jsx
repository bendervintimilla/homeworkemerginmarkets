import React from 'react';
import { motion } from 'framer-motion';
import { Crosshair, Globe, ArrowRight, Users, Brain, TrendingUp, Target, Shield } from 'lucide-react';

/* ─────────── Reusable Card ─────────── */
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

/* ─────────── Targeting Column ─────────── */
const TargetColumn = ({ title, items, color, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay }}
        className="flex-1"
    >
        <p className={`text-xs font-mono ${color} tracking-widest uppercase mb-6`}>{title}</p>
        <div className="space-y-4">
            {items.map((item, i) => (
                <div key={i} className="flex gap-4 items-start">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500 mt-1.5 flex-shrink-0 shadow-[0_0_6px_rgba(239,68,68,0.4)]"></div>
                    <span className="text-gray-300 text-sm leading-relaxed">{item}</span>
                </div>
            ))}
        </div>
    </motion.div>
);

const ThreatProfile = () => {
    return (
        <section className="py-32 px-4 relative">
            <div className="max-w-7xl mx-auto">
                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="text-center mb-6"
                >
                    <p className="text-xs font-mono text-red-400 tracking-[0.3em] uppercase mb-4">1 · Strategic Overview</p>
                    <h2 className="text-4xl md:text-5xl font-bold mb-4">The Iranian Nexus</h2>
                    <p className="text-sm text-gray-600 font-mono">Joyce Eskafi</p>
                </motion.div>

                <motion.p
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="text-xl text-gray-400 max-w-3xl mx-auto text-center mb-24"
                >
                    APT35 is not noisy or destructive — it is patient, targeted, and strategic.
                </motion.p>

                {/* Origin & Attribution + Why APT35 Matters Now */}
                <div className="flex flex-col md:flex-row gap-8 items-stretch mb-24">
                    <Card
                        title="Origin & Attribution"
                        subtitle="Iranian state-linked APT, active since 2011. Focus: intelligence collection & strategic influence."
                        details={[
                            "Alias: Charming Kitten",
                            "Alias: Phosphorus",
                            "Alias: Mint Sandstorm",
                            "Alias: ITG18",
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
                        title="Why APT35 Matters Now"
                        subtitle="Blends cyber operations with social engineering. Targets people first, systems second."
                        details={[
                            "Social engineering first — not exploit-first",
                            "Fake personas (journalists, academics)",
                            "Trust-building over time",
                            "Credential theft as primary objective",
                        ]}
                        icon={Brain}
                        delay={0.2}
                    />
                </div>

                {/* Targeting & Tactics — 3-column layout matching PPTX Slide 3 */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="max-w-5xl mx-auto mb-24"
                >
                    <div className="flex items-center gap-4 mb-10">
                        <div className="p-3 rounded-2xl bg-white/5">
                            <Target size={24} className="text-red-400" />
                        </div>
                        <h3 className="text-3xl md:text-4xl font-bold">Targeting & Tactics</h3>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
                        <TargetColumn
                            title="Early Focus"
                            items={[
                                "Activists",
                                "Journalists",
                                "Political dissidents",
                            ]}
                            color="text-red-400"
                            delay={0}
                        />
                        <TargetColumn
                            title="Expanded Targets"
                            items={[
                                "Governments & diplomats",
                                "Think tanks & policy",
                                "Defense & Telecom",
                                "Biotech & Nuclear research",
                            ]}
                            color="text-orange-400"
                            delay={0.15}
                        />
                    </div>

                    {/* Arrow + conclusion */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.5 }}
                        className="mt-10 flex items-center gap-3"
                    >
                        <ArrowRight size={18} className="text-red-400" />
                        <span className="text-gray-400 text-sm font-mono italic">Patient, targeted, strategic</span>
                    </motion.div>
                </motion.div>

            </div>
        </section>
    );
};

export default ThreatProfile;
