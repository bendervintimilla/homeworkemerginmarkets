import React from 'react';
import { motion } from 'framer-motion';
import { Crosshair, Globe, ArrowRight, Users, Brain, TrendingUp } from 'lucide-react';

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

const EvolutionItem = ({ title, desc, delay }) => (
    <motion.div
        initial={{ opacity: 0, x: -30 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay, ease: "easeOut" }}
        className="flex gap-5 items-start"
    >
        <div className="w-3 h-3 rounded-full bg-red-500 mt-2 flex-shrink-0 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
        <div>
            <h4 className="text-white font-semibold text-lg mb-1">{title}</h4>
            <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
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
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                        An Iranian state-linked APT active since 2011, conducting long-term cyber espionage aligned with Iran's geopolitical and national security objectives — not financial crime.
                    </p>
                </motion.div>

                {/* Origin & Targets Cards */}
                <div className="flex flex-col md:flex-row gap-8 items-stretch mb-24">
                    <Card
                        title="Origin & Attribution"
                        subtitle="Iranian state-sponsored group linked to the Islamic Revolutionary Guard Corps (IRGC). First gained widespread attention in 2014."
                        details={[
                            "Also known as: Charming Kitten",
                            "Alias: Phosphorus / TA453",
                            "Alias: Mint Sandstorm (Microsoft)",
                            "Alias: ITG18 / Magic Hound / Newscaster",
                            "Active since: ~2011",
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
                        subtitle="Strategic intelligence collection against high-value sectors — primarily in the U.S., Israel, and the Middle East."
                        details={[
                            "Government & diplomatic entities",
                            "Defense Industrial Base & energy",
                            "Think tanks & nuclear policy institutions",
                            "Telecom & logistics / transportation",
                            "Biotech, pharma & academic research",
                            "Journalists, activists & dissidents",
                        ]}
                        icon={Globe}
                        delay={0.2}
                    />
                </div>

                {/* Evolution of Targeting */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="max-w-5xl mx-auto mb-24"
                >
                    <div className="flex items-center gap-4 mb-10">
                        <div className="p-3 rounded-2xl bg-white/5">
                            <TrendingUp size={24} className="text-red-400" />
                        </div>
                        <h3 className="text-3xl md:text-4xl font-bold">Evolution of Targeting</h3>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-16 gap-y-8">
                        {/* Left: Foundational */}
                        <div>
                            <p className="text-xs font-mono text-red-400 tracking-widest uppercase mb-6">Foundational Focus (Ongoing)</p>
                            <div className="space-y-6">
                                <EvolutionItem
                                    title="Domestic Surveillance"
                                    desc="Consistently targets activists, human rights defenders, journalists, and political dissidents to identify, track, and suppress opposition."
                                    delay={0}
                                />
                                <EvolutionItem
                                    title="AI-Enhanced Lures"
                                    desc="Utilizes AI-generated content to increase the success rate of social engineering operations against domestic targets."
                                    delay={0.1}
                                />
                            </div>
                        </div>

                        {/* Right: Strategic */}
                        <div>
                            <p className="text-xs font-mono text-cyan-400 tracking-widest uppercase mb-6">Strategic Expansion (Modern)</p>
                            <div className="space-y-6">
                                <EvolutionItem
                                    title="Government & Diplomacy"
                                    desc="Foreign policy and intelligence espionage across the U.S., Israel, and the Middle East."
                                    delay={0.15}
                                />
                                <EvolutionItem
                                    title="Think Tanks & Policy"
                                    desc="Institutions focused on Middle Eastern security and nuclear policy — a supply chain for state secrets."
                                    delay={0.25}
                                />
                                <EvolutionItem
                                    title="Critical Infrastructure"
                                    desc="Defense Industrial Base, telecommunications, and logistics/transportation sectors."
                                    delay={0.35}
                                />
                                <EvolutionItem
                                    title="Academic & Scientific"
                                    desc="Emphasis on biotechnology, pharmaceuticals, and nuclear engineering research."
                                    delay={0.45}
                                />
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Why Relevant Today */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="max-w-5xl mx-auto"
                >
                    <div className="relative glass-panel rounded-3xl border border-white/10 overflow-hidden">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-red-500 via-orange-500 to-cyan-500"></div>
                        <div className="p-10 md:p-12">
                            <div className="flex items-center gap-4 mb-6">
                                <div className="p-3 rounded-2xl bg-white/5">
                                    <Brain size={24} className="text-orange-400" />
                                </div>
                                <h3 className="text-2xl md:text-3xl font-bold">Why APT35 Is Relevant Today</h3>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                                <div>
                                    <Users size={20} className="text-red-400 mb-3" />
                                    <h4 className="text-white font-semibold mb-2">People-First Strategy</h4>
                                    <p className="text-gray-400 text-sm leading-relaxed">Impersonates trusted figures — journalists, academics, professionals — to build rapport before striking. Human behavior is exploited before technical systems.</p>
                                </div>
                                <div>
                                    <Globe size={20} className="text-orange-400 mb-3" />
                                    <h4 className="text-white font-semibold mb-2">Persona Engineering</h4>
                                    <p className="text-gray-400 text-sm leading-relaxed">Exploits publicly available information to craft realistic fake identities. Establishes trust over time, then harvests credentials or delivers payloads.</p>
                                </div>
                                <div>
                                    <Brain size={20} className="text-cyan-400 mb-3" />
                                    <h4 className="text-white font-semibold mb-2">Patient & Strategic</h4>
                                    <p className="text-gray-400 text-sm leading-relaxed">Not noisy or destructive. Prioritizes long-term access and intelligence collection over immediate disruption — combining technical capability with psychological manipulation.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>

            </div>
        </section>
    );
};

export default ThreatProfile;
