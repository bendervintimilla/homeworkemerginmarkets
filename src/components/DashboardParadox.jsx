import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Crosshair, Key, Network, Trash2 } from 'lucide-react';

const EventPoint = ({ icon: Icon, time, title, isActive, onClick, color }) => (
    <motion.button
        onClick={onClick}
        className="relative flex flex-col items-center cursor-pointer group outline-none"
        whileHover={{ y: -5 }}
        whileTap={{ scale: 0.95 }}
    >
        <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-all duration-500 ${isActive ? `${color} text-white shadow-lg scale-110` : 'bg-gray-900 text-gray-500 border border-gray-800 hover:border-gray-600'}`}>
            <Icon size={28} />
        </div>
        <div className="text-center w-40">
            <span className={`font-mono text-xs mb-2 block transition-colors ${isActive ? 'text-red-400' : 'text-gray-600'}`}>{time}</span>
            <h4 className={`font-bold text-sm transition-colors duration-300 ${isActive ? 'text-white' : 'text-gray-500 group-hover:text-gray-300'}`}>{title}</h4>
        </div>

        {isActive && (
            <motion.div
                layoutId="activeDot"
                className="absolute -bottom-8 w-2 h-2 bg-white rounded-full"
            />
        )}
    </motion.button>
);

const KillChainWalkthrough = () => {
    const [activeStep, setActiveStep] = useState(0);

    const steps = [
        {
            time: "PHASES 1–2",
            title: "INITIAL ACCESS",
            desc: "Spearphishing link (T1566.002) targets an energy company employee via a tailored recruiting email. Victim trusts the domain — PupyRAT delivered via resume lure. A custom backdoor (BROKEYOLK) ensures persistent access with no noisy exploits.",
            icon: Crosshair,
            color: "bg-green-600",
            image: "/assets/killchain_phase1.png",
            techniques: ["T1566.002 — Spearphishing Link", "T1189 — Drive-by Compromise", "S0192 — PupyRAT C2 Channel"]
        },
        {
            time: "PHASES 3–4",
            title: "CREDENTIAL THEFT",
            desc: "LSASS memory dumped via Mimikatz — domain-level credentials harvested enabling broad authentication. Stolen credentials used to log into VPN and O365 as a legitimate employee. Traffic looks completely normal.",
            icon: Key,
            color: "bg-yellow-600",
            image: "/assets/killchain_phase2.png",
            techniques: ["T1003.001 — LSASS Memory Dump", "S0002 — Mimikatz", "T1078 — Valid Accounts Abuse"]
        },
        {
            time: "PHASES 5–6",
            title: "LATERAL MOVEMENT",
            desc: "PsExec deploys batch files running Mimikatz across multiple systems via service execution. Access scales exponentially. Exchange permissions manipulated via PowerShell — hundreds of mailboxes accessed through OWA with no additional malware.",
            icon: Network,
            color: "bg-orange-600",
            image: "/assets/killchain_phase3.png",
            techniques: ["T1569.002 — Service Execution", "S0029 — PsExec", "T1098.002 — Email Delegate", "T1114 — Email Collection"]
        },
        {
            time: "PHASE 7",
            title: "CLEANUP & EVASION",
            desc: "Mimikatz binaries and artifacts deleted from remote systems. Forensic reconstruction severely hindered. The adversary leaves minimal trace — using legitimate tools that blend into normal enterprise activity.",
            icon: Trash2,
            color: "bg-red-600",
            image: "/assets/killchain_phase4.png",
            techniques: ["T1070.004 — Indicator Removal", "File Deletion Post-Exploitation", "Anti-Forensic Operations"]
        }
    ];

    return (
        <section className="py-32 px-4 bg-[#050505] relative overflow-hidden">
            {/* Background Glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-red-500/5 rounded-full blur-[120px] pointer-events-none"></div>

            <div className="max-w-7xl mx-auto relative z-10">
                <div className="text-center mb-24">
                    <h2 className="text-4xl md:text-6xl font-bold mb-6">Kill Chain Walkthrough</h2>
                    <p className="text-xl text-gray-400 max-w-2xl mx-auto">7-Phase Breach Scenario — From spearphishing to full enterprise compromise and evasion.</p>
                </div>

                {/* Timeline */}
                <div className="relative mb-24 px-8">
                    <div className="absolute top-8 left-0 right-0 h-[2px] bg-gray-900 z-0"></div>
                    <motion.div
                        className="absolute top-8 left-0 h-[2px] bg-gradient-to-r from-green-500 via-orange-500 to-red-600 z-0 shadow-[0_0_10px_rgba(255,255,255,0.2)]"
                        initial={{ width: 0 }}
                        animate={{ width: `${(activeStep / (steps.length - 1)) * 100}%` }}
                        transition={{ duration: 0.8, ease: "easeInOut" }}
                    />

                    <div className="relative z-10 flex justify-between">
                        {steps.map((step, index) => (
                            <EventPoint
                                key={index}
                                icon={step.icon}
                                time={step.time}
                                title={step.title}
                                color={step.color}
                                isActive={index === activeStep}
                                onClick={() => setActiveStep(index)}
                            />
                        ))}
                    </div>
                </div>

                {/* Detail Panel */}
                <div className="max-w-5xl mx-auto relative min-h-[450px]">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeStep}
                            initial={{ opacity: 0, y: 30, filter: "blur(10px)" }}
                            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                            exit={{ opacity: 0, y: -30, filter: "blur(10px)" }}
                            transition={{ duration: 0.5, ease: "easeOut" }}
                            className="glass-panel p-8 md:p-10 rounded-[32px] border border-white/10 shadow-2xl flex flex-col md:flex-row gap-10 items-center"
                        >
                            {/* Image — unique per step */}
                            <div className="w-full md:w-1/2 aspect-square overflow-hidden rounded-2xl border border-white/5 relative group flex-shrink-0">
                                <div className="absolute inset-0 bg-gradient-to-tr from-black/60 to-transparent z-10"></div>
                                <img
                                    src={steps[activeStep].image}
                                    alt={steps[activeStep].title}
                                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-700"
                                />
                            </div>

                            {/* Text content */}
                            <div className="w-full md:w-1/2 text-left">
                                <div className={`font-mono text-sm font-bold mb-3 px-3 py-1 rounded-full inline-block bg-white/5 ${activeStep === 3 ? 'text-red-400' : activeStep === 2 ? 'text-orange-400' : activeStep === 1 ? 'text-yellow-400' : 'text-green-400'}`}>
                                    {steps[activeStep].time}
                                </div>
                                <h3 className="text-3xl md:text-4xl font-bold mb-6 text-white leading-tight">{steps[activeStep].title}</h3>
                                <p className="text-lg text-gray-300 leading-relaxed mb-6">
                                    {steps[activeStep].desc}
                                </p>

                                {/* Techniques List */}
                                <div className="space-y-2 mb-6">
                                    {steps[activeStep].techniques.map((t, i) => (
                                        <div key={i} className="flex items-center gap-2 text-sm font-mono text-gray-400">
                                            <span className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></span>
                                            <span>{t}</span>
                                        </div>
                                    ))}
                                </div>

                                {activeStep === 3 && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        className="bg-red-500/10 border border-red-500/20 p-6 rounded-xl relative overflow-hidden"
                                    >
                                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-500"></div>
                                        <h5 className="text-red-400 font-bold mb-2 text-sm uppercase tracking-wider flex items-center gap-2">
                                            <Trash2 size={16} /> Key Insight
                                        </h5>
                                        <p className="text-red-100/80 text-base">The adversary didn't break in — they logged in. Success relied on legitimate enterprise tools (Exchange, PowerShell, Identity), not loud malware.</p>
                                    </motion.div>
                                )}
                            </div>
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </section>
    );
};

export default KillChainWalkthrough;
