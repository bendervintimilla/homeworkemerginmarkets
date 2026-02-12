import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const AttackArsenal = () => {
    const containerRef = useRef(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start end", "end start"]
    });

    const opacity = useTransform(scrollYProgress, [0, 0.3], [0, 1]);

    const ttps = [
        { id: "T1566.002", name: "Spearphishing Link", desc: "Trusted domains deliver malware" },
        { id: "T1189", name: "Drive-by Compromise", desc: "Watering hole attacks on legit sites" },
        { id: "T1078", name: "Valid Accounts", desc: "Stolen credentials bypass controls" },
        { id: "T1003.001", name: "LSASS Credential Dump", desc: "Harvest domain-wide credentials" },
        { id: "T1569.002", name: "Service Execution", desc: "PsExec for lateral movement" },
        { id: "T1098.002", name: "Email Delegate Access", desc: "Mailbox permissions manipulation" },
    ];

    const software = [
        { id: "S0192", name: "PupyRAT", desc: "Modular RAT for remote access and C2 control" },
        { id: "S0002", name: "Mimikatz", desc: "Credential dumping from Windows LSASS memory" },
        { id: "S0029", name: "PsExec", desc: "Remote execution utility for lateral movement" },
        { id: "T1059.001", name: "PowerShell", desc: "Key execution engine for Exchange cmdlets" },
    ];

    return (
        <section ref={containerRef} className="py-32 relative overflow-hidden bg-black">
            <div className="max-w-7xl mx-auto px-4 mb-20 relative z-20 text-center">
                <motion.h2
                    style={{ opacity }}
                    className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-white to-cyan-500"
                >
                    Attack Arsenal
                </motion.h2>
                <p className="text-xl text-gray-400 max-w-3xl mx-auto">MITRE ATT&CK techniques and software weaponized by APT35 for credential theft, persistence, and intelligence collection.</p>
            </div>

            {/* Two-card layout */}
            <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 md:grid-cols-2 gap-8">

                {/* Left Card: TTPs */}
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
                            src="/assets/technique_credential.png"
                            alt="APT35 TTPs — Credential Access"
                            className="absolute inset-0 w-full h-full object-cover transition-transform duration-[3s] group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/70 to-transparent"></div>
                    </div>

                    <div className="absolute bottom-0 left-0 right-0 p-10">
                        <h3 className="text-4xl md:text-5xl font-bold text-red-500 mb-6 tracking-tight">TTPs</h3>
                        <div className="space-y-4 text-sm font-mono text-gray-300">
                            {ttps.map((t, i) => (
                                <div key={i} className="flex justify-between border-b border-white/10 pb-3">
                                    <span className="text-gray-500">{t.id}</span>
                                    <span className={i === 0 ? "text-red-400 font-bold" : ""}>{t.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </motion.div>

                {/* Right Card: Software */}
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
                            src="/assets/technique_email.png"
                            alt="APT35 Software — Email Collection"
                            className="absolute inset-0 w-full h-full object-cover transition-transform duration-[3s] group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-blue-950 via-blue-950/60 to-transparent"></div>
                    </div>

                    <div className="absolute bottom-0 left-0 right-0 p-10">
                        <h3 className="text-4xl md:text-5xl font-bold text-cyan-400 mb-6 tracking-tight">SOFTWARE</h3>
                        <div className="space-y-4 text-sm font-mono text-gray-300">
                            {software.map((s, i) => (
                                <div key={i} className="flex justify-between border-b border-cyan-500/20 pb-3">
                                    <span className="text-gray-500">{s.id}</span>
                                    <span className={i === 0 ? "text-cyan-300 font-bold" : ""}>{s.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default AttackArsenal;
