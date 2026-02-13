import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { FileText, ShieldAlert, Zap } from 'lucide-react';

/* ─────────── TTP Card ─────────── */
const TTPCard = ({ id, title, items, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay }}
        className="flex flex-col h-full"
    >
        {/* Header Block */}
        <div className="bg-gradient-to-b from-[#0055ff] to-[#00aaff] p-6 text-center">
            <div className="bg-black/40 inline-block px-3 py-1 mb-3">
                <span className="text-white font-bold font-mono text-xl tracking-wider">{id}</span>
            </div>
            <h3 className="text-white font-bold text-lg leading-tight uppercase">{title}</h3>
        </div>

        {/* Content Block */}
        <div className="flex-1 bg-gradient-to-b from-[#00aaff] to-[#00d4ff] p-6 pt-8 text-black">
            <ul className="space-y-3">
                {items.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm font-medium leading-snug">
                        <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-black flex-shrink-0"></span>
                        <span>{item}</span>
                    </li>
                ))}
            </ul>
        </div>
    </motion.div>
);

const AttackArsenal = () => {
    const containerRef = useRef(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start end", "end start"]
    });

    const opacity = useTransform(scrollYProgress, [0, 0.2], [0, 1]);

    const techniquesPart1 = [
        {
            id: "T1566.002",
            title: "PHISHING: SPEARPHISHING LINK",
            items: [
                "Targeted, legitimate-looking email",
                "Goal: get victim to click a link that leads to a payload or credential theft",
                "Classic initial access via social engineering",
                "Why it works: low cost, high success, easy to personalize"
            ]
        },
        {
            id: "T1189",
            title: "DRIVE-BY COMPROMISE",
            items: [
                "Lure hosted on a compromised legitimate website (not attacker domain)",
                "Victim trusts the site → clicks → malicious content delivered",
                "Benefit to attacker: lowers suspicion, can bypass reputation-based defenses"
            ]
        },
        {
            id: "T1078",
            title: "VALID ACCOUNTS",
            items: [
                "Use stolen credentials to log in \"normally\" (e.g., VPN / O365)",
                "Blends in as legitimate user activity",
                "Often bypasses controls unless MFA, conditional access, and anomaly detection are strong"
            ]
        },
        {
            id: "T1003.00", // Image says T1003.00, likely typo for .001 but keeping faithful to image
            title: "OS CREDENTIAL DUMPING: LSASS MEMORY",
            items: [
                "Dump credentials from LSASS on Windows systems",
                "Enables scaling access across a domain environment",
                "Common tooling: Mimikatz automates extraction"
            ]
        }
    ];

    const techniquesPart2 = [
        {
            id: "T1569.002",
            title: "SYSTEM SERVICES: SERVICE EXECUTION",
            items: [
                "Remote execution by creating/abusing services on other hosts",
                "Enables lateral movement + mass execution",
                "Often uses PsExec: fast, commonly permitted in enterprise environments"
            ]
        },
        {
            id: "T1098.002",
            title: "ACCOUNT MANIPULATION: ADDITIONAL EMAIL DELEGATE PERMISSIONS",
            items: [
                "Modify mailbox permissions instead of deploying mailbox malware",
                "Compromised account gains access to multiple mailboxes",
                "Stealthy: uses legitimate, admin-like functionality"
            ]
        },
        {
            id: "T1114",
            title: "EMAIL COLLECTION",
            items: [
                "Collect email at scale for intelligence/data theft",
                "Access and read email across many mailboxes (potentially hundreds)",
                "High-value collection with minimal additional malware deployment"
            ]
        },
        {
            id: "T1070.004",
            title: "INDICATOR REMOVAL: FILE DELETION",
            items: [
                "Delete tools/artifacts after use (e.g., copied Mimikatz binary)",
                "Reduces forensic evidence",
                "Complicates detection and incident response"
            ]
        }
    ];

    return (
        <section ref={containerRef} className="py-32 relative overflow-hidden bg-black text-white">
            <div className="max-w-7xl mx-auto px-4 relative z-20">

                {/* ─── SLIDE 4: Technical Overview Header ─── */}
                <motion.div
                    style={{ opacity }}
                    className="mb-24"
                >
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16">
                        <div className="text-left">
                            <p className="text-xs font-mono text-cyan-400 tracking-[0.3em] uppercase mb-4">2 · Technical Overview</p>
                            <h2 className="text-5xl md:text-7xl font-bold leading-tight mb-4">
                                <span className="block text-white">APT 35 TECHNICAL</span>
                                <span className="block text-white">OVERVIEW & ATTACK</span>
                                <span className="block text-cyan-400">LIFECYCLE +</span>
                                <span className="block text-cyan-400">THREAT SCENARIO</span>
                            </h2>
                        </div>

                        <div className="relative h-full min-h-[300px] rounded-3xl overflow-hidden group">
                            <img
                                src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop"
                                alt="Cyber threat scenario"
                                className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700"
                            />
                            <div className="absolute inset-0 bg-gradient-to-r from-cyan-900/80 to-transparent"></div>
                        </div>
                    </div>

                    {/* Scenario Cards (Slide 4 bottom) */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto mb-12">
                        <div className="bg-[#1a1a1a] p-8 text-center flex flex-col items-center justify-center aspect-square md:aspect-auto md:h-48 group hover:bg-[#252525] transition-colors">
                            <FileText size={40} className="text-white mb-4 stroke-1" />
                            <h4 className="text-white font-bold">Mandiant<br />Analysis</h4>
                        </div>
                        <div className="bg-gradient-to-b from-[#0055ff] to-[#00d4ff] p-8 text-center flex flex-col items-center justify-center md:h-48 shadow-[0_0_30px_rgba(0,212,255,0.3)]">
                            <ShieldAlert size={40} className="text-white mb-4 stroke-1" />
                            <h4 className="text-white font-bold">2017 Cyber<br />Attack</h4>
                        </div>
                        <div className="bg-[#1a1a1a] p-8 text-center flex flex-col items-center justify-center md:h-48 group hover:bg-[#252525] transition-colors">
                            <Zap size={40} className="text-white mb-4 stroke-1" />
                            <h4 className="text-white font-bold">Energy<br />Company</h4>
                        </div>
                    </div>

                    <div className="text-center">
                        <p className="text-sm text-gray-500 font-mono tracking-widest uppercase">
                            Pablo Rosales & Juan David Vintimilla
                        </p>
                    </div>
                </motion.div>

                {/* ─── SLIDES 5 & 6: APT 35 TECHNIQUES ─── */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <div className="flex items-center gap-6 mb-16">
                        <div className="h-px bg-cyan-500/50 flex-1"></div>
                        <h3 className="text-3xl md:text-5xl font-bold text-center uppercase tracking-tight">
                            <span className="text-white">APT 35</span> <span className="text-cyan-400">Techniques</span>
                        </h3>
                        <div className="h-px bg-cyan-500/50 flex-1"></div>
                    </div>

                    {/* Row 1 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                        {techniquesPart1.map((t, i) => (
                            <TTPCard key={i} {...t} delay={i * 0.1} />
                        ))}
                    </div>

                    {/* Row 2 */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {techniquesPart2.map((t, i) => (
                            <TTPCard key={i} {...t} delay={0.4 + (i * 0.1)} />
                        ))}
                    </div>

                </motion.div>

                {/* ─── SLIDE 7: SOFTWARE ─── */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mt-24"
                >
                    <div className="flex items-center gap-6 mb-16">
                        <div className="h-px bg-cyan-500/50 flex-1"></div>
                        <h3 className="text-3xl md:text-5xl font-bold text-center uppercase tracking-tight">
                            <span className="text-white">APT 35</span> <span className="text-cyan-400">Software</span>
                        </h3>
                        <div className="h-px bg-cyan-500/50 flex-1"></div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {/* 1. PupyRAT - Dark */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0 }}
                            className="bg-[#1a1a1a] p-8 flex flex-col items-center text-center h-full group hover:bg-[#252525] transition-colors"
                        >
                            <h4 className="text-white font-bold text-lg mb-4">S0192 – Pupy<br />(PupyRAT)</h4>
                            <p className="text-gray-400 text-sm leading-relaxed">
                                A modular RAT/backdoor used for remote access and control once executed on a host.
                            </p>
                        </motion.div>

                        {/* 2. Mimikatz - Gradient */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 }}
                            className="bg-gradient-to-b from-[#0055ff] to-[#00d4ff] p-8 flex flex-col items-center text-center h-full shadow-[0_0_30px_rgba(0,212,255,0.2)]"
                        >
                            <h4 className="text-white font-bold text-lg mb-4">S0002 – Mimikatz</h4>
                            <p className="text-white text-sm leading-relaxed font-medium">
                                Credential dumping tool used to extract credentials from Windows—commonly tied to LSASS dumping.
                            </p>
                        </motion.div>

                        {/* 3. PsExec - Gradient */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.2 }}
                            className="bg-gradient-to-b from-[#0055ff] to-[#00d4ff] p-8 flex flex-col items-center text-center h-full shadow-[0_0_30px_rgba(0,212,255,0.2)]"
                        >
                            <h4 className="text-white font-bold text-lg mb-4">S0029 – PsExec</h4>
                            <p className="text-white text-sm leading-relaxed font-medium">
                                PsExec (Microsoft Sysinternals) is a remote execution utility used to run commands on remote Windows hosts, often to enable lateral movement.
                            </p>
                        </motion.div>

                        {/* 4. PowerShell - Dark */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.3 }}
                            className="bg-[#1a1a1a] p-8 flex flex-col items-center text-center h-full group hover:bg-[#252525] transition-colors"
                        >
                            <h4 className="text-white font-bold text-lg mb-4">T1059.001 –<br />PowerShell</h4>
                            <p className="text-gray-400 text-sm leading-relaxed">
                                Not a “software entry” but a key execution method: PowerShell is used to run commands and administrative actions.
                            </p>
                        </motion.div>
                    </div>
                </motion.div>

            </div>
        </section>
    );
};

export default AttackArsenal;
