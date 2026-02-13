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
        { id: "T1566.002", name: "Spearphishing Link", tactic: "Initial Access", desc: "Targeted email with a link to a payload or credential theft page. Low cost, high success rate, easy to tailor per target." },
        { id: "T1189", name: "Drive-by Compromise", tactic: "Initial Access", desc: "Compromised legitimate website serves malicious content. Victim trusts the domain, bypassing reputation-based defenses." },
        { id: "T1078", name: "Valid Accounts", tactic: "Initial Access", desc: "Stolen credentials used to log in 'normally' via VPN/O365. Looks like real user activity — bypasses controls without MFA." },
        { id: "T1003.001", name: "LSASS Credential Dump", tactic: "Credential Access", desc: "Targets LSASS memory which holds credential material. Tools like Mimikatz automate extraction of domain-wide credentials." },
        { id: "T1569.002", name: "Service Execution", tactic: "Execution", desc: "Remote execution via services (PsExec) for lateral movement across enterprise systems. Fast and often allowed." },
        { id: "T1098.002", name: "Email Delegate Access", tactic: "Persistence", desc: "Manipulates mailbox permissions so one compromised account can access many mailboxes. Stealthy — uses legitimate admin functionality." },
        { id: "T1114", name: "Email Collection", tactic: "Collection", desc: "Reads and steals hundreds of emails from hundreds of mailboxes — high-value intelligence collection without dropping malware." },
        { id: "T1070.004", name: "File Deletion", tactic: "Defense Evasion", desc: "Deletes artifacts (e.g., Mimikatz binary) after execution. Reduces forensic evidence and makes incident response harder." },
    ];

    const software = [
        { id: "S0192", name: "PupyRAT", desc: "Modular RAT/backdoor for remote access and C2 control once executed on host." },
        { id: "S0002", name: "Mimikatz", desc: "Credential dumping tool — extracts credentials from Windows LSASS memory." },
        { id: "S0029", name: "PsExec", desc: "Sysinternals remote execution utility — runs commands on remote Windows hosts for lateral movement." },
        { id: "T1059.001", name: "PowerShell", desc: "Key execution engine. Exchange cmdlets like Add-MailboxPermission used to expand mailbox access stealthily." },
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
                <p className="text-xl text-gray-400 max-w-3xl mx-auto">MITRE ATT&CK techniques and software weaponized by APT35 — mapped from the documented case study.</p>
            </div>

            {/* TTPs Section — Full width list */}
            <div className="max-w-6xl mx-auto px-4 mb-20">
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <h3 className="text-2xl font-bold text-red-400 mb-8 font-mono tracking-tight">TTPs (Tactics, Techniques & Procedures)</h3>
                    <div className="space-y-1">
                        {ttps.map((t, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: i * 0.06 }}
                                className="group bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.05] hover:border-red-500/20 rounded-xl p-5 transition-all duration-300 cursor-default"
                            >
                                <div className="flex flex-col md:flex-row md:items-center gap-3 md:gap-6">
                                    <span className="font-mono text-xs text-red-400 bg-red-500/10 px-3 py-1 rounded-full w-fit flex-shrink-0">{t.id}</span>
                                    <span className="font-mono text-xs text-gray-500 w-32 flex-shrink-0">{t.tactic}</span>
                                    <span className="text-white font-semibold w-48 flex-shrink-0">{t.name}</span>
                                    <span className="text-gray-400 text-sm leading-relaxed">{t.desc}</span>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </div>

            {/* Software Section — Cards */}
            <div className="max-w-6xl mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                >
                    <h3 className="text-2xl font-bold text-cyan-400 mb-8 font-mono tracking-tight">Software & Tools</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {software.map((s, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: i * 0.1 }}
                                className="bg-white/[0.03] border border-cyan-500/10 hover:border-cyan-500/30 rounded-2xl p-6 group hover:bg-cyan-500/[0.03] transition-all duration-300"
                            >
                                <span className="font-mono text-xs text-cyan-400 bg-cyan-500/10 px-3 py-1 rounded-full">{s.id}</span>
                                <h4 className="text-xl font-bold text-white mt-4 mb-2">{s.name}</h4>
                                <p className="text-gray-400 text-sm leading-relaxed">{s.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </div>
        </section>
    );
};

export default AttackArsenal;
