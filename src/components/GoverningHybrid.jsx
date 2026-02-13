import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Fingerprint, Shield, Eye, Radio, ClipboardCheck, AlertTriangle, Lock, Server, FileX, ChevronDown } from 'lucide-react';

/* ────────────────────────── Governance Failure Row ────────────────────────── */
const FailureRow = ({ technique, policy, breakdown, delay }) => (
    <motion.tr
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay }}
        className="border-b border-white/5 hover:bg-white/[0.02] transition-colors"
    >
        <td className="py-4 px-4 font-mono text-xs text-red-400">{technique}</td>
        <td className="py-4 px-4 text-sm text-gray-300">{policy}</td>
        <td className="py-4 px-4 text-sm text-gray-400">{breakdown}</td>
    </motion.tr>
);

/* ────────────────────────── Impact Card ────────────────────────── */
const ImpactCard = ({ icon: Icon, title, desc, color, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay }}
        className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-6 hover:bg-white/[0.04] hover:border-white/10 transition-all duration-300"
    >
        <div className={`p-2 rounded-xl bg-white/5 w-fit mb-4`}>
            <Icon size={20} className={color} />
        </div>
        <h4 className="text-white font-semibold mb-2">{title}</h4>
        <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
    </motion.div>
);

/* ────────────────────────── Expandable Accountability ────────────────────────── */
const AccountabilityItem = ({ title, desc }) => {
    const [open, setOpen] = useState(false);
    return (
        <div className="border-b border-white/5">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between py-4 text-left group"
            >
                <span className="text-white font-medium group-hover:text-red-400 transition-colors">{title}</span>
                <ChevronDown size={16} className={`text-gray-500 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                    >
                        <p className="text-gray-400 text-sm leading-relaxed pb-4 pl-4 border-l-2 border-red-500/30">{desc}</p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

/* ────────────────────────── Defense Pillar ────────────────────────── */
const Pillar = ({ title, icon: Icon, desc, color, delay, number }) => (
    <motion.div
        initial={{ y: 60, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
        className="flex-1 relative group cursor-pointer"
    >
        <div className={`absolute -top-px left-0 right-0 h-[3px] ${color} rounded-full opacity-80 group-hover:opacity-100 group-hover:shadow-[0_0_20px_currentColor] transition-all duration-500`}></div>
        <div className="bg-white/[0.03] backdrop-blur-sm border border-white/[0.06] rounded-2xl p-8 h-full flex flex-col items-start text-left group-hover:bg-white/[0.06] group-hover:border-white/10 transition-all duration-500 group-hover:-translate-y-2">
            <span className="text-[80px] font-bold leading-none text-white/[0.04] absolute top-4 right-6 select-none">{number}</span>
            <div className={`p-3 rounded-2xl bg-white/[0.05] mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <Icon size={24} className="text-gray-300" />
            </div>
            <h3 className="text-xl font-semibold mb-3 text-white tracking-tight">{title}</h3>
            <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
        </div>
    </motion.div>
);

/* ════════════════════════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                                     */
/* ════════════════════════════════════════════════════════════════════ */
const GovernanceAndDefense = () => {
    return (
        <section className="py-32 px-4 relative overflow-hidden">
            <div className="max-w-6xl mx-auto">

                {/* ─── Section Title ─── */}
                <motion.div
                    className="text-center mb-20"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <h2 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight">Impact, Governance & Defense</h2>
                    <p className="text-lg text-gray-500 max-w-2xl mx-auto">Why the organization failed — and how to build resilience against trust-based attacks.</p>
                </motion.div>

                {/* ═══════════════════════ PART A: TECHNICAL IMPACT ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-24"
                >
                    <div className="flex items-center gap-4 mb-8">
                        <div className="p-3 rounded-2xl bg-red-500/10">
                            <AlertTriangle size={24} className="text-red-400" />
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold">Technical Impact — The "Blast Radius"</h3>
                            <p className="text-gray-500 text-sm mt-1">The impact expanded rapidly through the systematic abuse of trust across identity, email, and administrative control planes.</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <ImpactCard
                            icon={Lock}
                            title="Enterprise-Wide Credential Exposure"
                            desc="LSASS credential dumping (T1003) converted a single endpoint compromise into a domain-wide identity risk. Credential material was implicitly trusted on standard workstations."
                            color="text-red-400"
                            delay={0}
                        />
                        <ImpactCard
                            icon={Fingerprint}
                            title="Identity Deception"
                            desc="Valid Accounts (T1078) for VPN and O365 access made malicious behavior indistinguishable from legitimate access — a total loss of trust in identity as a security control."
                            color="text-orange-400"
                            delay={0.1}
                        />
                        <ImpactCard
                            icon={Server}
                            title="Scalable Intelligence Collection"
                            desc="Manipulation of Email Delegate Permissions (T1098.002) transformed legitimate Exchange functionality into an automated intelligence-collection mechanism at scale."
                            color="text-cyan-400"
                            delay={0.2}
                        />
                        <ImpactCard
                            icon={FileX}
                            title="Forensic Erosion"
                            desc="File Deletion (T1070.004) removed execution artifacts, obstructing the ability to reconstruct lateral movement scope and identify initial compromise paths."
                            color="text-purple-400"
                            delay={0.3}
                        />
                    </div>
                </motion.div>

                {/* ═══════════════════════ PART B: GOVERNANCE BREAKDOWN ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-12"
                >
                    <div className="flex items-center gap-4 mb-8">
                        <div className="p-3 rounded-2xl bg-orange-500/10">
                            <Shield size={24} className="text-orange-400" />
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold">Governance Breakdown — Policy vs. Reality</h3>
                            <p className="text-gray-500 text-sm mt-1">APT35 exploited the gap between policy intent and operational enforcement.</p>
                        </div>
                    </div>

                    {/* Governance Failure Table */}
                    <div className="overflow-x-auto rounded-2xl border border-white/[0.08] bg-white/[0.02] mb-10">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="border-b border-white/10 bg-white/[0.03]">
                                    <th className="py-3 px-4 text-xs font-mono text-red-400 tracking-wider uppercase">Technique</th>
                                    <th className="py-3 px-4 text-xs font-mono text-gray-400 tracking-wider uppercase">Documented Policy</th>
                                    <th className="py-3 px-4 text-xs font-mono text-orange-400 tracking-wider uppercase">Governance Breakdown</th>
                                </tr>
                            </thead>
                            <tbody>
                                <FailureRow
                                    technique="T1078 — Valid Accounts"
                                    policy={`"Only authorized users may access VPN."`}
                                    breakdown="Conditional Access not enforced against unusual VPN endpoints or improbable travel."
                                    delay={0}
                                />
                                <FailureRow
                                    technique="T1098.002 — Mailbox Permissions"
                                    policy={`"Data access is restricted."`}
                                    breakdown="No Data Owner assigned to audit/approve delegation changes. Static persistent permissions."
                                    delay={0.05}
                                />
                                <FailureRow
                                    technique="S0029 (PsExec) & T1003 (LSASS)"
                                    policy={`"Administrative tools restricted."`}
                                    breakdown="Standard user accounts retained technical rights to execute admin tools (SoD failure)."
                                    delay={0.1}
                                />
                                <FailureRow
                                    technique="T1114 — Email Collection"
                                    policy={`"All activity is logged."`}
                                    breakdown="Logs existed (technical) but were not reviewed (governance). No mandated alerting."
                                    delay={0.15}
                                />
                            </tbody>
                        </table>
                    </div>

                    {/* Accountability Failures — Expandable */}
                    <div className="glass-panel rounded-2xl border border-white/[0.08] p-6 md:p-8">
                        <h4 className="text-lg font-semibold text-white mb-4">Security Accountability Failures</h4>
                        <AccountabilityItem
                            title="Missing Data Ownership"
                            desc="Access to hundreds of mailboxes occurred without a clearly accountable data owner. Governance failed to assign ownership over email systems as high-value information assets requiring continuous oversight."
                        />
                        <AccountabilityItem
                            title="Lack of Segregation of Duties"
                            desc="Compromised user contexts could leverage PsExec — standard user accounts should never retain the technical capability to execute administrative-level remote tools. A structural governance flaw."
                        />
                        <AccountabilityItem
                            title="Response Authority Gaps"
                            desc="The attacker progressed through credential dumping, lateral movement, and mailbox access without timely containment. Escalation thresholds were unclear and security operations lacked authority to isolate systems during active credential abuse."
                        />
                        <AccountabilityItem
                            title="Detection Bias (Signatures over Behavior)"
                            desc="Oversight was biased toward signature-based malware detection. Because APT35 relied on living-off-the-land techniques (PowerShell, OWA, PsExec), malicious activity blended into normal administrative behavior. The failure was not the absence of data, but the absence of enforced oversight."
                        />
                    </div>
                </motion.div>

                {/* ─── Divider ─── */}
                <div className="flex items-center gap-6 max-w-5xl mx-auto my-20">
                    <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                    <span className="text-xs font-mono text-gray-500 tracking-widest uppercase">Strategic Defense</span>
                    <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                </div>

                {/* ═══════════════════════ PART C: DEFENSE STRATEGY ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-12"
                >
                    <p className="text-lg text-gray-400 max-w-2xl mx-auto">Defending against APT35 requires aligning people, process, and technology — treating identity as the primary perimeter.</p>
                </motion.div>

                {/* Defense Pillar Cards */}
                <div className="flex flex-col md:flex-row gap-5 max-w-5xl mx-auto mb-8">
                    <Pillar
                        number="01"
                        title="Identity-Centric Security"
                        desc="Phishing-resistant MFA (FIDO2/passkeys), conditional access enforcement against impossible travel, and continuous credential monitoring. Stop valid accounts from being a free pass."
                        icon={Fingerprint}
                        color="bg-red-500"
                        delay={0}
                    />
                    <Pillar
                        number="02"
                        title="Governance & Oversight"
                        desc="Assign asset/control owners. Implement PAM + Just-In-Time admin elevation. Define escalation paths so security can isolate accounts immediately when identity trust is broken."
                        icon={Shield}
                        color="bg-cyan-500"
                        delay={0.15}
                    />
                    <Pillar
                        number="03"
                        title="Detection & Response"
                        desc="Behavioral monitoring over signatures. Detect patterns: new delegate permissions + bulk mailbox reads, PsExec bursts, credential dump indicators. Enforce mandated review, not just logging."
                        icon={Eye}
                        color="bg-orange-500"
                        delay={0.3}
                    />
                </div>

                <div className="flex flex-col md:flex-row gap-5 max-w-5xl mx-auto">
                    <Pillar
                        number="04"
                        title="Compliance & Assurance"
                        desc="Kill chain control testing via tabletop + technical validation. Routine audits of mailbox delegation, role assignments, and CA exceptions. Prove controls work — regularly, with evidence."
                        icon={ClipboardCheck}
                        color="bg-purple-500"
                        delay={0.15}
                    />
                    <Pillar
                        number="05"
                        title="Communication & Trust"
                        desc="Pre-approved comms playbooks — no delays during active intrusion. Stakeholder assurance with measurable actions tied directly to observed intrusion methods."
                        icon={Radio}
                        color="bg-emerald-500"
                        delay={0.3}
                    />
                </div>

                {/* ─── Final Insight ─── */}
                <div className="mt-20 text-center">
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        whileInView={{ scale: 1, opacity: 1 }}
                        viewport={{ once: true }}
                        className="inline-block p-[1.5px] rounded-full bg-gradient-to-r from-red-500 via-purple-500 to-cyan-500"
                    >
                        <div className="bg-black rounded-full px-10 py-5">
                            <span className="text-xl md:text-2xl font-semibold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
                                The adversary didn't break in — they logged in.
                            </span>
                        </div>
                    </motion.div>
                    <p className="mt-6 text-gray-500 text-sm max-w-xl mx-auto">
                        Cybersecurity failure is often not a technological problem, but a governance problem — where policy, accountability, and oversight fail to keep pace with how adversaries exploit legitimate access.
                    </p>
                </div>
            </div>
        </section>
    );
};

export default GovernanceAndDefense;
