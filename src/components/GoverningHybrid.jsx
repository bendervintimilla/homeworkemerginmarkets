import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Fingerprint, Shield, Eye, Radio, ClipboardCheck, AlertTriangle, Lock, Server, FileX, ChevronDown, Search, BarChart3 } from 'lucide-react';

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

/* ────────────────────────── Governance Failure Item ────────────────────────── */
const GovernanceFailureItem = ({ title, desc, delay }) => (
    <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay }}
        className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-6 hover:bg-white/[0.04] transition-all duration-300"
    >
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
const Pillar = ({ title, subtitle, icon: Icon, desc, color, delay, number }) => (
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
            {subtitle && (
                <p className="text-xs font-mono uppercase tracking-wider text-white/50 mb-2">{subtitle}</p>
            )}
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

                {/* ─── Section Title — Andres's Part ─── */}
                <motion.div
                    className="text-center mb-6"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <p className="text-xs font-mono text-red-400 tracking-[0.3em] uppercase mb-4">4 · Technical Impact & Governance Failures</p>
                    <h2 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight">Impact, Governance & Defense</h2>
                    <p className="text-sm text-gray-600 font-mono mb-6">Andres</p>
                    <p className="text-lg text-gray-500 max-w-2xl mx-auto">Why the organization failed — and how to build resilience against trust-based attacks.</p>
                </motion.div>

                {/* ═══════════════════════ SLIDE 12: TECHNICAL IMPACT ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-24 mt-16"
                >
                    <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 rounded-2xl bg-red-500/10">
                            <AlertTriangle size={24} className="text-red-400" />
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold">Technical Impact: Scaling the APT35 Blast Radius</h3>
                        </div>
                    </div>
                    <p className="text-gray-500 text-sm mb-8 ml-16">The impact was a systematic abuse of trust, not just a malware infection.</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <ImpactCard
                            icon={Lock}
                            title="Credential Exposure (T1003)"
                            desc="LSASS dumping turned a single compromise into domain-wide risk (500+ systems)."
                            color="text-red-400"
                            delay={0}
                        />
                        <ImpactCard
                            icon={Fingerprint}
                            title="Identity Deception (T1078)"
                            desc="Valid account abuse rendered authentication policies ineffective."
                            color="text-orange-400"
                            delay={0.1}
                        />
                        <ImpactCard
                            icon={Server}
                            title="Email Control Plane Abuse (T1098.002)"
                            desc="Automated intelligence collection via legitimate Exchange delegation."
                            color="text-cyan-400"
                            delay={0.2}
                        />
                        <ImpactCard
                            icon={FileX}
                            title="Forensic Erosion (T1070.004)"
                            desc="Intentional artifact deletion hindered incident reconstruction."
                            color="text-purple-400"
                            delay={0.3}
                        />
                    </div>
                </motion.div>

                {/* ═══════════════════════ SLIDE 13: GOVERNANCE BREAKDOWN ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-16"
                >
                    <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 rounded-2xl bg-orange-500/10">
                            <Shield size={24} className="text-orange-400" />
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold">Governance Breakdown — Policy vs. Implementation</h3>
                        </div>
                    </div>
                    <p className="text-gray-500 text-sm mb-8 ml-16">Security failed because policy intent was not translated into enforceable technical controls.</p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <GovernanceFailureItem
                            title="Inadequate Enforcement of Access Control"
                            desc="VPN policies existed, but Conditional Access (location/travel) was not technically enforced."
                            delay={0}
                        />
                        <GovernanceFailureItem
                            title="Gaps in Technical Implementation"
                            desc='Policies required "authorized access," but lacked Just-In-Time (JIT) elevation for high-impact admin tasks.'
                            delay={0.1}
                        />
                        <GovernanceFailureItem
                            title="Remote Execution Failures"
                            desc="Restrictions on tools like PsExec were documented but not technically blocked on user workstations."
                            delay={0.2}
                        />
                    </div>
                </motion.div>

                {/* ═══════════════════════ SLIDE 14: ACCOUNTABILITY FAILURES ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-16"
                >
                    <div className="glass-panel rounded-2xl border border-white/[0.08] p-6 md:p-8">
                        <h4 className="text-lg font-semibold text-white mb-2">Accountability Failures — Ownership & Response</h4>
                        <p className="text-gray-500 text-sm mb-6">Without defined ownership and escalation authority, detection without response is meaningless.</p>
                        <AccountabilityItem
                            title="Undefined Asset Ownership"
                            desc='No clear "Data Owner" responsible for auditing mailbox delegation changes.'
                        />
                        <AccountabilityItem
                            title="Segregation of Duties Failure"
                            desc="Standard user accounts retained the capability to execute administrative-level remote tools."
                        />
                        <AccountabilityItem
                            title="Response Authority Gaps"
                            desc="Security operations lacked the clear authority to contain the attack during lateral movement."
                        />
                    </div>
                </motion.div>

                {/* ═══════════════════════ SLIDE 15: OVERSIGHT GAPS ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-16"
                >
                    <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 rounded-2xl bg-cyan-500/10">
                            <Search size={24} className="text-cyan-400" />
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold">Oversight Gaps — Visibility Without Action</h3>
                        </div>
                    </div>
                    <p className="text-gray-500 text-sm mb-8 ml-16">Data without analysis is noise. Controls without context are theater.</p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <GovernanceFailureItem
                            title="Signature-Based Bias"
                            desc='Oversight was biased toward malware detection, allowing "Living-off-the-Land" (PowerShell/OWA) to blend in.'
                            delay={0}
                        />
                        <GovernanceFailureItem
                            title="The Logging vs. Auditing Gap"
                            desc="Logging: technical data was captured (success). Auditing: governance failed to mandate review of anomalous mailbox changes (failure)."
                            delay={0.1}
                        />
                        <GovernanceFailureItem
                            title="Control Ineffectiveness"
                            desc='Security controls were "visible" but not "interpreted" in a risk context.'
                            delay={0.2}
                        />
                    </div>
                </motion.div>

                {/* ═══════════════════════ SLIDE 16: STRATEGIC ASSESSMENT ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.8 }}
                    className="mb-24"
                >
                    <div className="relative glass-panel rounded-3xl border border-red-500/20 overflow-hidden p-10 md:p-12">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-red-500 via-orange-500 to-purple-500"></div>

                        <div className="flex items-center gap-4 mb-6">
                            <div className="p-3 rounded-2xl bg-red-500/10">
                                <BarChart3 size={24} className="text-red-400" />
                            </div>
                            <h3 className="text-2xl md:text-3xl font-bold">Strategic Assessment — Governance as the Primary Vulnerability</h3>
                        </div>

                        {/* Main quote */}
                        <p className="text-2xl md:text-3xl font-bold text-white mb-8 leading-snug">
                            The adversary didn't break in — <br />
                            <span className="text-red-400">they logged in.</span>
                        </p>
                        <p className="text-lg text-gray-400 mb-8">Governance was the primary vulnerability.</p>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div>
                                <h4 className="text-white font-semibold mb-2">Not a Malware Problem</h4>
                                <p className="text-gray-400 text-sm leading-relaxed">Success relied on legitimate enterprise tools (Exchange, PowerShell, Identity).</p>
                            </div>
                            <div>
                                <h4 className="text-white font-semibold mb-2">Systemic Failure</h4>
                                <p className="text-gray-400 text-sm leading-relaxed">Governance failed to constrain trusted systems within a "Zero Trust" framework.</p>
                            </div>
                            <div>
                                <h4 className="text-white font-semibold mb-2">Final Verdict</h4>
                                <p className="text-gray-400 text-sm leading-relaxed">Cybersecurity failure occurs when policy, accountability, and oversight fail to keep pace with adversary exploitation.</p>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* ─────── Divider: Mattia's Section ─────── */}
                <div className="flex items-center gap-6 max-w-5xl mx-auto my-20">
                    <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                    <div className="text-center">
                        <span className="text-xs font-mono text-gray-500 tracking-widest uppercase block">5 · Strategic Defense & Organizational Resilience</span>
                        <span className="text-xs font-mono text-gray-600 mt-1 block">Mattia</span>
                    </div>
                    <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                </div>

                {/* ═══════════════════════ SLIDES 17-19: DEFENSE STRATEGY (MATTIA) ═══════════════════════ */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-12"
                >
                    <p className="text-lg text-gray-400 max-w-3xl mx-auto">APT35 succeeds by abusing trust — not by deploying loud malware. The defense must be equally trust-focused: lock down identity, govern admin actions, monitor behavior, and make oversight enforceable.</p>
                </motion.div>

                {/* Defense Pillar Cards — A & B */}
                <div className="flex flex-col md:flex-row gap-5 max-w-5xl mx-auto mb-5">
                    <Pillar
                        number="A"
                        subtitle="Prevent entry"
                        title="Identity-Centric Security"
                        desc="Stop 'Valid Accounts' from being a free pass. Phishing-resistant MFA (FIDO2/passkeys), Conditional Access enforcement against impossible travel, and continuous credential monitoring."
                        icon={Fingerprint}
                        color="bg-red-500"
                        delay={0}
                    />
                    <Pillar
                        number="B"
                        subtitle="Limit damage if they get in"
                        title="Governance & Oversight"
                        desc="High-impact actions require ownership & approval. Assign asset & control owners. PAM + Just-In-Time admin elevation. Defined escalation paths — security must isolate accounts immediately when identity trust is broken."
                        icon={Shield}
                        color="bg-cyan-500"
                        delay={0.15}
                    />
                </div>

                {/* Defense Pillar Cards — C, D & E */}
                <div className="flex flex-col md:flex-row gap-5 max-w-5xl mx-auto mb-8">
                    <Pillar
                        number="C"
                        subtitle="Catch them moving"
                        title="Detection & Response"
                        desc="From malware detection to behavior detection. Detect patterns: delegate permissions + bulk reads, PsExec bursts, credential dumping. Logs → Enforced Auditing — the failure wasn't missing logs, it was missing mandated review."
                        icon={Eye}
                        color="bg-orange-500"
                        delay={0}
                    />
                    <Pillar
                        number="D"
                        subtitle="Keep the environment clean"
                        title="Compliance & Assurance"
                        desc="Prove controls work — regularly, with evidence. Kill chain control testing via tabletop + technical validation. Routine audits of mailbox delegation, role assignments, and CA exceptions."
                        icon={ClipboardCheck}
                        color="bg-purple-500"
                        delay={0.15}
                    />
                    <Pillar
                        number="E"
                        subtitle="Maintain organizational trust"
                        title="Communication & Trust"
                        desc="Pre-approved comms playbooks — what, when, and who approves. No delays during active intrusion. Measurable assurance: communicate concrete actions tied to observed attack methods."
                        icon={Radio}
                        color="bg-emerald-500"
                        delay={0.3}
                    />
                </div>

                {/* ─── Slide 20: Key Takeaway ─── */}
                <div className="mt-20 text-center">
                    <motion.p
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        className="text-lg text-gray-400 max-w-2xl mx-auto mb-10"
                    >
                        Defending against APT35 requires aligning <strong className="text-white">people, process, and technology</strong>
                    </motion.p>

                    <div className="flex flex-col md:flex-row gap-8 justify-center items-center mb-12">
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            whileInView={{ scale: 1, opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0 }}
                            className="text-center"
                        >
                            <div className="inline-block p-[1.5px] rounded-2xl bg-gradient-to-r from-red-500 to-orange-500 mb-3">
                                <div className="bg-black rounded-2xl px-6 py-3">
                                    <span className="text-white font-bold">IDENTITY</span>
                                </div>
                            </div>
                            <p className="text-gray-500 text-sm">Treat identity as the<br />primary perimeter</p>
                        </motion.div>
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            whileInView={{ scale: 1, opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.1 }}
                            className="text-center"
                        >
                            <div className="inline-block p-[1.5px] rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-500 mb-3">
                                <div className="bg-black rounded-2xl px-6 py-3">
                                    <span className="text-white font-bold">GOVERNANCE</span>
                                </div>
                            </div>
                            <p className="text-gray-500 text-sm">Govern high-impact<br />admin actions</p>
                        </motion.div>
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            whileInView={{ scale: 1, opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: 0.2 }}
                            className="text-center"
                        >
                            <div className="inline-block p-[1.5px] rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 mb-3">
                                <div className="bg-black rounded-2xl px-6 py-3">
                                    <span className="text-white font-bold">BEHAVIOR</span>
                                </div>
                            </div>
                            <p className="text-gray-500 text-sm">Detect behavior —<br />not just malware</p>
                        </motion.div>
                    </div>

                    <p className="text-gray-600 text-sm font-mono">
                        Group 7 &nbsp;|&nbsp; Andres · Mattia · Joyce · Juan David · Pablo
                    </p>
                </div>
            </div>
        </section>
    );
};

export default GovernanceAndDefense;
