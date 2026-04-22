"use client";

import React, { useEffect, useState } from "react";
import {
    ArrowLeft, CheckCircle2, AlertCircle, ShieldAlert,
    Play, Volume2, ChevronDown, Activity, Brain, BarChart, Loader2, Eye, RotateCcw
} from "lucide-react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area,
    BarChart as RechartsBarChart, Bar, PieChart as RechartsPieChart, Pie, Cell,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Session { session_id: string; candidate_name: string; job_opening_id: string; created_at: string }

interface QuestionResponse {
    id: number;
    question_id: string;
    question_text: string;
    ideal_answer: string | null;
    transcript: string;
    semantic_score: number;
    sentiment: Record<string, number>;
    combined_score: number;
    technical_score: number | null;
    communication_score: number | null;
    behavioral_score: number | null;
    engagement_score: number | null;
    authenticity_score: number | null;
    video_file_id: string | null;
    audio_file_id: string | null;
    video_url: string | null;
    audio_url: string | null;
    transcript_flagged: boolean;
    llm_verdict: string | null;
    llm_verdict_reason: string | null;
    llm_key_gaps: string[] | null;
    llm_strengths: string[] | null;
}

interface GazeMetrics {
    provider: string;
    status: string;
    gaze_points_count?: number;
    zone_distribution?: Record<string, number>;
    robotic_reading?: { detected: boolean; confidence: number; reversal_rate: number; y_stdev: number };
    cheat_flags?: {
        risk_level: string;
        signals?: Record<string, unknown>;
        [key: string]: unknown;
    };
    offscreen_ratio?: number;
}

interface VideoSignal {
    question_id: string;
    gaze_zone_distribution: Record<string, number>;
    cheat_flags: Record<string, unknown>;
    emotion_distribution: Record<string, number>;
    avg_hrv_rmssd: number;
    hr_bpm: number | null;
    stress_spike_detected: boolean;
    gaze_metrics: GazeMetrics | null;
}

interface OceanReport {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
    job_fit_score: number;
    success_prediction: string;
    role_recommendation: string;
    ocean_confidence?: string;
    trait_coverage?: Record<string, string>;   // "full" | "partial" | "limited" | "none"
    stages_covered?: string[];
}

interface FullReport {
    session: Session | null;
    question_responses: QuestionResponse[];
    video_signals: VideoSignal[];
    ocean_report: OceanReport | null;
    interview_completed: boolean;
}

const OCEAN_COLORS = ["#6C63FF","#22C55E","#F59E0B","#3B82F6","#EF4444"];
const PIE_COLORS   = ["#6C63FF","#22C55E","#F59E0B","#EF4444","#3B82F6","#8B5CF6","#06B6D4","#F97316"];

// ── Trait coverage helpers ─────────────────────────────────────────────────────
const COVERAGE_BADGE: Record<string, { label: string; cls: string }> = {
    full:    { label: "Full data",    cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
    partial: { label: "Partial data", cls: "bg-amber-500/10  text-amber-400  border-amber-500/20"  },
    limited: { label: "Limited data", cls: "bg-orange-500/10 text-orange-400 border-orange-500/20" },
    none:    { label: "No data",      cls: "bg-slate-700/50  text-slate-500  border-slate-600/20"  },
};

// Which question stages provide meaningful signal for each trait
const TRAIT_STAGE_HINT: Record<string, string> = {
    openness:          "technical, logical, behavioral, situational",
    conscientiousness: "technical, logical, behavioral, situational",
    extraversion:      "behavioral, situational",
    agreeableness:     "behavioral, situational",
    neuroticism:       "behavioral, situational, technical, logical",
};

export default function CandidateInsight({ params }: { params: Promise<{ id: string }> }) {
    const { id } = React.use(params);
    const [report, setReport]       = useState<FullReport | null>(null);
    const [loading, setLoading]     = useState(true);
    const [activeTab, setActiveTab] = useState("overview");
    const [expandedQ, setExpandedQ] = useState<number | null>(null);
    const [refreshingRec, setRefreshingRec] = useState(false);

    useEffect(() => {
        (async () => {
            try {
                const res = await fetch(`${API}/session/${id}/report`);
                if (res.ok) setReport(await res.json());
            } catch (e) {
                console.error("[Examiney][CandidateInsight]", e);
            } finally {
                setLoading(false);
            }
        })();
    }, [id]);

    const handleRefreshRecommendation = async () => {
        setRefreshingRec(true);
        console.log("[Examiney][RefreshRec] 🔄 Starting AI recommendation refresh for session:", id);
        try {
            // Trigger re-processing
            console.log("[Examiney][RefreshRec] 📤 Triggering POST /session/{id}/process...");
            const processRes = await fetch(`${API}/session/${id}/process`, { method: 'POST' });
            console.log("[Examiney][RefreshRec] 📡 Process response status:", processRes.status);
            
            if (!processRes.ok) {
                const errText = await processRes.text();
                console.error("[Examiney][RefreshRec] ❌ Failed to start processing:", errText);
                throw new Error(`Failed to start processing: ${processRes.status}`);
            }
            
            const processData = await processRes.json();
            console.log("[Examiney][RefreshRec] ✓ Processing started:", processData);
            
            // Poll for completion
            console.log("[Examiney][RefreshRec] ⏳ Polling for completion (max 60 attempts)...");
            let isReady = false;
            let attempts = 0;
            while (!isReady && attempts < 60) {
                await new Promise(r => setTimeout(r, 1000)); // Wait 1 second
                const statusRes = await fetch(`${API}/session/${id}/status`);
                
                if (statusRes.ok) {
                    const statusData = await statusRes.json();
                    console.log(`[Examiney][RefreshRec] Status attempt ${attempts + 1}/60 - Stage: ${statusData.stage} (${statusData.progress}%)`);
                    
                    if (statusData.status === 'ready') {
                        console.log("[Examiney][RefreshRec] ✓ Processing complete! Status:", statusData.status);
                        isReady = true;
                    }
                } else {
                    console.warn(`[Examiney][RefreshRec] ⚠️ Status check failed with HTTP ${statusRes.status}`);
                }
                attempts++;
            }
            
            if (!isReady) {
                console.warn("[Examiney][RefreshRec] ⚠️ Polling timeout - processing may still be running");
            }
            
            // Refresh the report
            console.log("[Examiney][RefreshRec] 📥 Fetching updated report...");
            const reportRes = await fetch(`${API}/session/${id}/report`);
            if (reportRes.ok) {
                const updatedReport = await reportRes.json();
                const newRec = updatedReport.ocean_report?.role_recommendation;
                console.log("[Examiney][RefreshRec] ✅ Report updated! New recommendation preview:", newRec?.substring(0, 150) + "...");
                setReport(updatedReport);
            } else {
                console.error("[Examiney][RefreshRec] ❌ Failed to fetch updated report:", reportRes.status);
            }
        } catch (e) {
            console.error("[Examiney][RefreshRec] 💥 Error during refresh:", {
                name: (e as Error).name,
                message: (e as Error).message,
                stack: (e as Error).stack,
            });
        } finally {
            setRefreshingRec(false);
            console.log("[Examiney][RefreshRec] 🏁 Refresh process completed");
        }
    };

    if (loading) return (
        <div className="flex items-center justify-center h-96 gap-4 text-slate-400">
            <Loader2 size={28} className="animate-spin" />
            <span className="font-ui text-sm uppercase tracking-widest">Loading report…</span>
        </div>
    );

    if (!report?.session) return (
        <div className="flex flex-col items-center justify-center h-96 gap-4 text-slate-500">
            <AlertCircle size={36} />
            <p className="font-ui text-sm uppercase tracking-widest">Session not found.</p>
            <Link href="/dashboard" className="text-primary underline text-sm">Back to Dashboard</Link>
        </div>
    );

    const { session, question_responses: qs, video_signals: vs, ocean_report: ocean } = report;

    // ── Chart data ─────────────────────────────────────────────────────────────
    const oceanData = ocean ? [
        { subject: "Openness",          A: ocean.openness,          fullMark: 100 },
        { subject: "Conscientiousness", A: ocean.conscientiousness, fullMark: 100 },
        { subject: "Extraversion",      A: ocean.extraversion,      fullMark: 100 },
        { subject: "Agreeableness",     A: ocean.agreeableness,     fullMark: 100 },
        { subject: "Neuroticism",       A: ocean.neuroticism,       fullMark: 100 },
    ] : [];

    const sentimentData = qs.map((q, i) => ({
        name: `Q${i + 1}`,
        score: Math.round((q.sentiment?.compound ?? 0) * 50 + 50),
    }));

    // Merge emotion distributions across all questions
    const emotionAgg: Record<string, number> = {};
    vs.forEach(v => Object.entries(v.emotion_distribution ?? {}).forEach(([k, val]) => {
        emotionAgg[k] = (emotionAgg[k] ?? 0) + val;
    }));
    const emotionTotal = Object.values(emotionAgg).reduce((a, b) => a + b, 0) || 1;
    const emotionPieData = Object.entries(emotionAgg).map(([name, val]) => ({
        name, value: Math.round((val / emotionTotal) * 100),
    }));

    const hrvData = vs.map((v, i) => ({ name: `Q${i + 1}`, hrv: v.avg_hrv_rmssd, hr: v.hr_bpm ?? 0 }));

    const predColor = !ocean ? "text-slate-500"
        : ocean.success_prediction === "High" ? "text-success"
        : ocean.success_prediction === "Medium" ? "text-warning" : "text-danger";
    const successPct = !ocean ? 0
        : ocean.success_prediction === "High" ? 82
        : ocean.success_prediction === "Medium" ? 55 : 30;

    // Aggregate gaze zone distribution from GazeFollower metrics across questions
    const gazeAgg: Record<string, number> = {};
    let gazeTotal = 0;
    vs.forEach(v => {
        const zones = v.gaze_metrics?.zone_distribution ?? v.gaze_zone_distribution ?? {};
        Object.entries(zones).forEach(([z, pct]) => {
            gazeAgg[z] = (gazeAgg[z] ?? 0) + (pct as number);
            gazeTotal += (pct as number);
        });
    });
    const gazeBarData = Object.entries(gazeAgg).map(([zone, total]) => ({
        zone, pct: Math.round((total / Math.max(vs.length, 1)) * 100),
    }));

    // Overall cheat risk (highest level across all questions)
    const riskRank: Record<string, number> = { low: 0, medium: 1, high: 2 };
    let overallRisk = "low";
    vs.forEach(v => {
        const gfRisk = (v.gaze_metrics?.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
        const rtRisk = (v.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
        if ((riskRank[gfRisk] ?? 0) > (riskRank[overallRisk] ?? 0)) overallRisk = gfRisk;
        if ((riskRank[rtRisk] ?? 0) > (riskRank[overallRisk] ?? 0)) overallRisk = rtRisk;
    });

    const tabs = [
        { id: "overview",   label: "Overview",       icon: Brain },
        { id: "gaze",       label: "Focus & Signals", icon: Eye },
        { id: "questions",  label: "Per-Question",   icon: BarChart },
    ];

    return (
        <div className="space-y-10">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-6">
                    <Link href={`/dashboard/openings/${session.job_opening_id}`}
                        className="p-3 bg-white border border-border rounded-xl hover:bg-gray-50 transition-colors text-foreground/50 hover:text-foreground shadow-sm">
                        <ArrowLeft size={20} />
                    </Link>
                    <div>
                        <h1 className="font-heading text-4xl font-bold">{session.candidate_name}</h1>
                        <p className="font-body text-slate-500 text-lg mt-1">
                            Session: <span className="text-foreground/60 font-mono text-sm">{session.session_id}</span>
                            {" · "}{new Date(session.created_at).toLocaleDateString()}
                        </p>
                    </div>
                </div>
                <div className={`px-4 py-2 rounded-full text-xs font-ui font-black uppercase tracking-widest border
                    ${overallRisk === "high" ? "bg-danger/10 text-danger border-danger/30"
                    : overallRisk === "medium" ? "bg-warning/10 text-warning border-warning/30"
                    : "bg-success/10 text-success border-success/30"}`}>
                    Integrity: {overallRisk.toUpperCase()}
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 bg-slate-900/50 p-1.5 rounded-2xl border border-white/5 w-fit">
                {tabs.map(({ id, label, icon: Icon }) => (
                    <button key={id} onClick={() => setActiveTab(id)}
                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-ui text-sm font-bold tracking-wide transition-all
                            ${activeTab === id ? "bg-primary text-white shadow-violet" : "text-slate-500 hover:text-foreground"}`}>
                        <Icon size={16} /> {label}
                    </button>
                ))}
            </div>

            {/* ── Overview Tab ──────────────────────────────────────────────────── */}
            {activeTab === "overview" && (
                <div className="space-y-8">
                {/* Coverage warning — shown when key traits have limited/no data */}
                {ocean?.trait_coverage && (() => {
                    const weakTraits = (["agreeableness","extraversion","neuroticism","openness","conscientiousness"] as const)
                        .filter(t => ["limited","none"].includes(ocean.trait_coverage![t] ?? "none"));
                    const missingStages = (["behavioral","situational","logical","technical"] as const)
                        .filter(s => !(ocean.stages_covered ?? []).includes(s));
                    if (weakTraits.length === 0) return null;
                    return (
                        <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 px-5 py-4 flex gap-3">
                            <AlertCircle size={18} className="text-amber-400 shrink-0 mt-0.5" />
                            <div className="text-sm">
                                <p className="text-amber-300 font-bold mb-1">
                                    Limited OCEAN coverage — {weakTraits.length} trait{weakTraits.length > 1 ? "s" : ""} based on insufficient question data
                                </p>
                                <p className="text-amber-400/70 leading-relaxed">
                                    <span className="capitalize">{weakTraits.join(", ")}</span> could not be fully assessed
                                    {missingStages.length > 0 && (
                                        <> because <span className="font-semibold">{missingStages.join(", ")}</span> question{missingStages.length > 1 ? "s were" : " was"} not included in this interview.</>
                                    )}.
                                    Scores for these traits are estimates only — treat them with caution or re-run the interview with a richer question mix.
                                </p>
                            </div>
                        </div>
                    );
                })()}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left: success ring + OCEAN radar + sentiment */}
                    <div className="lg:col-span-2 space-y-8">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                            {/* 6-month success ring */}
                            <div className="glass-card p-8 rounded-3xl border border-white/10 flex flex-col items-center justify-center gap-4">
                                <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black">6-Month Success Probability</p>
                                {!report.interview_completed ? (
                                    <div className="flex flex-col items-center justify-center gap-3 py-4">
                                        <div className="w-20 h-20 rounded-full border-4 border-dashed border-slate-700 flex items-center justify-center">
                                            <span className="font-heading text-2xl font-black text-slate-600">—</span>
                                        </div>
                                        <p className="font-ui text-xs text-slate-600 uppercase tracking-widest text-center">Interview not taken</p>
                                    </div>
                                ) : (
                                    <>
                                        <div className="relative w-36 h-36">
                                            <svg className="w-full h-full -rotate-90">
                                                <circle cx="72" cy="72" r="62" fill="transparent" stroke="#e5e7eb" strokeWidth="10" />
                                                <circle cx="72" cy="72" r="62" fill="transparent"
                                                    stroke="#6C63FF" strokeWidth="10"
                                                    strokeDasharray={`${2 * Math.PI * 62}`}
                                                    strokeDashoffset={`${2 * Math.PI * 62 * (1 - successPct / 100)}`}
                                                    strokeLinecap="round" />
                                            </svg>
                                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                                <span className="font-heading text-3xl font-black">{successPct}%</span>
                                                <span className={`font-ui text-xs font-black uppercase ${predColor}`}>{ocean?.success_prediction ?? "—"}</span>
                                            </div>
                                        </div>
                                        <p className="font-ui text-xs text-slate-500 text-center leading-relaxed max-w-[200px]">
                                            Job Fit: <span className="text-foreground font-bold">{ocean?.job_fit_score?.toFixed(1) ?? "—"}/100</span>
                                        </p>
                                    </>
                                )}
                            </div>

                            {/* OCEAN radar + coverage */}
                            <div className="glass-card p-6 rounded-3xl border border-white/10">
                                <div className="flex items-center justify-between mb-1">
                                    <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black">Big Five OCEAN</p>
                                    {ocean?.ocean_confidence && (
                                        <span className={`text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full border
                                            ${ocean.ocean_confidence === "High"   ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                            : ocean.ocean_confidence === "Medium" ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                                            :                                       "bg-slate-700/50 text-slate-500 border-slate-600/20"}`}>
                                            {ocean.ocean_confidence} confidence
                                        </span>
                                    )}
                                </div>
                                {ocean ? (
                                    <>
                                        <ResponsiveContainer width="100%" height={180}>
                                            <RadarChart data={oceanData}>
                                                <PolarGrid stroke="#e5e7eb" />
                                                <PolarAngleAxis dataKey="subject" tick={{ fill: "#6b7280", fontSize: 10 }} />
                                                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                                                <Radar name="OCEAN" dataKey="A" stroke="#6C63FF" fill="#6C63FF" fillOpacity={0.3} />
                                            </RadarChart>
                                        </ResponsiveContainer>
                                        {/* Per-trait coverage pills */}
                                        {ocean.trait_coverage && (
                                            <div className="mt-3 space-y-1.5">
                                                {(["openness","conscientiousness","extraversion","agreeableness","neuroticism"] as const).map(trait => {
                                                    const cov = ocean.trait_coverage![trait] ?? "none";
                                                    const badge = COVERAGE_BADGE[cov] ?? COVERAGE_BADGE.none;
                                                    return (
                                                        <div key={trait} className="flex items-center justify-between text-[10px]" title={`Needs: ${TRAIT_STAGE_HINT[trait]}`}>
                                                            <span className="text-slate-500 capitalize">{trait}</span>
                                                            <span className={`px-2 py-0.5 rounded-full border font-black uppercase tracking-wide ${badge.cls}`}>
                                                                {badge.label}
                                                            </span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </>
                                ) : <div className="h-48 flex items-center justify-center text-slate-600 text-sm">{report.interview_completed ? "Processing…" : "No interview data"}</div>}
                            </div>
                        </div>

                        {/* Sentiment timeline */}
                        <div className="glass-card p-8 rounded-3xl border border-white/10">
                            <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black mb-6">Sentiment Timeline</p>
                            <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={sentimentData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                    <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} />
                                    <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: 12, color: "#1f2937" }} />
                                    <Line type="monotone" dataKey="score" stroke="#6C63FF" strokeWidth={2} dot={{ fill: "#6C63FF", r: 4 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Sidebar: emotion pie + score cards + recommendation */}
                    <div className="space-y-6">
                        <div className="glass-card p-6 rounded-3xl border border-white/10">
                            <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black mb-4">Emotion Distribution</p>
                            {emotionPieData.length > 0 ? (
                                <ResponsiveContainer width="100%" height={180}>
                                    <RechartsPieChart>
                                        <Pie data={emotionPieData} dataKey="value" cx="50%" cy="50%" outerRadius={70} labelLine={false}
                                            label={({ name, percent }) => (percent as number) > 0.08 ? `${name} ${Math.round((percent as number) * 100)}%` : ""}>
                                            {emotionPieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                                        </Pie>
                                        <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: 12, color: "#1f2937" }} />
                                    </RechartsPieChart>
                                </ResponsiveContainer>
                            ) : <div className="h-44 flex items-center justify-center text-slate-600 text-sm">No emotion data</div>}
                        </div>

                        {[
                            { label: "Cognitive Load Resilience", value: ocean ? Math.round(100 - ocean.neuroticism) : null, color: "text-primary" },
                            {
                                label: "Authenticity Baseline",
                                value: (() => {
                                    const scores = qs.map(q => q.authenticity_score).filter(s => s !== null && s !== undefined);
                                    if (scores.length === 0) return null;
                                    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
                                    return Math.round(avg * 10);
                                })(),
                                color: "text-success"
                            },
                            {
                                label: "Logical Efficiency",
                                value: (() => {
                                    const scores = qs.map(q => q.technical_score).filter(s => s !== null && s !== undefined);
                                    if (scores.length === 0) return null;
                                    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
                                    return Math.round(avg * 10);
                                })(),
                                color: "text-warning"
                            },
                        ].map(({ label, value, color }) => (
                            <div key={label} className="glass-card p-6 rounded-3xl border border-white/10">
                                <p className="font-ui text-[10px] text-slate-500 uppercase tracking-widest font-black mb-3">{label}</p>
                                <p className={`font-heading text-4xl font-black ${color}`}>{value !== null ? `${value}%` : "—"}</p>
                                {value === null && (
                                    <p className="font-ui text-[9px] text-slate-600 mt-2">Scoring unavailable</p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Full-width AI Recommendation Card */}
                {ocean?.role_recommendation && (
                    <div className="glass-card p-8 rounded-3xl border border-primary/20 bg-primary/5">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <Brain size={20} className="text-primary" />
                                <p className="font-ui text-sm text-primary uppercase tracking-widest font-black">AI Recommendation</p>
                            </div>
                            <button
                                onClick={handleRefreshRecommendation}
                                disabled={refreshingRec}
                                className="p-2 rounded-lg hover:bg-primary/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Regenerate AI recommendation"
                            >
                                <RotateCcw size={18} className={`text-primary ${refreshingRec ? 'animate-spin' : ''}`} />
                            </button>
                        </div>
                        <div className="prose prose-invert max-w-none">
                            <ReactMarkdown
                                components={{
                                    h1: ({ children }) => <h1 className="text-2xl font-black text-slate-100 mt-4 mb-3">{children}</h1>,
                                    h2: ({ children }) => <h2 className="text-xl font-bold text-slate-100 mt-4 mb-3">{children}</h2>,
                                    h3: ({ children }) => <h3 className="text-lg font-bold text-slate-200 mt-3 mb-2">{children}</h3>,
                                    p: ({ children }) => <p className="text-slate-300 text-base leading-relaxed mb-4">{children}</p>,
                                    ul: ({ children }) => <ul className="list-disc list-inside text-slate-300 text-base space-y-2 mb-4">{children}</ul>,
                                    ol: ({ children }) => <ol className="list-decimal list-inside text-slate-300 text-base space-y-2 mb-4">{children}</ol>,
                                    li: ({ children }) => <li className="text-slate-300 text-base">{children}</li>,
                                    strong: ({ children }) => <strong className="font-black text-slate-100">{children}</strong>,
                                    em: ({ children }) => <em className="italic text-slate-200">{children}</em>,
                                    code: ({ children }) => <code className="bg-slate-800 px-2 py-1 rounded text-slate-300 font-mono text-sm">{children}</code>,
                                    blockquote: ({ children }) => <blockquote className="border-l-4 border-primary/50 pl-4 italic text-slate-400 my-4">{children}</blockquote>,
                                }}
                            >
                                {ocean.role_recommendation}
                            </ReactMarkdown>
                        </div>
                    </div>
                )}
                </div>
            )}

            {/* ── Gaze & Signals Tab ────────────────────────────────────────────── */}
            {activeTab === "gaze" && (
                <div className="space-y-8">
                    {/* Summary row */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                        {[
                            { label: "Integrity Risk",   value: overallRisk.toUpperCase(), color: overallRisk === "high" ? "text-danger" : overallRisk === "medium" ? "text-warning" : "text-success" },
                            { label: "Avg HRV rMSSD",    value: vs.length ? `${(vs.reduce((a, v) => a + (v.avg_hrv_rmssd ?? 0), 0) / vs.length).toFixed(1)} ms` : "—", color: "text-primary" },
                            { label: "Avg Heart Rate",   value: vs.some(v => v.hr_bpm) ? `${Math.round(vs.reduce((a, v) => a + (v.hr_bpm ?? 0), 0) / vs.filter(v => v.hr_bpm).length)} bpm` : "—", color: "text-blue-400" },
                            { label: "Stress Spikes",    value: vs.filter(v => v.stress_spike_detected).length + " / " + vs.length, color: vs.some(v => v.stress_spike_detected) ? "text-warning" : "text-success" },
                        ].map(({ label, value, color }) => (
                            <div key={label} className="glass-card p-6 rounded-3xl border border-white/10">
                                <p className="font-ui text-[10px] text-slate-500 uppercase tracking-widest font-black mb-3">{label}</p>
                                <p className={`font-heading text-3xl font-black ${color}`}>{value}</p>
                            </div>
                        ))}
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Gaze zone distribution bar chart */}
                        <div className="glass-card p-8 rounded-3xl border border-white/10">
                            <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black mb-6">Avg Attention Zone Distribution</p>
                            {gazeBarData.length > 0 ? (
                                <ResponsiveContainer width="100%" height={200}>
                                    <RechartsBarChart data={gazeBarData} barCategoryGap="30%">
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                        <XAxis dataKey="zone" tick={{ fill: "#6b7280", fontSize: 12 }} />
                                        <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 11 }} unit="%" />
                                        <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: 12, color: "#1f2937" }}
                                            formatter={(v: number) => [`${v}%`, "Percentage"]} />
                                        <Bar dataKey="pct" radius={[6, 6, 0, 0]}>
                                            {gazeBarData.map((entry, i) => (
                                                <Cell key={i} fill={entry.zone === "red" ? "#EF4444" : entry.zone === "strategic" ? "#22C55E" : "#6C63FF"} />
                                            ))}
                                        </Bar>
                                    </RechartsBarChart>
                                </ResponsiveContainer>
                            ) : <div className="h-48 flex items-center justify-center text-slate-600 text-sm">No attention data available</div>}
                            <p className="text-slate-600 text-[10px] mt-4 font-ui">
                                Red = looking down (notes); Strategic = upper-center (thinking); Neutral = normal engagement
                            </p>
                        </div>

                        {/* HRV + HR area chart */}
                        <div className="glass-card p-8 rounded-3xl border border-white/10">
                            <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black mb-6">HRV rMSSD &amp; Heart Rate per Question</p>
                            <ResponsiveContainer width="100%" height={200}>
                                <AreaChart data={hrvData}>
                                    <defs>
                                        <linearGradient id="hrv" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="hr" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                    <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} />
                                    <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: 12, color: "#1f2937" }} />
                                    <Area type="monotone" dataKey="hrv" stroke="#22C55E" fill="url(#hrv)" strokeWidth={2} name="HRV ms" />
                                    <Area type="monotone" dataKey="hr" stroke="#3B82F6" fill="url(#hr)" strokeWidth={2} name="HR bpm" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Per-question attention analysis */}
                    <div className="glass-card p-8 rounded-3xl border border-white/10">
                        <div className="flex items-center gap-3 mb-6">
                            <Eye size={20} className="text-primary" />
                            <h3 className="font-heading text-xl font-bold">Eye Tracking — Per Question</h3>
                        </div>
                        {vs.length === 0 && (
                            <p className="text-slate-500 text-sm text-center py-8">No video signal data recorded.</p>
                        )}
                        <div className="space-y-4">
                            {vs.map((v, i) => {
                                const gm = v.gaze_metrics;
                                const gfCheatRisk = (gm?.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
                                const rtCheatRisk = (v.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
                                const highestRisk = (riskRank[gfCheatRisk] ?? 0) >= (riskRank[rtCheatRisk] ?? 0) ? gfCheatRisk : rtCheatRisk;
                                const isHighRisk = highestRisk === "high" || highestRisk === "medium";
                                const zones = gm?.zone_distribution ?? v.gaze_zone_distribution ?? {};
                                const robotic = gm?.robotic_reading;
                                const qMatch = qs.find(q => q.question_id === v.question_id);

                                return (
                                    <div key={i} className={`rounded-2xl border p-5 ${isHighRisk ? "border-danger/30 bg-danger/5" : "border-white/5 bg-slate-900/30"}`}>
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="flex items-center gap-3">
                                                <span className="font-heading text-lg font-bold text-primary">Q{i + 1}</span>
                                                {isHighRisk && <ShieldAlert size={16} className="text-danger" />}
                                                {robotic?.detected && <AlertCircle size={16} className="text-warning" title="Robotic reading pattern detected" />}
                                                <span className={`px-2 py-0.5 rounded-full text-[10px] font-ui font-black uppercase tracking-widest border
                                                    ${highestRisk === "high" ? "bg-danger/10 text-danger border-danger/20"
                                                    : highestRisk === "medium" ? "bg-warning/10 text-warning border-warning/20"
                                                    : "bg-success/10 text-success border-success/20"}`}>
                                                    {highestRisk} risk
                                                </span>
                                                {v.stress_spike_detected && (
                                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-ui font-black uppercase tracking-widest border bg-orange-500/10 text-orange-400 border-orange-500/20">
                                                        stress spike
                                                    </span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-4 text-xs text-slate-400 font-ui">
                                                {gm?.status === "not_installed" && <span className="text-slate-600 italic">Eye tracking unavailable</span>}
                                                {gm?.gaze_points_count != null && <span>{gm.gaze_points_count} tracking pts</span>}
                                                {v.hr_bpm != null && <span>{Math.round(v.hr_bpm)} bpm</span>}
                                                <span>HRV {v.avg_hrv_rmssd?.toFixed(1) ?? "—"} ms</span>
                                            </div>
                                        </div>

                                        {/* Attention zones */}
                                        {Object.keys(zones).length > 0 && (
                                            <div className="flex gap-3 mb-4">
                                                {Object.entries(zones).map(([zone, pct]) => (
                                                    <div key={zone} className="flex-1 bg-slate-800/50 rounded-xl p-3 text-center">
                                                        <p className="font-ui text-[9px] text-slate-500 uppercase tracking-wider mb-1">{zone}</p>
                                                        <p className={`font-heading text-xl font-black
                                                            ${zone === "red" ? "text-danger"
                                                            : zone === "strategic" ? "text-success"
                                                            : "text-slate-300"}`}>
                                                            {Math.round((pct as number) * 100)}%
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Robotic reading */}
                                        {robotic && (
                                            <div className={`flex items-center gap-3 rounded-xl p-3 text-sm
                                                ${robotic.detected ? "bg-warning/10 border border-warning/20 text-warning" : "bg-slate-800/30 text-slate-400"}`}>
                                                <Activity size={14} className="flex-shrink-0" />
                                                <span className="font-ui text-xs">
                                                    Robotic reading: <strong>{robotic.detected ? "DETECTED" : "none"}</strong>
                                                    {" · "}reversal rate {(robotic.reversal_rate * 100).toFixed(1)}%
                                                    {" · "}Y-stdev {robotic.y_stdev.toFixed(3)}
                                                </span>
                                            </div>
                                        )}

                                        {/* Head pose signals */}
                                        {(() => {
                                            const hp = (v.cheat_flags as Record<string,unknown>)?.head_pose as Record<string,unknown> | undefined;
                                            if (!hp?.head_pose_available) return null;
                                            const sustained = hp.sustained_head_turn as boolean;
                                            const offAxis   = hp.predominantly_off_axis as boolean;
                                            const erratic   = hp.erratic_head_movement as boolean;
                                            const leftPct   = hp.head_turn_left_pct  as number ?? 0;
                                            const rightPct  = hp.head_turn_right_pct as number ?? 0;
                                            const fwdPct    = hp.forward_facing_pct  as number ?? 1;
                                            const flagged   = sustained || offAxis || erratic;
                                            return (
                                                <div className={`mt-2 rounded-xl p-3 border text-xs font-ui
                                                    ${flagged ? "bg-danger/8 border-danger/20 text-danger" : "bg-slate-800/30 border-white/5 text-slate-400"}`}>
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0"><circle cx="12" cy="8" r="5"/><path d="M12 13v8"/><path d="M8 17h8"/></svg>
                                                        <span className="font-black uppercase tracking-widest text-[10px]">Head Pose</span>
                                                        {flagged && <span className="px-1.5 py-0.5 rounded bg-danger/15 text-danger text-[9px] font-black uppercase tracking-wider">flagged</span>}
                                                    </div>
                                                    <div className="grid grid-cols-3 gap-2 mb-2">
                                                        <div className="bg-slate-900/40 rounded-lg p-2 text-center">
                                                            <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Turn Left</p>
                                                            <p className={`font-heading text-base font-black ${leftPct > 0.15 ? "text-danger" : "text-slate-300"}`}>{Math.round(leftPct * 100)}%</p>
                                                        </div>
                                                        <div className="bg-slate-900/40 rounded-lg p-2 text-center">
                                                            <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Forward</p>
                                                            <p className={`font-heading text-base font-black ${fwdPct > 0.75 ? "text-success" : "text-warning"}`}>{Math.round(fwdPct * 100)}%</p>
                                                        </div>
                                                        <div className="bg-slate-900/40 rounded-lg p-2 text-center">
                                                            <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Turn Right</p>
                                                            <p className={`font-heading text-base font-black ${rightPct > 0.15 ? "text-danger" : "text-slate-300"}`}>{Math.round(rightPct * 100)}%</p>
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-wrap gap-1.5">
                                                        {(() => {
                                                            const absentPct   = hp.face_absent_pct  as number ?? 0;
                                                            const absentFlag  = hp.face_absent_flagged as boolean;
                                                            return (
                                                                <span className={`px-2 py-0.5 rounded-full border text-[9px] font-black uppercase tracking-wider
                                                                    ${absentFlag ? "bg-danger/10 text-danger border-danger/20" : "bg-slate-800 text-slate-500 border-white/5"}`}>
                                                                    face absent {Math.round(absentPct * 100)}% {absentFlag ? "✓" : "✗"}
                                                                </span>
                                                            );
                                                        })()}
                                                        <span className={`px-2 py-0.5 rounded-full border text-[9px] font-black uppercase tracking-wider
                                                            ${sustained ? "bg-danger/10 text-danger border-danger/20" : "bg-slate-800 text-slate-500 border-white/5"}`}>
                                                            sustained turn {sustained ? "✓" : "✗"}
                                                        </span>
                                                        <span className={`px-2 py-0.5 rounded-full border text-[9px] font-black uppercase tracking-wider
                                                            ${offAxis ? "bg-danger/10 text-danger border-danger/20" : "bg-slate-800 text-slate-500 border-white/5"}`}>
                                                            extreme off-axis {offAxis ? "✓" : "✗"}
                                                        </span>
                                                        <span className={`px-2 py-0.5 rounded-full border text-[9px] font-black uppercase tracking-wider
                                                            ${erratic ? "bg-warning/10 text-warning border-warning/20" : "bg-slate-800 text-slate-500 border-white/5"}`}>
                                                            erratic movement {erratic ? "✓" : "✗"}
                                                        </span>
                                                    </div>
                                                </div>
                                            );
                                        })()}

                                        {/* Emotion distribution for this question */}
                                        {Object.keys(v.emotion_distribution ?? {}).length > 0 && (
                                            <div className="mt-3 flex flex-wrap gap-2">
                                                {Object.entries(v.emotion_distribution).slice(0, 5).map(([emo, val]) => (
                                                    <span key={emo} className="px-2 py-1 bg-slate-800/50 rounded-lg text-[10px] font-ui text-slate-400">
                                                        {emo} {Math.round((val as number) * 100)}%
                                                    </span>
                                                ))}
                                            </div>
                                        )}

                                        {/* Video + audio players for this question */}
                                        {(qMatch?.video_url || qMatch?.audio_url) && (
                                            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4 border-t border-white/5 pt-4">
                                                {qMatch?.video_url && (
                                                    <div>
                                                        <p className="font-ui text-[9px] text-slate-500 uppercase tracking-widest mb-2">Video Recording</p>
                                                        <video
                                                            src={qMatch.video_url}
                                                            controls
                                                            className="w-full rounded-xl border border-border bg-black"
                                                            style={{ maxHeight: 180 }}
                                                        />
                                                    </div>
                                                )}
                                                {qMatch?.audio_url && (
                                                    <div>
                                                        <p className="font-ui text-[9px] text-slate-500 uppercase tracking-widest mb-2">Audio Recording</p>
                                                        <audio
                                                            src={qMatch.audio_url}
                                                            controls
                                                            className="w-full rounded-xl"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Per-Question Tab ──────────────────────────────────────────────── */}
            {activeTab === "questions" && (
                <div className="space-y-4">
                    {qs.length === 0 && (
                        <p className="text-slate-500 font-ui text-sm uppercase tracking-widest text-center py-16">No question responses recorded.</p>
                    )}
                    {qs.map((q, i) => {
                        const signal = vs.find(v => v.question_id === q.question_id);
                        const isExpanded = expandedQ === i;
                        // Use GazeFollower cheat flags if available, else real-time
                        const gfRisk = (signal?.gaze_metrics?.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
                        const rtRisk = (signal?.cheat_flags as Record<string,unknown>)?.risk_level as string ?? "low";
                        const cheatRisk = (riskRank[gfRisk] ?? 0) >= (riskRank[rtRisk] ?? 0) ? gfRisk : rtRisk;
                        const hasCheatFlag = cheatRisk === "high" || cheatRisk === "medium";

                        // Verdict badge config
                        const verdictCfg: Record<string, { label: string; color: string }> = {
                            correct:           { label: "Correct",           color: "bg-success/10 text-success border-success/30" },
                            partially_correct: { label: "Partial",           color: "bg-blue-500/10 text-blue-400 border-blue-500/30" },
                            can_be_better:     { label: "Can Be Better",     color: "bg-warning/10 text-warning border-warning/30" },
                            incorrect:         { label: "Incorrect",         color: "bg-danger/10 text-danger border-danger/30" },
                            not_attempted:     { label: "Not Attempted",     color: "bg-slate-700/40 text-slate-400 border-slate-600/30" },
                        };
                        const verdict = q.llm_verdict ?? null;
                        const vCfg = verdict ? (verdictCfg[verdict] ?? verdictCfg.incorrect) : null;

                        return (
                            <div key={q.id} className={`glass-card rounded-3xl border transition-all ${hasCheatFlag ? "border-danger/30" : "border-white/5"}`}>
                                <button className="w-full px-8 py-6 flex items-center justify-between text-left"
                                    onClick={() => setExpandedQ(isExpanded ? null : i)}>
                                    <div className="flex items-center gap-4">
                                        <span className="font-heading text-xl font-bold text-primary">Q{i + 1}</span>
                                        {hasCheatFlag && <ShieldAlert size={18} className="text-danger" />}
                                        {q.transcript_flagged && <AlertCircle size={18} className="text-warning" />}
                                        <p className="font-body text-slate-200 text-base line-clamp-1 max-w-2xl">{q.question_text}</p>
                                    </div>
                                    <div className="flex items-center gap-3 ml-4 flex-shrink-0">
                                        {vCfg && (
                                            <span className={`px-3 py-1 rounded-full font-ui font-black text-xs uppercase tracking-widest border ${vCfg.color}`}>
                                                {vCfg.label}
                                            </span>
                                        )}
                                        <span className={`px-3 py-1 rounded-full font-heading font-bold text-sm border
                                            ${q.combined_score >= 7 ? "bg-success/10 text-success border-success/20"
                                            : q.combined_score >= 4 ? "bg-warning/10 text-warning border-warning/20"
                                            : "bg-danger/10 text-danger border-danger/20"}`}>
                                            {q.combined_score.toFixed(1)}/10
                                        </span>
                                        <ChevronDown size={20} className={`text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                                    </div>
                                </button>

                                <AnimatePresence>
                                    {isExpanded && (
                                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                                            className="overflow-hidden border-t border-white/5">
                                            <div className="p-8 space-y-8">
                                                {/* Transcript */}
                                                <div>
                                                    <p className="font-ui text-[10px] text-slate-500 uppercase tracking-widest mb-3 font-black">Transcript</p>
                                                    {q.transcript_flagged && (
                                                        <div className="flex items-center gap-2 text-warning mb-3">
                                                            <AlertCircle size={14} />
                                                            <span className="font-ui text-xs">Whisper transcription unavailable — scoring based on empty response.</span>
                                                        </div>
                                                    )}
                                                    <p className="font-body text-slate-300 text-sm leading-relaxed bg-slate-900/50 p-5 rounded-2xl border border-white/5">
                                                        {q.transcript || <span className="text-slate-600 italic">No transcript</span>}
                                                    </p>
                                                </div>

                                                {/* LLM Verdict */}
                                                {verdict && vCfg && (
                                                    <div className={`rounded-2xl border p-5 ${vCfg.color.includes("success") ? "bg-success/5 border-success/20" : vCfg.color.includes("blue") ? "bg-blue-500/5 border-blue-500/20" : vCfg.color.includes("warning") ? "bg-warning/5 border-warning/20" : vCfg.color.includes("danger") ? "bg-danger/5 border-danger/20" : "bg-slate-800/40 border-slate-700/30"}`}>
                                                        <div className="flex items-center gap-3 mb-3">
                                                            <span className={`px-3 py-1 rounded-full font-ui font-black text-xs uppercase tracking-widest border ${vCfg.color}`}>
                                                                {vCfg.label}
                                                            </span>
                                                            <p className="font-ui text-xs text-slate-500 uppercase tracking-widest font-black">AI Verdict</p>
                                                        </div>
                                                        {q.llm_verdict_reason && (
                                                            <p className="font-body text-sm text-slate-300 leading-relaxed mb-4">{q.llm_verdict_reason}</p>
                                                        )}
                                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                            {(q.llm_strengths?.length ?? 0) > 0 && (
                                                                <div>
                                                                    <p className="font-ui text-[9px] text-success uppercase tracking-widest font-black mb-2">Strengths</p>
                                                                    <ul className="space-y-1">
                                                                        {q.llm_strengths!.map((s, si) => (
                                                                            <li key={si} className="flex items-start gap-2 text-xs text-slate-300 font-body">
                                                                                <CheckCircle2 size={12} className="text-success mt-0.5 flex-shrink-0" />
                                                                                {s}
                                                                            </li>
                                                                        ))}
                                                                    </ul>
                                                                </div>
                                                            )}
                                                            {(q.llm_key_gaps?.length ?? 0) > 0 && (
                                                                <div>
                                                                    <p className="font-ui text-[9px] text-danger uppercase tracking-widest font-black mb-2">Key Gaps</p>
                                                                    <ul className="space-y-1">
                                                                        {q.llm_key_gaps!.map((g, gi) => (
                                                                            <li key={gi} className="flex items-start gap-2 text-xs text-slate-300 font-body">
                                                                                <AlertCircle size={12} className="text-danger mt-0.5 flex-shrink-0" />
                                                                                {g}
                                                                            </li>
                                                                        ))}
                                                                    </ul>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Score bars */}
                                                <div>
                                                    <p className="font-ui text-[10px] text-slate-500 uppercase tracking-widest mb-4 font-black">Dimension Scores</p>
                                                    <div className="space-y-3">
                                                        {[
                                                            { label: "Semantic",      val: q.semantic_score * 10, color: "bg-primary" },
                                                            { label: "Technical",     val: q.technical_score,     color: "bg-blue-500" },
                                                            { label: "Communication", val: q.communication_score, color: "bg-violet-500" },
                                                            { label: "Behavioral",    val: q.behavioral_score,    color: "bg-success" },
                                                            { label: "Engagement",    val: q.engagement_score,    color: "bg-warning" },
                                                            { label: "Authenticity",  val: q.authenticity_score,  color: "bg-orange-500" },
                                                        ].map(({ label, val, color }) => (
                                                            <div key={label} className="flex items-center gap-4">
                                                                <span className="font-ui text-xs text-slate-400 w-28 uppercase tracking-wide">{label}</span>
                                                                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                                                                    <div className={`h-full ${color} rounded-full`} style={{ width: `${val != null ? val * 10 : 0}%` }} />
                                                                </div>
                                                                <span className="font-heading font-bold text-sm w-10 text-right">{val != null ? val.toFixed(1) : "—"}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Cheat flag banner */}
                                                {hasCheatFlag && (
                                                    <div className="flex items-center gap-3 bg-danger/10 border border-danger/30 rounded-2xl p-4 text-danger">
                                                        <ShieldAlert size={18} className="flex-shrink-0" />
                                                        <div>
                                                            <p className="font-ui text-xs font-black uppercase tracking-widest">
                                                                Risk Level: {cheatRisk.toUpperCase()}
                                                            </p>
                                                            <p className="font-body text-sm mt-1 opacity-80">
                                                                Unusual eye movement patterns detected during this response.
                                                            </p>
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Attention zone distribution (best available source) */}
                                                {signal && Object.keys(signal.gaze_metrics?.zone_distribution ?? signal.gaze_zone_distribution ?? {}).length > 0 && (
                                                    <div>
                                                        <p className="font-ui text-[10px] text-slate-500 uppercase tracking-widest mb-4 font-black">
                                                            Focus Areas
                                                        </p>
                                                        <div className="flex gap-4">
                                                            {Object.entries(signal.gaze_metrics?.zone_distribution ?? signal.gaze_zone_distribution).map(([zone, pct]) => (
                                                                <div key={zone} className="flex-1 bg-slate-900/50 rounded-2xl p-4 text-center border border-white/5">
                                                                    <p className="font-ui text-[10px] text-slate-500 uppercase tracking-wider mb-2">{zone}</p>
                                                                    <p className={`font-heading text-2xl font-black
                                                                        ${zone === "red" ? "text-danger"
                                                                        : zone === "strategic" ? "text-success"
                                                                        : "text-slate-300"}`}>
                                                                        {Math.round((pct as number) * 100)}%
                                                                    </p>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Media players — use Cloudinary URLs directly */}
                                                {(q.video_url || q.audio_url) && (
                                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                        {q.video_url && (
                                                            <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5">
                                                                <div className="flex items-center gap-2 mb-3">
                                                                    <Play size={14} className="text-primary" />
                                                                    <p className="font-ui text-xs text-slate-400 uppercase tracking-widest">Video Response</p>
                                                                </div>
                                                                <video src={q.video_url} controls className="w-full h-40 rounded-xl object-cover bg-black" />
                                                            </div>
                                                        )}
                                                        {q.audio_url && (
                                                            <div className="bg-slate-900/50 rounded-2xl p-4 border border-white/5">
                                                                <div className="flex items-center gap-2 mb-3">
                                                                    <Volume2 size={14} className="text-primary" />
                                                                    <p className="font-ui text-xs text-slate-400 uppercase tracking-widest">Audio Response</p>
                                                                </div>
                                                                <audio src={q.audio_url} controls className="w-full mt-6" />
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        );
                    })}
                </div>
            )}

        </div>
    );
}
