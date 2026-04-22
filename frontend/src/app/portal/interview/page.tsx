"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ShieldAlert, Wifi, WifiOff, ChevronRight, BrainCircuit,
    AlertTriangle, RotateCcw, MousePointerClick, Mic, MicOff, Loader2
} from "lucide-react";
import { useRouter } from "next/navigation";
import Script from "next/script";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const HEARTBEAT_MS = 30_000;

interface Question {
    id: string; stage: string; question: string;
    ideal_answer: string; time_window_seconds: number;
}
interface GazeSample { x: number; y: number }
interface HeadPoseSample { yaw: number; pitch: number }
declare global { interface Window { FaceMesh: unknown } }

function Toast({ msg }: { msg: string }) {
    return (
        <motion.div initial={{ y: -60, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -60, opacity: 0 }}
            className="fixed top-6 left-1/2 -translate-x-1/2 z-[200] flex items-center gap-3 bg-warning/90 text-white px-6 py-4 rounded-2xl shadow-2xl backdrop-blur-xl border border-warning/30">
            <AlertTriangle size={18} className="flex-shrink-0" />
            <span className="font-ui text-sm font-bold">{msg}</span>
        </motion.div>
    );
}

export default function InterviewPage() {
    const router = useRouter();

    const [questions,          setQuestions]          = useState<Question[]>([]);
    const [questionsLoaded,    setQuestionsLoaded]    = useState(false);
    const [sessionId,          setSessionId]          = useState("");
    const [currentQ,           setCurrentQ]           = useState(0);
    const [questionRevealed,   setQuestionRevealed]   = useState(false);
    const [timeLeft,           setTimeLeft]           = useState(0);
    const [isInterstitial,     setIsInterstitial]     = useState(false);
    const [interstitialCount,  setInterstitialCount]  = useState(5);
    const [showWarning,        setShowWarning]        = useState(false);
    const [warningCount,       setWarningCount]       = useState(0);
    const [waveHeights,        setWaveHeights]        = useState<number[]>(Array(80).fill(4));
    const [submitting,         setSubmitting]         = useState(false);
    const [toast,              setToast]              = useState<string | null>(null);
    const [isConnected,        setIsConnected]        = useState(true);
    const [isRecording,        setIsRecording]        = useState(false);

    const videoRecorderRef   = useRef<MediaRecorder | null>(null);
    const videoChunksRef     = useRef<Blob[]>([]);
    const audioRecorderRef   = useRef<MediaRecorder | null>(null);
    const audioChunksRef     = useRef<Blob[]>([]);
    const streamRef          = useRef<MediaStream | null>(null);
    const waveFrameRef     = useRef<number>(0);
    const gazeVideoRef     = useRef<HTMLVideoElement>(null);
    const gazeCanvasRef    = useRef<HTMLCanvasElement>(null);
    const faceMeshRef         = useRef<unknown>(null);
    const gazeSamplesRef      = useRef<GazeSample[]>([]);
    const headPoseSamplesRef  = useRef<HeadPoseSample[]>([]);
    const faceAbsentFramesRef = useRef<number>(0);   // frames where no face detected
    const totalPoseFramesRef  = useRef<number>(0);   // total FaceMesh frames attempted
    const gazeRafRef          = useRef<number>(0);
    const heartbeatRef     = useRef<ReturnType<typeof setInterval> | null>(null);

    const showToast = useCallback((msg: string) => {
        setToast(msg); setTimeout(() => setToast(null), 5000);
    }, []);

    // ── Load session ──────────────────────────────────────────────────────────
    useEffect(() => {
        try {
            const raw = sessionStorage.getItem("examiney_session");
            if (!raw) { router.replace("/portal/login"); return; }
            const sess = JSON.parse(raw);
            setSessionId(sess.session_id);
            setQuestions(sess.questions ?? []);
        } catch (e) {
            console.error("[Examiney][Interview] Failed to load session:", e);
        } finally {
            setQuestionsLoaded(true);
        }
    }, [router]);

    // ── Block screenshot keyboard shortcuts ───────────────────────────────────
    useEffect(() => {
        const block = (e: KeyboardEvent) => {
            // PrintScreen
            if (e.key === "PrintScreen" || e.code === "PrintScreen") { e.preventDefault(); return; }
            // Ctrl+P / Ctrl+Shift+P (print dialog = screenshot proxy)
            if ((e.ctrlKey || e.metaKey) && (e.key === "p" || e.key === "P")) { e.preventDefault(); return; }
            // Ctrl+Shift+S / Cmd+Shift+S
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === "s" || e.key === "S")) { e.preventDefault(); return; }
            // Windows Snipping Tool: Win+Shift+S (can't fully block, but attempt)
            if (e.shiftKey && e.metaKey && (e.key === "s" || e.key === "S")) { e.preventDefault(); return; }
            // Mac screenshot: Cmd+Shift+3/4/5/6
            if (e.metaKey && e.shiftKey && ["3","4","5","6"].includes(e.key)) { e.preventDefault(); return; }
        };
        const blockCtxMenu = (e: MouseEvent) => e.preventDefault();
        document.addEventListener("keydown", block, true);
        document.addEventListener("contextmenu", blockCtxMenu);
        return () => {
            document.removeEventListener("keydown", block, true);
            document.removeEventListener("contextmenu", blockCtxMenu);
        };
    }, []);

    // ── Heartbeat ─────────────────────────────────────────────────────────────
    useEffect(() => {
        if (!sessionId) return;
        heartbeatRef.current = setInterval(async () => {
            try {
                const r = await fetch(`${API}/session/${sessionId}/health`);
                setIsConnected((await r.json()).status === "healthy");
            } catch { setIsConnected(false); }
        }, HEARTBEAT_MS);
        return () => { if (heartbeatRef.current) clearInterval(heartbeatRef.current); };
    }, [sessionId]);

    // ── Timer (only when question revealed) ───────────────────────────────────
    useEffect(() => {
        if (!questionRevealed || isInterstitial || showWarning || submitting || timeLeft <= 0) return;
        const t = setInterval(() => setTimeLeft(v => {
            if (v <= 1) { clearInterval(t); return 0; }
            return v - 1;
        }), 1000);
        return () => clearInterval(t);
    }, [questionRevealed, isInterstitial, showWarning, submitting, timeLeft]);

    useEffect(() => {
        if (timeLeft === 0 && questionRevealed && !submitting && !isInterstitial) handleSubmit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [timeLeft]);

    // ── Interstitial countdown ────────────────────────────────────────────────
    useEffect(() => {
        if (!isInterstitial) return;
        if (interstitialCount <= 0) { setIsInterstitial(false); setInterstitialCount(5); return; }
        const t = setInterval(() => setInterstitialCount(v => v - 1), 1000);
        return () => clearInterval(t);
    }, [isInterstitial, interstitialCount]);

    // ── Gaze tracking init ────────────────────────────────────────────────────
    const initGazeTracking = useCallback(() => {
        if (!window.FaceMesh) return;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const fm = new (window.FaceMesh as any)({
            locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${f}`,
        });
        fm.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.3, minTrackingConfidence: 0.3 });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        fm.onResults((res: any) => {
            totalPoseFramesRef.current += 1;
            const lms = res?.multiFaceLandmarks?.[0];

            if (!lms || lms.length <= 473) {
                // No face detected in this frame — candidate off-screen or fully turned away
                faceAbsentFramesRef.current += 1;
                return;
            }

            // Iris gaze sample (landmarks 468 = left iris, 473 = right iris)
            const l = lms[468], r = lms[473];
            gazeSamplesRef.current.push({ x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 });

            // Head pose estimation from stable facial landmarks:
            //   1   = nose tip,  234 = left temple,  454 = right temple
            //   10  = forehead centre,  152 = chin centre
            if (lms.length > 454) {
                const noseTip   = lms[1];
                const leftEdge  = lms[234];
                const rightEdge = lms[454];
                const forehead  = lms[10];
                const chin      = lms[152];

                const faceW = rightEdge.x - leftEdge.x;
                const faceH = chin.y - forehead.y;
                const midX  = (leftEdge.x + rightEdge.x) / 2;
                const midY  = (forehead.y + chin.y) / 2;

                // Normalised yaw/pitch: 0 = straight, ±1 ≈ ±45° (matches solvePnP scale)
                const yaw   = faceW > 0.01 ? (noseTip.x - midX) / (faceW * 0.5) : 0;
                const pitch = faceH > 0.01 ? (noseTip.y - midY) / (faceH * 0.5) : 0;

                headPoseSamplesRef.current.push({
                    yaw:   Math.round(yaw   * 1000) / 1000,
                    pitch: Math.round(pitch * 1000) / 1000,
                });
            }
        });
        faceMeshRef.current = fm;
        const loop = () => {
            const v = gazeVideoRef.current, c = gazeCanvasRef.current;
            if (v && c && !v.paused && v.readyState >= 2) {
                const ctx = c.getContext("2d");
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                if (ctx) { ctx.drawImage(v, 0, 0, 320, 240); (fm as any).send({ image: c }).catch(() => {}); }
            }
            gazeRafRef.current = requestAnimationFrame(loop);
        };
        gazeRafRef.current = requestAnimationFrame(loop);
    }, []);

    // ── Start recording for one question ─────────────────────────────────────
    const startRecording = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            streamRef.current = stream;

            if (gazeVideoRef.current) {
                gazeVideoRef.current.srcObject = stream;
                gazeVideoRef.current.play().catch(() => {});
            }

            // Waveform
            const actx     = new AudioContext();
            const src      = actx.createMediaStreamSource(stream);
            const analyser = actx.createAnalyser();
            analyser.fftSize = 256;
            src.connect(analyser);
            const drawWave = () => {
                const data = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(data);
                setWaveHeights(Array.from({ length: 80 }, (_, i) => {
                    const idx = Math.floor(i * data.length / 80);
                    return Math.max(4, (data[idx] / 255) * 56 + 4);
                }));
                waveFrameRef.current = requestAnimationFrame(drawWave);
            };
            drawWave();

            videoChunksRef.current = [];
            videoRecorderRef.current = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8,opus" });
            videoRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) videoChunksRef.current.push(e.data); };
            videoRecorderRef.current.start(250);

            // Audio-only recorder for separate Whisper transcription
            audioChunksRef.current = [];
            const audioStream = new MediaStream(stream.getAudioTracks());
            audioRecorderRef.current = new MediaRecorder(audioStream, { mimeType: "audio/webm;codecs=opus" });
            audioRecorderRef.current.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
            audioRecorderRef.current.start(250);

            setIsRecording(true);
            console.log(`[Examiney][Interview] Recording started for Q${currentQ + 1}`);
        } catch (err) {
            console.error("[Examiney][Recording] failed:", err);
            showToast("Camera/microphone access required.");
        }
    }, [currentQ, showToast]);

    // ── Stop + collect blobs ──────────────────────────────────────────────────
    const stopRecording = useCallback((): Promise<{ video: Blob; audio: Blob }> =>
        new Promise(resolve => {
            cancelAnimationFrame(waveFrameRef.current);
            setWaveHeights(Array(80).fill(4));
            setIsRecording(false);

            // Stop audio recorder immediately (doesn't need onstop coordination)
            if (audioRecorderRef.current?.state !== "inactive") {
                audioRecorderRef.current!.stop();
            }

            if (videoRecorderRef.current?.state !== "inactive") {
                videoRecorderRef.current!.onstop = () => {
                    const video = new Blob(videoChunksRef.current, { type: "video/webm" });
                    const audio = new Blob(audioChunksRef.current, { type: "audio/webm" });
                    resolve({ video, audio });
                };
                videoRecorderRef.current!.stop();
            } else {
                resolve({ video: new Blob(), audio: new Blob() });
            }

            streamRef.current?.getTracks().forEach(t => t.stop());
        }), []);

    // ── Reveal question → start recording ────────────────────────────────────
    const handleReveal = useCallback(async () => {
        if (questionRevealed || submitting) return;
        const q = questions[currentQ];
        if (!q) return;
        gazeSamplesRef.current = [];
        await startRecording();
        setTimeLeft(q.time_window_seconds ?? 90);
        setQuestionRevealed(true);
    }, [questionRevealed, submitting, questions, currentQ, startRecording]);

    // ── Submit answer ─────────────────────────────────────────────────────────
    /** Retry a fetch up to `maxAttempts` times with exponential back-off. */
    const fetchWithRetry = useCallback(async (
        url: string, init: RequestInit, maxAttempts = 3,
    ): Promise<void> => {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const res = await fetch(url, init);
                if (res.ok || res.status < 500) return; // success or client error — don't retry
            } catch (_) { /* network error — retry */ }
            if (attempt < maxAttempts - 1) {
                await new Promise(r => setTimeout(r, 800 * 2 ** attempt)); // 0.8s, 1.6s
            }
        }
    }, []);

    const handleSubmit = useCallback(async () => {
        if (submitting || !questionRevealed) return;
        setSubmitting(true);
        try {
            const q                  = questions[currentQ];
            const { video, audio }   = await stopRecording();
            const gazeCopy     = [...gazeSamplesRef.current];
            gazeSamplesRef.current = [];
            const headPoseCopy = [...headPoseSamplesRef.current];
            headPoseSamplesRef.current = [];
            const faceAbsentFrames = faceAbsentFramesRef.current;
            const totalPoseFrames  = totalPoseFramesRef.current;
            faceAbsentFramesRef.current = 0;
            totalPoseFramesRef.current  = 0;

            const qNum  = currentQ + 1;
            const fname = `question${qNum}_video.webm`;

            const saveFd = new FormData();
            saveFd.append("session_id",      sessionId);
            saveFd.append("question_id",     q.id);
            saveFd.append("question_number", String(qNum));
            saveFd.append("question_text",   q.question);
            saveFd.append("ideal_answer",    q.ideal_answer ?? "");
            saveFd.append("question_stage",  q.stage ?? "intro");
            if (video.size > 0) saveFd.append("video_file", video, fname);
            if (audio.size > 0) saveFd.append("audio_file", audio, `question${qNum}_audio.webm`);

            const gazeFd = new FormData();
            gazeFd.append("session_id",   sessionId);
            gazeFd.append("question_id",  q.id);
            gazeFd.append("gaze_samples",        JSON.stringify(gazeCopy));
            gazeFd.append("head_pose_samples",   JSON.stringify(headPoseCopy));
            gazeFd.append("face_absent_frames",  String(faceAbsentFrames));
            gazeFd.append("total_pose_frames",   String(totalPoseFrames));
            if (video.size > 0) gazeFd.append("video_file", video, fname);

            const isLast = currentQ >= questions.length - 1;

            if (isLast) {
                // For the final answer, await the upload (with retry) so the video is
                // in Cloudinary before the backend starts post-session processing.
                await fetchWithRetry(
                    `${API}/session/${sessionId}/save-response`,
                    { method: "POST", body: saveFd },
                    3,
                );
                fetch(`${API}/video/analyze-chunk`, { method: "POST", body: gazeFd }).catch(() => {});
                router.push("/portal/thank-you");
                return;
            }

            // Non-last questions: fire-and-forget with a silent retry on failure
            fetchWithRetry(`${API}/session/${sessionId}/save-response`, { method: "POST", body: saveFd }, 2).catch(() => {});
            fetch(`${API}/video/analyze-chunk`,               { method: "POST", body: gazeFd }).catch(() => {});

            setCurrentQ(currentQ + 1);
            setQuestionRevealed(false);
            setTimeLeft(0);
            setIsInterstitial(true);
            setInterstitialCount(5);
        } catch (err) {
            console.error("[Examiney][Interview] submit error:", err);
            showToast("Failed to save answer — continuing interview.");
        } finally {
            setSubmitting(false);
        }
    }, [submitting, questionRevealed, questions, currentQ, stopRecording, sessionId, router, showToast, fetchWithRetry]);

    // ── Fullscreen ────────────────────────────────────────────────────────────
    useEffect(() => {
        const handler = () => {
            if (!document.fullscreenElement) { setWarningCount(v => v + 1); setShowWarning(true); }
            else setShowWarning(false);
        };
        document.addEventListener("fullscreenchange", handler);
        return () => document.removeEventListener("fullscreenchange", handler);
    }, []);
    const attemptFullscreen = () => {
        if (warningCount >= 2) {
            // Flag session as high-integrity-risk before redirecting
            if (sessionId) {
                fetch(`${API}/session/${sessionId}/flag-integrity`, { method: "POST" }).catch(() => {});
            }
            router.push("/portal/terminated");
            return;
        }
        document.documentElement.requestFullscreen().catch(() => {});
    };

    // ── Cleanup ───────────────────────────────────────────────────────────────
    useEffect(() => () => {
        cancelAnimationFrame(waveFrameRef.current);
        cancelAnimationFrame(gazeRafRef.current);
        if (heartbeatRef.current) clearInterval(heartbeatRef.current);
        streamRef.current?.getTracks().forEach(t => t.stop());
    }, []);

    const getTimerColor = () => {
        if (timeLeft <= 10) return "text-danger drop-shadow-[0_0_10px_rgba(239,68,68,0.5)]";
        if (timeLeft <= 30) return "text-warning";
        return "text-primary";
    };

    const totalQ          = questions.length || 18;
    const currentQuestion = questions[currentQ];
    const maxTime         = currentQuestion?.time_window_seconds ?? 90;

    return (
        <>
            <Script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/face_mesh.js"
                strategy="afterInteractive" onLoad={initGazeTracking} />
            <video  ref={gazeVideoRef}  className="hidden" width={320} height={240} muted playsInline />
            <canvas ref={gazeCanvasRef} className="hidden" width={320} height={240} />

            <div className="min-h-screen bg-[#f5f1eb] flex flex-col items-center justify-center p-8 relative select-none overflow-hidden font-body text-foreground">
                <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute top-[10%] left-[10%] w-[30%] h-[30%] bg-primary/5 blur-[150px] rounded-full" />
                    <div className="absolute bottom-[10%] right-[10%] w-[40%] h-[40%] bg-violet-200/10 blur-[130px] rounded-full" />
                </div>

                <AnimatePresence>{toast && <Toast msg={toast} />}</AnimatePresence>

                <AnimatePresence>
                    {!isConnected && (
                        <motion.div initial={{ y: -40 }} animate={{ y: 0 }} exit={{ y: -40 }}
                            className="fixed top-0 inset-x-0 z-50 bg-warning/90 text-white text-center py-3 font-ui text-sm font-black uppercase tracking-widest flex items-center justify-center gap-2">
                            <WifiOff size={16} /> Reconnecting to server…
                        </motion.div>
                    )}
                </AnimatePresence>

                <AnimatePresence mode="wait">
                    {/* Loading state */}
                    {!questionsLoaded && (
                        <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-4 text-foreground/40">
                            <Loader2 size={40} className="animate-spin" />
                            <p className="font-ui text-sm uppercase tracking-widest">Loading interview…</p>
                        </motion.div>
                    )}

                    {/* Empty questions fallback */}
                    {questionsLoaded && !currentQuestion && !isInterstitial && !showWarning && (
                        <motion.div key="no-questions" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="flex flex-col items-center gap-6 text-center max-w-md">
                            <AlertTriangle size={48} className="text-warning" />
                            <div>
                                <h2 className="font-heading text-2xl font-bold mb-2">No Questions Found</h2>
                                <p className="text-foreground/50 text-sm">Your interview questions could not be loaded. Please contact your recruiter or try logging in again.</p>
                            </div>
                            <button onClick={() => router.replace("/portal/login")}
                                className="bg-primary text-white font-heading font-bold px-8 py-4 rounded-2xl hover:bg-primary/90 transition-all">
                                Return to Login
                            </button>
                        </motion.div>
                    )}

                    {questionsLoaded && !isInterstitial && !showWarning && currentQuestion && (
                        <motion.div key={`q-${currentQ}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className="w-full max-w-5xl flex flex-col items-center relative z-10">

                            {/* Top bar */}
                            <div className="w-full flex items-center justify-between mb-16 px-10 py-6 border border-border bg-white/80 backdrop-blur-xl rounded-[2.5rem] shadow-sm">
                                <div className="flex items-center gap-6">
                                    <span className="font-ui text-xs font-black text-foreground/40 uppercase tracking-[0.3em]">
                                        {currentQuestion.stage?.toUpperCase()}
                                    </span>
                                    <div className="h-6 w-px bg-border" />
                                    <span className="font-heading text-2xl font-bold italic tracking-tight uppercase">
                                        Q{currentQ + 1} <span className="text-foreground/40 font-black tracking-widest text-sm opacity-60">OF {totalQ}</span>
                                    </span>
                                </div>

                                {/* Timer */}
                                {questionRevealed ? (
                                    <div className="relative">
                                        <svg className="w-24 h-24 -rotate-90">
                                            <circle cx="48" cy="48" r="40" fill="transparent" stroke="#e5e0d8" strokeWidth="4" />
                                            <motion.circle cx="48" cy="48" r="40" fill="transparent"
                                                stroke={timeLeft <= 10 ? "#EF4444" : timeLeft <= 30 ? "#F59E0B" : "#6C63FF"}
                                                strokeWidth="4" strokeDasharray="251"
                                                strokeDashoffset={251 * (1 - timeLeft / maxTime)}
                                                strokeLinecap="round" className="transition-all duration-1000" />
                                        </svg>
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <span className={`font-heading text-2xl font-black italic ${getTimerColor()}`}>{timeLeft}s</span>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="w-24 h-24 flex items-center justify-center rounded-full border-2 border-border">
                                        <span className="font-ui text-xs text-foreground/40 uppercase tracking-widest">Ready</span>
                                    </div>
                                )}

                                <div className="flex items-center gap-4">
                                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${isRecording ? "bg-danger/10 border border-danger/30" : "bg-gray-100 border border-border"}`}>
                                        {isRecording ? <Mic size={14} className="text-danger animate-pulse" /> : <MicOff size={14} className="text-foreground/40" />}
                                        <span className={`font-ui text-[10px] font-bold uppercase tracking-widest ${isRecording ? "text-danger" : "text-foreground/40"}`}>
                                            {isRecording ? "REC" : "IDLE"}
                                        </span>
                                    </div>
                                    <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-success animate-pulse" : "bg-danger"}`} />
                                    {isConnected ? <Wifi size={16} className="text-foreground/40" /> : <WifiOff size={16} className="text-danger" />}
                                </div>
                            </div>

                            {/* Question area */}
                            <div className="w-full max-w-4xl text-center mb-24 min-h-[260px] flex flex-col items-center justify-center">
                                {!questionRevealed ? (
                                    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                                        className="flex flex-col items-center gap-8">
                                        <div className="w-24 h-24 rounded-[2rem] bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                                            <BrainCircuit size={40} className="text-indigo-400" strokeWidth={1.5} />
                                        </div>
                                        <div className="text-center">
                                            <p className="font-ui text-foreground/40 text-sm uppercase tracking-widest mb-3">Question {currentQ + 1} of {totalQ}</p>
                                            <p className="font-heading text-2xl font-black italic text-foreground/20 mb-4 tracking-widest">
                                                ● ● ● ● ● ● ● ●
                                            </p>
                                            <p className="text-foreground/40 text-sm">Recording starts when you reveal the question.</p>
                                        </div>
                                        <button onClick={handleReveal} disabled={submitting}
                                            className="group bg-indigo-600 hover:bg-indigo-500 active:scale-95 disabled:opacity-50 text-white font-heading font-black text-sm uppercase tracking-[0.2em] px-12 py-5 rounded-2xl flex items-center gap-3 transition-all shadow-[0_0_40px_rgba(99,102,241,0.35)]">
                                            <MousePointerClick size={20} />
                                            Reveal Question &amp; Start Recording
                                        </button>
                                    </motion.div>
                                ) : (
                                    <motion.div key="revealed" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="w-full">
                                        <div className="bg-danger/10 border border-danger/20 inline-flex items-center gap-3 px-5 py-2 rounded-full mb-6 text-danger">
                                            <Mic size={16} className="animate-pulse" />
                                            <span className="font-ui text-xs font-black uppercase tracking-[0.25em]">Recording in Progress</span>
                                        </div>
                                        <div className="bg-white border border-border rounded-2xl p-8 shadow-sm">
                                            <p className="font-ui text-xs uppercase tracking-widest text-foreground/50 mb-3">Question {currentQ + 1} of {totalQ}</p>
                                            <h2 className="font-heading text-2xl font-semibold leading-relaxed text-foreground">
                                                {currentQuestion.question}
                                            </h2>
                                        </div>
                                    </motion.div>
                                )}
                            </div>

                            {/* Bottom controls */}
                            <div className="fixed bottom-12 w-full max-w-5xl flex flex-col items-center gap-8 px-10">
                                {questionRevealed && (
                                    <>
                                        <div className="flex items-center gap-0.5 h-14 w-full">
                                            {waveHeights.map((h, i) => (
                                                <div key={i} className="flex-1 bg-primary/40 rounded-full transition-all duration-75"
                                                    style={{ height: `${h}px` }} />
                                            ))}
                                        </div>
                                        <div className="flex items-center gap-8">
                                            <div className="flex items-center gap-3 bg-danger/10 border border-danger/20 px-6 py-3 rounded-2xl">
                                                <div className="w-2.5 h-2.5 bg-danger rounded-full animate-pulse" />
                                                <span className="font-ui text-sm font-black uppercase tracking-widest text-foreground/70">
                                                    {submitting ? "Saving…" : "Recording"}
                                                </span>
                                            </div>
                                            <button onClick={handleSubmit} disabled={submitting}
                                                className="group bg-foreground text-background font-heading font-black text-sm uppercase tracking-[0.2em] px-10 py-5 rounded-3xl flex items-center gap-3 transition-all hover:bg-foreground/90 active:scale-95 shadow-md disabled:opacity-50">
                                                {currentQ >= totalQ - 1 ? "Finish Interview" : "Next Question"}
                                                <ChevronRight size={20} className="group-hover:translate-x-1 transition-transform" />
                                            </button>
                                        </div>
                                    </>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {isInterstitial && (
                        <motion.div key="interstitial" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#1a1a1a]/97 backdrop-blur-3xl text-center px-10">
                            <div className="relative mb-12">
                                <div className="absolute inset-0 bg-primary/20 blur-3xl animate-pulse" />
                                <ShieldAlert className="w-28 h-28 text-primary relative z-10" strokeWidth={1} />
                            </div>
                            <h3 className="font-heading text-5xl font-black mb-4 tracking-tighter">Answer Saved</h3>
                            <p className="text-foreground/50 text-xl italic mb-6 opacity-80">Next question in…</p>
                            <motion.div key={interstitialCount} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                                className="font-heading text-[10rem] font-black italic tracking-tighter leading-none text-primary">
                                {interstitialCount}
                            </motion.div>
                            <p className="text-foreground/40 font-ui text-xs uppercase tracking-widest mt-8">
                                Click &quot;Reveal Question&quot; on the next screen to start recording
                            </p>
                        </motion.div>
                    )}

                    {showWarning && (
                        <motion.div key="warning" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                            className="fixed inset-0 z-[100] bg-danger/20 backdrop-blur-3xl flex items-center justify-center p-10">
                            <div className="max-w-xl w-full bg-white border-2 border-danger/50 rounded-[3rem] p-16 shadow-[0_0_60px_rgba(239,68,68,0.15)] text-center">
                                <div className="w-24 h-24 bg-danger/20 text-danger rounded-[2rem] flex items-center justify-center mx-auto mb-10">
                                    <ShieldAlert size={48} />
                                </div>
                                <h3 className="font-heading text-4xl font-bold mb-6 uppercase italic">Security Violation</h3>
                                <p className="text-foreground/60 text-lg mb-12 italic">
                                    {warningCount === 1
                                        ? "Full-screen mode was exited. This is your final warning."
                                        : "Multiple violations detected. Session locked for review."}
                                </p>
                                <button onClick={attemptFullscreen}
                                    className="w-full bg-danger text-white font-heading font-black py-6 rounded-2xl flex items-center justify-center gap-4 hover:bg-danger/90 active:scale-95 transition-all">
                                    <RotateCcw size={24} /> Restore Full-Screen
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </>
    );
}
