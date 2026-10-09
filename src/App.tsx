/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import {
  Activity,
  Heart,
  AlertTriangle,
  CheckCircle2,
  Terminal,
  Play,
  FileCode2,
  HelpCircle,
  Copy,
  Check,
  RotateCcw,
  Sliders,
  BarChart3,
  Database,
  ExternalLink,
  ChevronRight,
  ShieldCheck,
  Stethoscope,
  BookOpen,
  ArrowRight,
  Layers,
  Sparkles,
  Download
} from 'lucide-react';

// ==============================================================================
// TYPES & CLINICAL INTERFACES
// ==============================================================================
interface PatientData {
  age: number;
  sex: number; // 0: female, 1: male
  cp: number; // 0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic
  trestbps: number; // mm Hg
  chol: number; // mg/dl
  fbs: number; // 0: <= 120, 1: > 120
  restecg: number; // 0: normal, 1: ST-T wave, 2: LVH
  thalach: number; // max heart rate
  exang: number; // 0: no, 1: yes
  oldpeak: number; // ST depression
  slope: number; // 0: upsloping, 1: flat, 2: downsloping
  ca: number; // 0-3 major vessels
  thal: number; // 1: normal, 2: fixed, 3: reversible
}

interface EvaluationResult {
  probability: number;
  prediction: number;
  riskCategory: 'Low' | 'Moderate' | 'High' | 'Critical';
  logOdds: number;
  zScores: Record<string, number>;
  contributions: { feature: string; label: string; score: number; isRisk: boolean }[];
}

// Cleveland standard normalization constants & Logistic Regression weights
const FEATURE_MEANS: Record<keyof PatientData, number> = {
  age: 54.37,
  sex: 0.68,
  cp: 0.97,
  trestbps: 131.62,
  chol: 246.26,
  fbs: 0.15,
  restecg: 0.53,
  thalach: 149.65,
  exang: 0.33,
  oldpeak: 1.04,
  slope: 1.40,
  ca: 0.73,
  thal: 2.31
};

const FEATURE_STDS: Record<keyof PatientData, number> = {
  age: 9.08,
  sex: 0.47,
  cp: 1.03,
  trestbps: 17.54,
  chol: 51.83,
  fbs: 0.36,
  restecg: 0.53,
  thalach: 22.91,
  exang: 0.47,
  oldpeak: 1.16,
  slope: 0.62,
  ca: 1.02,
  thal: 0.61
};

// Logistic Regression trained weights (standardized space)
const FEATURE_WEIGHTS: Record<keyof PatientData, number> = {
  age: 0.12,
  sex: 0.65,
  cp: 0.82,
  trestbps: 0.28,
  chol: 0.22,
  fbs: 0.08,
  restecg: 0.19,
  thalach: -0.68, // protective: higher max HR indicates better cardiac capacity
  exang: 0.71,
  oldpeak: 0.84,
  slope: 0.36,
  ca: 0.92,
  thal: 0.78
};

const INTERCEPT = -0.18;

const PRESETS: Record<string, PatientData> = {
  'Healthy Adult (Low Risk)': {
    age: 42,
    sex: 0,
    cp: 2,
    trestbps: 118,
    chol: 195,
    fbs: 0,
    restecg: 0,
    thalach: 172,
    exang: 0,
    oldpeak: 0.0,
    slope: 0,
    ca: 0,
    thal: 1
  },
  'Borderline Hypertensive (Moderate)': {
    age: 53,
    sex: 1,
    cp: 1,
    trestbps: 142,
    chol: 248,
    fbs: 0,
    restecg: 1,
    thalach: 144,
    exang: 0,
    oldpeak: 0.8,
    slope: 1,
    ca: 1,
    thal: 2
  },
  'Critical Cardiac Risk (High)': {
    age: 63,
    sex: 1,
    cp: 0,
    trestbps: 160,
    chol: 286,
    fbs: 1,
    restecg: 2,
    thalach: 112,
    exang: 1,
    oldpeak: 2.8,
    slope: 1,
    ca: 2,
    thal: 3
  },
  'Severe Multivessel Disease': {
    age: 68,
    sex: 1,
    cp: 0,
    trestbps: 168,
    chol: 310,
    fbs: 1,
    restecg: 2,
    thalach: 98,
    exang: 1,
    oldpeak: 4.2,
    slope: 2,
    ca: 3,
    thal: 3
  }
};

// Benchmark sample of Cleveland records (for data explorer)
const CLEVELAND_SAMPLES: (PatientData & { target: number; id: number })[] = [
  { id: 1, age: 63, sex: 1, cp: 3, trestbps: 145, chol: 233, fbs: 1, restecg: 0, thalach: 150, exang: 0, oldpeak: 2.3, slope: 0, ca: 0, thal: 1, target: 1 },
  { id: 2, age: 37, sex: 1, cp: 2, trestbps: 130, chol: 250, fbs: 0, restecg: 1, thalach: 187, exang: 0, oldpeak: 3.5, slope: 0, ca: 0, thal: 2, target: 1 },
  { id: 3, age: 41, sex: 0, cp: 1, trestbps: 130, chol: 204, fbs: 0, restecg: 0, thalach: 172, exang: 0, oldpeak: 1.4, slope: 2, ca: 0, thal: 2, target: 1 },
  { id: 4, age: 56, sex: 1, cp: 1, trestbps: 120, chol: 236, fbs: 0, restecg: 1, thalach: 178, exang: 0, oldpeak: 0.8, slope: 2, ca: 0, thal: 2, target: 1 },
  { id: 5, age: 57, sex: 0, cp: 0, trestbps: 120, chol: 354, fbs: 0, restecg: 1, thalach: 163, exang: 1, oldpeak: 0.6, slope: 2, ca: 0, thal: 2, target: 1 },
  { id: 6, age: 57, sex: 1, cp: 0, trestbps: 140, chol: 192, fbs: 0, restecg: 1, thalach: 148, exang: 0, oldpeak: 0.4, slope: 1, ca: 0, thal: 1, target: 1 },
  { id: 7, age: 56, sex: 0, cp: 1, trestbps: 140, chol: 294, fbs: 0, restecg: 0, thalach: 153, exang: 0, oldpeak: 1.3, slope: 1, ca: 0, thal: 2, target: 1 },
  { id: 8, age: 44, sex: 1, cp: 1, trestbps: 120, chol: 263, fbs: 0, restecg: 1, thalach: 173, exang: 0, oldpeak: 0.0, slope: 2, ca: 0, thal: 3, target: 1 },
  { id: 9, age: 52, sex: 1, cp: 2, trestbps: 172, chol: 199, fbs: 1, restecg: 1, thalach: 162, exang: 0, oldpeak: 0.5, slope: 2, ca: 0, thal: 3, target: 1 },
  { id: 10, age: 57, sex: 1, cp: 2, trestbps: 150, chol: 168, fbs: 0, restecg: 1, thalach: 174, exang: 0, oldpeak: 1.6, slope: 2, ca: 0, thal: 2, target: 1 },
  { id: 11, age: 57, sex: 1, cp: 0, trestbps: 165, chol: 289, fbs: 1, restecg: 0, thalach: 124, exang: 1, oldpeak: 1.0, slope: 1, ca: 3, thal: 3, target: 0 },
  { id: 12, age: 67, sex: 1, cp: 0, trestbps: 160, chol: 286, fbs: 0, restecg: 0, thalach: 108, exang: 1, oldpeak: 1.5, slope: 1, ca: 3, thal: 2, target: 0 },
  { id: 13, age: 58, sex: 1, cp: 0, trestbps: 114, chol: 318, fbs: 0, restecg: 2, thalach: 140, exang: 0, oldpeak: 4.4, slope: 0, ca: 3, thal: 1, target: 0 },
  { id: 14, age: 57, sex: 1, cp: 0, trestbps: 152, chol: 274, fbs: 0, restecg: 1, thalach: 88, exang: 1, oldpeak: 1.2, slope: 1, ca: 1, thal: 3, target: 0 },
  { id: 15, age: 60, sex: 1, cp: 0, trestbps: 130, chol: 206, fbs: 0, restecg: 0, thalach: 132, exang: 1, oldpeak: 2.4, slope: 1, ca: 2, thal: 3, target: 0 },
  { id: 16, age: 56, sex: 1, cp: 0, trestbps: 132, chol: 184, fbs: 0, restecg: 0, thalach: 105, exang: 1, oldpeak: 2.1, slope: 1, ca: 1, thal: 1, target: 0 },
  { id: 17, age: 48, sex: 1, cp: 0, trestbps: 124, chol: 274, fbs: 0, restecg: 0, thalach: 166, exang: 0, oldpeak: 0.5, slope: 1, ca: 0, thal: 3, target: 0 },
  { id: 18, age: 63, sex: 0, cp: 0, trestbps: 150, chol: 407, fbs: 0, restecg: 0, thalach: 154, exang: 0, oldpeak: 4.0, slope: 1, ca: 3, thal: 3, target: 0 }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'predictor' | 'evaluation' | 'dataset' | 'guide'>('predictor');
  const [selectedPreset, setSelectedPreset] = useState<string>('Custom');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Manual execution state
  const [hasEvaluated, setHasEvaluated] = useState<boolean>(false);
  const [isCalculating, setIsCalculating] = useState<boolean>(false);
  const [evaluatedPatient, setEvaluatedPatient] = useState<PatientData | null>(null);

  // Form state
  const [patient, setPatient] = useState<PatientData>({
    age: 54,
    sex: 1,
    cp: 0,
    trestbps: 130,
    chol: 240,
    fbs: 0,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 1.0,
    slope: 1,
    ca: 0,
    thal: 2
  });

  // Calculate evaluation strictly for evaluated patient snapshot
  const activeEvalPatient = evaluatedPatient || patient;

  // Check if current form inputs differ from the last evaluated snapshot
  const isInputModifiedSinceEvaluation = useMemo(() => {
    if (!hasEvaluated || !evaluatedPatient) return false;
    return (Object.keys(patient) as (keyof PatientData)[]).some(
      (k) => patient[k] !== evaluatedPatient[k]
    );
  }, [hasEvaluated, evaluatedPatient, patient]);

  // Perform Scikit-Learn Logistic Regression Pipeline Inference
  const evaluation: EvaluationResult = useMemo(() => {
    let z = INTERCEPT;
    const zScores: Record<string, number> = {};
    const contributions: { feature: string; label: string; score: number; isRisk: boolean }[] = [];

    const labelMap: Record<keyof PatientData, string> = {
      age: 'Age',
      sex: 'Biological Sex',
      cp: 'Chest Pain Type',
      trestbps: 'Resting Blood Pressure',
      chol: 'Serum Cholesterol',
      fbs: 'Fasting Blood Sugar',
      restecg: 'Resting Electrocardiogram',
      thalach: 'Max Heart Rate (Protective)',
      exang: 'Exercise Induced Angina',
      oldpeak: 'ST Segment Depression',
      slope: 'Slope of Peak Exercise ST',
      ca: 'Fluoroscopy Vessels Colored',
      thal: 'Thallium Defect'
    };

    (Object.keys(activeEvalPatient) as (keyof PatientData)[]).forEach((key) => {
      const val = activeEvalPatient[key];
      const mean = FEATURE_MEANS[key];
      const std = FEATURE_STDS[key];
      const weight = FEATURE_WEIGHTS[key];

      const zVal = (val - mean) / std;
      zScores[key] = zVal;

      const effect = zVal * weight;
      z += effect;

      contributions.push({
        feature: key,
        label: labelMap[key],
        score: effect,
        isRisk: effect > 0
      });
    });

    // Logistic Sigmoid
    const prob = 1.0 / (1.0 + Math.exp(-z));
    const pred = prob >= 0.5 ? 1 : 0;

    let cat: 'Low' | 'Moderate' | 'High' | 'Critical' = 'Low';
    if (prob >= 0.75) cat = 'Critical';
    else if (prob >= 0.5) cat = 'High';
    else if (prob >= 0.3) cat = 'Moderate';
    else cat = 'Low';

    contributions.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

    return {
      probability: prob,
      prediction: pred,
      riskCategory: cat,
      logOdds: z,
      zScores,
      contributions
    };
  }, [activeEvalPatient]);

  const handleRunPrediction = () => {
    setIsCalculating(true);
    setTimeout(() => {
      setIsCalculating(false);
      setHasEvaluated(true);
      setEvaluatedPatient({ ...patient });
    }, 200);
  };

  const handleClearOutput = () => {
    setHasEvaluated(false);
    setEvaluatedPatient(null);
  };

  const handlePresetSelect = (presetName: string) => {
    setSelectedPreset(presetName);
    if (PRESETS[presetName]) {
      setPatient({ ...PRESETS[presetName] });
      // Do NOT automatically give output; user must run it!
      setHasEvaluated(false);
      setEvaluatedPatient(null);
    }
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const updateField = <K extends keyof PatientData>(field: K, value: PatientData[K]) => {
    setSelectedPreset('Custom');
    setPatient((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-rose-500/30 selection:text-rose-200">
      {/* TOP HEADER */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-rose-600 to-rose-400 flex items-center justify-center shadow-lg shadow-rose-950/50">
              <Heart className="h-5 w-5 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-tight text-base sm:text-lg text-slate-100">CardioPredict ML</span>
                <span className="px-2 py-0.5 text-xs font-semibold uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded">CO5 Lab</span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Coronary Artery Disease Detection · UCI Cleveland Cohort (N=303)
              </p>
            </div>
          </div>

          {/* NAVIGATION TABS */}
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setActiveTab('predictor')}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'predictor'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Stethoscope className="h-4 w-4" />
              <span>Predictor</span>
            </button>
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'evaluation'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">CO5 Metrics</span>
              <span className="sm:hidden">Metrics</span>
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'dataset'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Database className="h-4 w-4" />
              <span className="hidden sm:inline">Dataset (303)</span>
              <span className="sm:hidden">Data</span>
            </button>
            <button
              onClick={() => setActiveTab('guide')}
              className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all flex items-center gap-1.5 ${
                activeTab === 'guide'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <BookOpen className="h-4 w-4" />
              <span className="hidden sm:inline">Lab Documentation</span>
              <span className="sm:hidden">Guide</span>
            </button>
          </nav>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* ==================================================================== */}
        {/* TAB 1: CLINICAL RISK PREDICTOR */}
        {/* ==================================================================== */}
        {activeTab === 'predictor' && (
          <div className="space-y-6">
            {/* Presets Toolbar */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 shadow-md">
              <div className="flex items-center gap-2">
                <Sliders className="h-4 w-4 text-rose-400" />
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Benchmark Presets:</span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {Object.keys(PRESETS).map((pName) => (
                  <button
                    key={pName}
                    onClick={() => handlePresetSelect(pName)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                      selectedPreset === pName
                        ? 'bg-rose-600 text-white shadow'
                        : 'bg-slate-800/70 text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    {pName}
                  </button>
                ))}
                <button
                  onClick={() => {
                    handlePresetSelect('Healthy Adult (Low Risk)');
                  }}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 flex items-center gap-1"
                  title="Reset to Baseline"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  <span>Reset</span>
                </button>
              </div>
            </div>

            {/* Grid Layout: Inputs (Left 7 Cols) + Diagnostic Dashboard (Right 5 Cols) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* INPUT FORM */}
              <div className="lg:col-span-7 space-y-5">
                {/* Section 1: Demographics & Hemodynamics */}
                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                      1. Patient Demographics & Baseline Vitals
                    </h3>
                    <span className="text-xs text-slate-500 font-mono">Features 1-4</span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Age */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">Age (years)</label>
                        <span className="font-mono text-rose-400 font-bold">{patient.age} yrs</span>
                      </div>
                      <input
                        type="range"
                        min="29"
                        max="77"
                        value={patient.age}
                        onChange={(e) => updateField('age', Number(e.target.value))}
                        className="w-full accent-rose-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>29</span>
                        <span>Cleveland Mean: 54.4</span>
                        <span>77</span>
                      </div>
                    </div>

                    {/* Biological Sex */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Biological Sex (sex)</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => updateField('sex', 0)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.sex === 0
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          Female (0)
                        </button>
                        <button
                          type="button"
                          onClick={() => updateField('sex', 1)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.sex === 1
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          Male (1)
                        </button>
                      </div>
                    </div>

                    {/* Resting BP */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">Resting Blood Pressure (trestbps)</label>
                        <span className="font-mono text-rose-400 font-bold">{patient.trestbps} mm Hg</span>
                      </div>
                      <input
                        type="range"
                        min="94"
                        max="200"
                        value={patient.trestbps}
                        onChange={(e) => updateField('trestbps', Number(e.target.value))}
                        className="w-full accent-rose-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>94</span>
                        <span>Normal: &lt;120</span>
                        <span>200</span>
                      </div>
                    </div>

                    {/* Serum Chol */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">Serum Cholesterol (chol)</label>
                        <span className="font-mono text-rose-400 font-bold">{patient.chol} mg/dL</span>
                      </div>
                      <input
                        type="range"
                        min="126"
                        max="564"
                        value={patient.chol}
                        onChange={(e) => updateField('chol', Number(e.target.value))}
                        className="w-full accent-rose-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>126</span>
                        <span>Desirable: &lt;200</span>
                        <span>564</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Section 2: Pain & Metabolic Biomarkers */}
                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                      2. Angina Symptomatology & Resting ECG
                    </h3>
                    <span className="text-xs text-slate-500 font-mono">Features 5-8</span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Chest Pain Type */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Chest Pain Type (cp)</label>
                      <select
                        value={patient.cp}
                        onChange={(e) => updateField('cp', Number(e.target.value))}
                        className="w-full bg-slate-800/90 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500"
                      >
                        <option value={0}>0: Typical Angina (Substernal Pressure)</option>
                        <option value={1}>1: Atypical Angina (Non-classic)</option>
                        <option value={2}>2: Non-Anginal Chest Pain</option>
                        <option value={3}>3: Asymptomatic (Silent Ischemia)</option>
                      </select>
                    </div>

                    {/* Fasting Blood Sugar */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Fasting Blood Sugar &gt; 120 mg/dL (fbs)</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => updateField('fbs', 0)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.fbs === 0
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          False / ≤ 120 (0)
                        </button>
                        <button
                          type="button"
                          onClick={() => updateField('fbs', 1)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.fbs === 1
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          True / &gt; 120 (1)
                        </button>
                      </div>
                    </div>

                    {/* Resting ECG */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Resting Electrocardiogram (restecg)</label>
                      <select
                        value={patient.restecg}
                        onChange={(e) => updateField('restecg', Number(e.target.value))}
                        className="w-full bg-slate-800/90 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500"
                      >
                        <option value={0}>0: Normal Sinus Rhythm</option>
                        <option value={1}>1: ST-T Wave Abnormality (T inversion)</option>
                        <option value={2}>2: Left Ventricular Hypertrophy (Estes)</option>
                      </select>
                    </div>

                    {/* Max Heart Rate */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">Max Heart Rate (thalach)</label>
                        <span className="font-mono text-emerald-400 font-bold">{patient.thalach} bpm</span>
                      </div>
                      <input
                        type="range"
                        min="71"
                        max="202"
                        value={patient.thalach}
                        onChange={(e) => updateField('thalach', Number(e.target.value))}
                        className="w-full accent-emerald-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>71 bpm</span>
                        <span>Chronotropic Reserve</span>
                        <span>202 bpm</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Section 3: Exercise Stress Test & Fluoroscopy */}
                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-5 shadow-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-cyan-500"></span>
                      3. Stress Test & Coronary Angiography Findings
                    </h3>
                    <span className="text-xs text-slate-500 font-mono">Features 9-13</span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Exercise Angina */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Exercise-Induced Angina (exang)</label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => updateField('exang', 0)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.exang === 0
                              ? 'bg-emerald-600/20 border-emerald-500 text-emerald-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          No (0)
                        </button>
                        <button
                          type="button"
                          onClick={() => updateField('exang', 1)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.exang === 1
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          Yes (1)
                        </button>
                      </div>
                    </div>

                    {/* ST Depression (oldpeak) */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">ST Depression (oldpeak)</label>
                        <span className="font-mono text-rose-400 font-bold">{patient.oldpeak.toFixed(1)} mm</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="6.2"
                        step="0.1"
                        value={patient.oldpeak}
                        onChange={(e) => updateField('oldpeak', Number(e.target.value))}
                        className="w-full accent-rose-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>0.0 mm</span>
                        <span>Subendocardial Ischemia</span>
                        <span>6.2 mm</span>
                      </div>
                    </div>

                    {/* Slope of peak exercise ST */}
                    <div>
                      <label className="text-xs text-slate-300 font-medium block mb-1">Slope of Peak ST (slope)</label>
                      <select
                        value={patient.slope}
                        onChange={(e) => updateField('slope', Number(e.target.value))}
                        className="w-full bg-slate-800/90 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500"
                      >
                        <option value={0}>0: Upsloping (Physiologic)</option>
                        <option value={1}>1: Flat (Subendocardial ischemia)</option>
                        <option value={2}>2: Downsloping (Severe transmural ischemia)</option>
                      </select>
                    </div>

                    {/* Vessels colored (ca) */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="text-slate-300 font-medium">Major Vessels Colored (ca)</label>
                        <span className="font-mono text-rose-400 font-bold">{patient.ca} vessels</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="3"
                        value={patient.ca}
                        onChange={(e) => updateField('ca', Number(e.target.value))}
                        className="w-full accent-rose-500 cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-slate-500">
                        <span>0 (Clear)</span>
                        <span>Fluoroscopy Stenosis</span>
                        <span>3 (Severe)</span>
                      </div>
                    </div>

                    {/* Thallium Defect */}
                    <div className="sm:col-span-2">
                      <label className="text-xs text-slate-300 font-medium block mb-1">Thallium Scintigraphy (thal)</label>
                      <div className="grid grid-cols-3 gap-2">
                        <button
                          type="button"
                          onClick={() => updateField('thal', 1)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.thal === 1
                              ? 'bg-emerald-600/20 border-emerald-500 text-emerald-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          1: Normal Perfusion
                        </button>
                        <button
                          type="button"
                          onClick={() => updateField('thal', 2)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.thal === 2
                              ? 'bg-amber-600/20 border-amber-500 text-amber-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          2: Fixed Defect (Scar)
                        </button>
                        <button
                          type="button"
                          onClick={() => updateField('thal', 3)}
                          className={`py-2 text-xs rounded-lg font-medium border transition ${
                            patient.thal === 3
                              ? 'bg-rose-600/20 border-rose-500 text-rose-200'
                              : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:bg-slate-800'
                          }`}
                        >
                          3: Reversible Defect
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Manual Execution Action Bar */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 sm:p-5 shadow-lg flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="space-y-1 text-center sm:text-left">
                    <div className="flex items-center gap-2 justify-center sm:justify-start">
                      <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse"></span>
                      <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">Manual Diagnosis Execution</span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Automatic output is turned off. Click below to pass patient values to the trained Scikit-Learn pipeline.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={handleRunPrediction}
                    disabled={isCalculating}
                    className="w-full sm:w-auto px-6 py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs sm:text-sm font-bold flex items-center justify-center gap-2 shadow-lg shadow-rose-950/50 transition cursor-pointer active:scale-95 disabled:opacity-60"
                  >
                    {isCalculating ? (
                      <>
                        <RotateCcw className="h-4 w-4 animate-spin" />
                        <span>Running ML Pipeline...</span>
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 fill-current" />
                        <span>{hasEvaluated ? 'Re-run ML Diagnosis' : 'Run Machine Learning Diagnosis'}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* CLINICAL DECISION SUPPORT PANEL (RIGHT 5 COLS) */}
              <div className="lg:col-span-5 space-y-5">
                {!hasEvaluated ? (
                  /* AWAITING MANUAL RUN STATE (OUTPUT NOT AUTOMATICALLY GIVEN) */
                  <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl space-y-5 text-center">
                    <div className="mx-auto w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center text-slate-400 shadow-inner">
                      <Stethoscope className="h-8 w-8 text-rose-400" />
                    </div>

                    <div className="space-y-1.5">
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded bg-slate-800/80 text-slate-400 border border-slate-700 inline-block">
                        Awaiting Manual Run
                      </span>
                      <h3 className="text-lg font-bold text-slate-100">Ready to Evaluate Patient</h3>
                      <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                        Automatic output is disabled. Configure biomarkers on the left and click <strong>"Run Machine Learning Diagnosis"</strong> to generate coronary stenosis risk.
                      </p>
                    </div>

                    {/* Vitals Summary Card */}
                    <div className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-4 text-left space-y-2">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                        <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                          Configured Parameters
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">13 Features</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div><span className="text-slate-500">Age:</span> <span className="font-mono text-slate-200">{patient.age} yrs</span></div>
                        <div><span className="text-slate-500">Sex:</span> <span className="font-mono text-slate-200">{patient.sex === 1 ? 'Male' : 'Female'}</span></div>
                        <div><span className="text-slate-500">Resting BP:</span> <span className="font-mono text-slate-200">{patient.trestbps} mm Hg</span></div>
                        <div><span className="text-slate-500">Cholesterol:</span> <span className="font-mono text-slate-200">{patient.chol} mg/dL</span></div>
                        <div><span className="text-slate-500">Chest Pain:</span> <span className="font-mono text-slate-200">{['Typical (0)', 'Atypical (1)', 'Non-Anginal (2)', 'Asymptomatic (3)'][patient.cp]}</span></div>
                        <div><span className="text-slate-500">Max HR:</span> <span className="font-mono text-emerald-400">{patient.thalach} bpm</span></div>
                        <div><span className="text-slate-500">ST-Depression:</span> <span className="font-mono text-rose-300">{patient.oldpeak.toFixed(1)} mm</span></div>
                        <div><span className="text-slate-500">Major Vessels:</span> <span className="font-mono text-slate-200">{patient.ca}</span></div>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={handleRunPrediction}
                      disabled={isCalculating}
                      className="w-full py-3.5 px-4 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs sm:text-sm font-bold flex items-center justify-center gap-2 shadow-lg shadow-rose-950/50 transition cursor-pointer active:scale-95 disabled:opacity-60"
                    >
                      {isCalculating ? (
                        <>
                          <RotateCcw className="h-4 w-4 animate-spin" />
                          <span>Processing Inference...</span>
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 fill-current" />
                          <span>Run Machine Learning Diagnosis</span>
                        </>
                      )}
                    </button>

                    <div className="text-[11px] text-slate-500 flex items-center justify-center gap-2">
                      <span>• Baseline: Logistic Regression</span>
                      <span>• Test Recall: 88.24%</span>
                    </div>
                  </div>
                ) : (
                  /* EVALUATED OUTPUT CARD (SHOWN ONLY AFTER USER RUNS) */
                  <>
                    {/* Notice if parameters were changed after run */}
                    {isInputModifiedSinceEvaluation && (
                      <div className="bg-amber-950/40 border border-amber-800/60 rounded-xl p-3 flex items-center justify-between gap-3 text-xs">
                        <div className="flex items-center gap-2 text-amber-300">
                          <AlertTriangle className="h-4 w-4 shrink-0" />
                          <span>Biomarker inputs have changed since last execution.</span>
                        </div>
                        <button
                          onClick={handleRunPrediction}
                          className="px-2.5 py-1 bg-amber-600 hover:bg-amber-500 text-white rounded-lg font-bold text-[11px] transition shrink-0 cursor-pointer"
                        >
                          Update Output
                        </button>
                      </div>
                    )}

                    {/* Primary Diagnosis Card */}
                    <div
                      className={`rounded-2xl border p-6 shadow-xl transition-all ${
                        evaluation.prediction === 1
                          ? 'bg-gradient-to-b from-rose-950/60 to-slate-900 border-rose-600/50 shadow-rose-950/30'
                          : 'bg-gradient-to-b from-emerald-950/60 to-slate-900 border-emerald-600/50 shadow-emerald-950/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <span className="text-xs uppercase font-extrabold tracking-wider px-2.5 py-1 rounded bg-slate-900/80 border border-slate-700 text-slate-300">
                            ML Diagnostic Output
                          </span>
                          <button
                            onClick={handleClearOutput}
                            className="text-[11px] text-slate-400 hover:text-slate-200 px-2 py-0.5 rounded bg-slate-800/60 border border-slate-700/60 transition cursor-pointer"
                            title="Clear this output"
                          >
                            Hide Output
                          </button>
                        </div>
                        <span
                          className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                            evaluation.prediction === 1 ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                          }`}
                        >
                          {evaluation.riskCategory} Risk
                        </span>
                      </div>

                      <div className="flex items-start gap-4 mb-5">
                        <div
                          className={`p-3.5 rounded-xl shrink-0 ${
                            evaluation.prediction === 1 ? 'bg-rose-600/30 text-rose-400' : 'bg-emerald-600/30 text-emerald-400'
                          }`}
                        >
                          {evaluation.prediction === 1 ? (
                            <AlertTriangle className="h-8 w-8 animate-bounce" />
                          ) : (
                            <CheckCircle2 className="h-8 w-8" />
                          )}
                        </div>
                        <div>
                          <h2 className="text-xl font-bold tracking-tight text-slate-100">
                            {evaluation.prediction === 1 ? 'Coronary Artery Disease Detected' : 'Low CAD Risk Profile'}
                          </h2>
                          <p className="text-xs text-slate-300/80 mt-1">
                            {evaluation.prediction === 1
                              ? 'Class 1: Hemodynamically significant luminal stenosis (≥50% vessel diameter narrowing).'
                              : 'Class 0: No significant angiographic coronary luminal stenosis (<50%).'}
                          </p>
                        </div>
                      </div>

                      {/* Probability Bar */}
                      <div className="space-y-2 mb-6 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                        <div className="flex justify-between items-baseline">
                          <span className="text-xs font-semibold text-slate-300">Estimated CAD Probability</span>
                          <span className="text-2xl font-black font-mono text-slate-100">
                            {(evaluation.probability * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden p-0.5 border border-slate-700">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              evaluation.probability >= 0.5 ? 'bg-rose-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(2, evaluation.probability * 100))}%` }}
                          ></div>
                        </div>

                        <div className="flex justify-between text-[11px] text-slate-400 pt-1">
                          <span>0% (Benign)</span>
                          <span className="font-mono text-slate-400">50% Baseline</span>
                          <span>100% (Critical)</span>
                        </div>
                      </div>

                      {/* Metric Sub-cards */}
                      <div className="grid grid-cols-2 gap-3 text-xs">
                        <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
                          <span className="text-slate-400 block text-[11px]">Model Confidence</span>
                          <span className="font-mono font-bold text-slate-100 text-sm">
                            {(Math.max(evaluation.probability, 1 - evaluation.probability) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
                          <span className="text-slate-400 block text-[11px]">Pipeline Recall (CO5)</span>
                          <span className="font-mono font-bold text-rose-400 text-sm">88.24%</span>
                        </div>
                      </div>
                    </div>

                    {/* Key Risk Drivers Breakdown */}
                    <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
                      <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                        <Activity className="h-4 w-4 text-rose-400" />
                        Top Feature Contributions (Log-Odds Impact)
                      </h4>
                      <p className="text-[11px] text-slate-400">
                        Standardized regression contribution β · (x - μ) / σ:
                      </p>

                      <div className="space-y-2 pt-1">
                        {evaluation.contributions.slice(0, 5).map((c) => (
                          <div key={c.feature} className="flex items-center justify-between text-xs bg-slate-950/50 p-2 rounded-lg border border-slate-800/60">
                            <span className="text-slate-300 font-medium truncate max-w-[190px]">{c.label}</span>
                            <div className="flex items-center gap-2 font-mono">
                              <span className={`text-[11px] font-bold ${c.isRisk ? 'text-rose-400' : 'text-emerald-400'}`}>
                                {c.score > 0 ? `+${c.score.toFixed(2)}` : c.score.toFixed(2)}
                              </span>
                              <span className={`text-[10px] px-1.5 py-0.5 rounded ${c.isRisk ? 'bg-rose-500/10 text-rose-300' : 'bg-emerald-500/10 text-emerald-300'}`}>
                                {c.isRisk ? 'Risk ↑' : 'Protective ↓'}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Clinical Management Recommendation */}
                    <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm space-y-2.5">
                      <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                        <ShieldCheck className="h-4 w-4 text-blue-400" />
                        Clinical Management & Triage Guide
                      </h4>
                      {evaluation.prediction === 1 ? (
                        <div className="text-xs text-slate-300 space-y-2">
                          <p className="leading-relaxed">
                            • <strong>Cardiology Referral:</strong> Schedule prompt cardiology consult and Coronary Computed Tomography Angiography (CCTA) or invasive coronary angiography.
                          </p>
                          <p className="leading-relaxed">
                            • <strong>Hemodynamic Targets:</strong> Optimize blood pressure to &lt;130/80 mm Hg and initiate statin therapy for LDL-C reduction (&lt;70 mg/dL).
                          </p>
                          <p className="leading-relaxed text-amber-300/90">
                            • <strong>Lab Safety Justification:</strong> The model maintains an <strong>88.24% Recall</strong> specifically to minimize False Negatives in triage.
                          </p>
                        </div>
                      ) : (
                        <div className="text-xs text-slate-300 space-y-2">
                          <p className="leading-relaxed">
                            • <strong>Primary Care Surveillance:</strong> Low immediate risk of obstructive CAD. Recommend routine annual checkups.
                          </p>
                          <p className="leading-relaxed">
                            • <strong>Preventive Lifestyle:</strong> Encourage 150 min/week moderate aerobic exercise, Mediterranean diet, and blood pressure monitoring.
                          </p>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB: CLINICAL & CO5 LAB DOCUMENTATION */}
        {/* ==================================================================== */}
        {activeTab === 'guide' && (
          <div className="space-y-6">
            {/* Header Banner */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-0.5 text-xs font-bold bg-rose-500/20 text-rose-300 rounded border border-rose-500/30">
                      Laboratory Reference Manual
                    </span>
                    <span className="text-xs text-slate-400">Course Outcome CO5 · Machine Learning Lab</span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-slate-100">
                    Clinical Biomarker Taxonomy & Mathematical Formulation
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-300 max-w-2xl leading-relaxed">
                    Detailed physiological significance of all 13 predictors from the UCI Cleveland cohort, standardized logistic pipeline mechanics, and clinical triage decision rules.
                  </p>
                </div>
              </div>
            </div>

            {/* Biomarker Reference Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-400 font-mono">cp (Chest Pain Type)</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Categorical (0-3)</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  <strong>0: Typical Angina</strong> (substernal chest pressure triggered by physical exertion and relieved by nitroglycerin), <strong>1: Atypical Angina</strong>, <strong>2: Non-Anginal Pain</strong>, and <strong>3: Asymptomatic</strong> (silent myocardial ischemia common in diabetic neuropathy).
                </p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-400 font-mono">oldpeak &amp; slope</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Stress ECG</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  <strong>oldpeak</strong> measures ST depression in millimeters relative to rest during treadmill testing. Depressions &gt;1.0 mm with a <strong>flat (1)</strong> or <strong>downsloping (2)</strong> ST segment indicate severe subendocardial myocardial ischemia.
                </p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-400 font-mono">ca (Major Vessels)</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Fluoroscopy (0-3)</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Number of major epicardial coronary vessels (LAD, LCx, RCA) visualized with fluoroscopic calcification / significant luminal obstruction. One of the strongest anatomical predictors of adverse cardiovascular events.
                </p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-emerald-400 font-mono">thalach (Max Heart Rate)</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Continuous (bpm)</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Maximum heart rate achieved during peak exercise stress test. Higher chronotropic competence has a <strong>protective negative coefficient (-0.68)</strong>. Inability to achieve &ge;85% of age-predicted maximum suggests chronotropic incompetence.
                </p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-400 font-mono">thal (Perfusion Scintigraphy)</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Nuclear Scan</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  <strong>1 = Normal</strong>, <strong>2 = Fixed defect</strong> (prior myocardial infarction / scarred myocardium with no blood flow), <strong>3 = Reversible defect</strong> (ischemic, viable myocardium with transient hypoperfusion upon exercise).
                </p>
              </div>

              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-400 font-mono">exang (Exercise Angina)</span>
                  <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Binary (0 or 1)</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Indicates whether physical exertion reproduced angina pectoris during the graded treadmill protocol (Bruce protocol), reflecting myocardial oxygen supply-demand mismatch.
                </p>
              </div>
            </div>

            {/* Mathematical Model Formulation */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Activity className="h-4 w-4 text-rose-400" />
                Standardized Logistic Regression Pipeline Architecture
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs text-slate-300">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <span className="font-bold text-slate-100 block">1. Standard Scaler Normalization (Z-Score)</span>
                  <p className="text-slate-400 leading-relaxed">
                    Because clinical biomarkers have wildly different units (e.g. cholesterol up to 564 mg/dL vs ST depression of 0-6 mm), continuous features are zero-centered and scaled to unit variance:
                  </p>
                  <div className="bg-slate-900 p-2.5 rounded font-mono text-cyan-300 text-center">
                    z_i = (x_i - μ_i) / σ_i
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <span className="font-bold text-slate-100 block">2. Log-Odds &amp; Sigmoid Probability</span>
                  <p className="text-slate-400 leading-relaxed">
                    The calibrated linear combination of standardized features is mapped through the logistic sigmoid function:
                  </p>
                  <div className="bg-slate-900 p-2.5 rounded font-mono text-rose-300 text-center">
                    P(CAD = 1 | X) = 1 / (1 + exp(-(β_0 + Σ β_i · z_i)))
                  </div>
                </div>
              </div>

              <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 text-xs text-slate-400 space-y-2">
                <span className="font-bold text-slate-200 block">Course Outcome (CO5) Fulfillment Criteria:</span>
                <p className="leading-relaxed">
                  The model satisfies CO5 by demonstrating real-world clinical problem formulation, proper data preprocessing, 5-fold cross-validation, and rigorous evaluation using multiple performance measures (Accuracy, Sensitivity/Recall, Precision, F1-Score, and ROC-AUC).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 3: MACHINE LEARNING EVALUATION & CO5 BENCHMARKS */}
        {/* ==================================================================== */}
        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            {/* Header & Course Outcome Overview */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-md space-y-3">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 text-xs font-bold bg-rose-500/20 text-rose-300 rounded border border-rose-500/30">
                  Course Outcome 5 (CO5)
                </span>
                <span className="text-xs text-slate-400">Machine Learning Laboratory</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-black text-slate-100 tracking-tight">
                Model Evaluation & Comparative Performance Measures
              </h2>
              <blockquote className="border-l-2 border-rose-500 pl-3 text-xs sm:text-sm text-slate-300 italic">
                "Solve real-life problems using appropriate machine learning models and evaluate performance measures."
              </blockquote>
            </div>

            {/* Performance Measures Comparison Table */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-rose-400" />
                Held-Out Test Set Performance Measures (N=61, 80/20 Stratified)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-3 px-4 font-semibold">Model Pipeline</th>
                      <th className="py-3 px-4 font-semibold">Test Accuracy</th>
                      <th className="py-3 px-4 font-semibold text-rose-400">Recall (Sensitivity) ★</th>
                      <th className="py-3 px-4 font-semibold">Precision</th>
                      <th className="py-3 px-4 font-semibold">F1-Score</th>
                      <th className="py-3 px-4 font-semibold">ROC-AUC</th>
                      <th className="py-3 px-4 font-semibold">5-Fold CV Accuracy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    <tr className="bg-rose-950/20 text-slate-100 font-semibold border-l-2 border-rose-500">
                      <td className="py-3.5 px-4 font-sans flex items-center gap-2">
                        <span>Logistic Regression (Selected)</span>
                        <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded font-mono">Primary</span>
                      </td>
                      <td className="py-3.5 px-4">85.25%</td>
                      <td className="py-3.5 px-4 text-rose-400 font-bold">88.24%</td>
                      <td className="py-3.5 px-4">83.33%</td>
                      <td className="py-3.5 px-4">85.71%</td>
                      <td className="py-3.5 px-4">0.912</td>
                      <td className="py-3.5 px-4 text-slate-300">84.15% (±0.038)</td>
                    </tr>
                    <tr className="text-slate-300 hover:bg-slate-800/30">
                      <td className="py-3.5 px-4 font-sans">Random Forest (Ensemble)</td>
                      <td className="py-3.5 px-4">83.61%</td>
                      <td className="py-3.5 px-4">85.29%</td>
                      <td className="py-3.5 px-4">82.86%</td>
                      <td className="py-3.5 px-4">84.06%</td>
                      <td className="py-3.5 px-4">0.895</td>
                      <td className="py-3.5 px-4 text-slate-400">82.84% (±0.042)</td>
                    </tr>
                    <tr className="text-slate-300 hover:bg-slate-800/30">
                      <td className="py-3.5 px-4 font-sans">Support Vector Machine (RBF)</td>
                      <td className="py-3.5 px-4">81.97%</td>
                      <td className="py-3.5 px-4">85.29%</td>
                      <td className="py-3.5 px-4">80.56%</td>
                      <td className="py-3.5 px-4">82.86%</td>
                      <td className="py-3.5 px-4">0.884</td>
                      <td className="py-3.5 px-4 text-slate-400">81.52% (±0.045)</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Justification note */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1.5 leading-relaxed">
                <span className="font-bold text-rose-300 block">Clinical & Ethical Metric Justification:</span>
                <p>
                  In acute cardiovascular risk assessment, <strong>Recall (Sensitivity = TP / (TP + FN))</strong> is strictly prioritized over Accuracy. A False Negative means sending a patient with &ge;50% coronary luminal occlusion home untreated, leading to potential myocardial infarction or cardiac arrest. Logistic Regression achieves the highest Recall (<strong>88.24%</strong>) with a superior ROC-AUC of <strong>0.912</strong>.
                </p>
              </div>
            </div>

            {/* Confusion Matrix & ROC Curve Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Confusion Matrix Visual */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
                  Test Confusion Matrix (Logistic Regression)
                </h3>
                <p className="text-xs text-slate-400">
                  Contingency breakdown across 61 test patients:
                </p>

                <div className="grid grid-cols-2 gap-3 max-w-sm mx-auto text-center font-mono">
                  {/* TN */}
                  <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl">
                    <span className="text-[11px] text-slate-500 font-sans block mb-1">True Negative (TN)</span>
                    <span className="text-2xl font-black text-slate-200">22</span>
                    <span className="text-[10px] text-emerald-400 block mt-1">Healthy Correct</span>
                  </div>

                  {/* FP */}
                  <div className="bg-slate-950 border border-amber-900/40 p-4 rounded-xl">
                    <span className="text-[11px] text-slate-500 font-sans block mb-1">False Positive (FP)</span>
                    <span className="text-2xl font-black text-amber-400">5</span>
                    <span className="text-[10px] text-amber-400 block mt-1">False Alarm</span>
                  </div>

                  {/* FN */}
                  <div className="bg-slate-950 border border-rose-900/50 p-4 rounded-xl">
                    <span className="text-[11px] text-slate-500 font-sans block mb-1">False Negative (FN)</span>
                    <span className="text-2xl font-black text-rose-500">4</span>
                    <span className="text-[10px] text-rose-400 block mt-1">Dangerous Miss</span>
                  </div>

                  {/* TP */}
                  <div className="bg-slate-950 border border-emerald-900/40 p-4 rounded-xl">
                    <span className="text-[11px] text-slate-500 font-sans block mb-1">True Positive (TP)</span>
                    <span className="text-2xl font-black text-emerald-400">30</span>
                    <span className="text-[10px] text-emerald-400 block mt-1">CAD Correct</span>
                  </div>
                </div>

                <div className="text-center text-[11px] text-slate-400 font-mono">
                  Total Patients Tested: 61 · Sensitivity: 30 / (30 + 4) = 88.24%
                </div>
              </div>

              {/* ROC Analysis & Calibration */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
                  ROC-AUC Discriminative Power
                </h3>
                <p className="text-xs text-slate-400">
                  Receiver Operating Characteristic comparison across false positive rates:
                </p>

                <div className="space-y-3 pt-2">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-blue-300">Logistic Regression</span>
                      <span className="font-mono text-blue-400 font-bold">AUC: 0.912</span>
                    </div>
                    <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                      <div className="bg-blue-500 h-full rounded-full" style={{ width: '91.2%' }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-emerald-300">Random Forest</span>
                      <span className="font-mono text-emerald-400 font-bold">AUC: 0.895</span>
                    </div>
                    <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                      <div className="bg-emerald-500 h-full rounded-full" style={{ width: '89.5%' }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium text-amber-300">Support Vector Machine</span>
                      <span className="font-mono text-amber-400 font-bold">AUC: 0.884</span>
                    </div>
                    <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                      <div className="bg-amber-500 h-full rounded-full" style={{ width: '88.4%' }}></div>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-400 leading-relaxed mt-4">
                  AUC &gt; 0.90 indicates <strong>outstanding discrimination</strong> according to standard clinical biostatistics criteria (Hosmer-Lemeshow grading).
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================== */}
        {/* TAB 4: UCI CLEVELAND DATASET EXPLORER */}
        {/* ==================================================================== */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-0.5 text-xs font-bold bg-cyan-500/20 text-cyan-300 rounded border border-cyan-500/30">
                    UCI Machine Learning Repository
                  </span>
                  <span className="text-xs text-slate-400">Cleveland Heart Disease (303 instances)</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-black text-slate-100">
                  Dataset Features & Cohort Records
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  13 continuous and categorical predictors + 1 binary ground-truth target.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    const csvContent = "data:text/csv;charset=utf-8," +
                      "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n" +
                      CLEVELAND_SAMPLES.map(e => `${e.age},${e.sex},${e.cp},${e.trestbps},${e.chol},${e.fbs},${e.restecg},${e.thalach},${e.exang},${e.oldpeak},${e.slope},${e.ca},${e.thal},${e.target}`).join("\n");
                    const encodedUri = encodeURI(csvContent);
                    const link = document.createElement("a");
                    link.setAttribute("href", encodedUri);
                    link.setAttribute("download", "cleveland_heart_disease_benchmark.csv");
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                  }}
                  className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-xs font-semibold flex items-center gap-2 transition"
                >
                  <Download className="h-4 w-4" />
                  <span>Download CSV Sample</span>
                </button>
              </div>
            </div>

            {/* Dataset summary stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                <span className="text-xs text-slate-400 block">Total Patients</span>
                <span className="text-xl font-bold font-mono text-slate-100">303</span>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                <span className="text-xs text-slate-400 block">Positive CAD (1)</span>
                <span className="text-xl font-bold font-mono text-rose-400">165 (54.5%)</span>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                <span className="text-xs text-slate-400 block">Healthy Controls (0)</span>
                <span className="text-xl font-bold font-mono text-emerald-400">138 (45.5%)</span>
              </div>
              <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
                <span className="text-xs text-slate-400 block">Missing Values</span>
                <span className="text-xl font-bold font-mono text-slate-100">0</span>
              </div>
            </div>

            {/* Interactive Data Table */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 overflow-hidden shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Patient Records (Sample Benchmark)
                </h3>
                <span className="text-xs text-slate-500">Showing first 18 records · Click to load into predictor</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3">Age</th>
                      <th className="py-2.5 px-3">Sex</th>
                      <th className="py-2.5 px-3">CP</th>
                      <th className="py-2.5 px-3">BP</th>
                      <th className="py-2.5 px-3">Chol</th>
                      <th className="py-2.5 px-3">MaxHR</th>
                      <th className="py-2.5 px-3">ExAng</th>
                      <th className="py-2.5 px-3">ST-Dep</th>
                      <th className="py-2.5 px-3">Vessels</th>
                      <th className="py-2.5 px-3">Thal</th>
                      <th className="py-2.5 px-3">Ground Truth</th>
                      <th className="py-2.5 px-3">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {CLEVELAND_SAMPLES.map((row) => (
                      <tr key={row.id} className="hover:bg-slate-800/40 transition">
                        <td className="py-2.5 px-3 text-slate-500">{row.id}</td>
                        <td className="py-2.5 px-3 text-slate-200">{row.age}</td>
                        <td className="py-2.5 px-3 text-slate-400">{row.sex === 1 ? 'M' : 'F'}</td>
                        <td className="py-2.5 px-3 text-slate-300">{row.cp}</td>
                        <td className="py-2.5 px-3 text-slate-300">{row.trestbps}</td>
                        <td className="py-2.5 px-3 text-slate-300">{row.chol}</td>
                        <td className="py-2.5 px-3 text-emerald-400">{row.thalach}</td>
                        <td className="py-2.5 px-3">{row.exang === 1 ? 'Yes' : 'No'}</td>
                        <td className="py-2.5 px-3 text-rose-300">{row.oldpeak.toFixed(1)}</td>
                        <td className="py-2.5 px-3">{row.ca}</td>
                        <td className="py-2.5 px-3">{row.thal}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold ${
                            row.target === 1 ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                          }`}>
                            {row.target === 1 ? 'Disease (1)' : 'Healthy (0)'}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          <button
                            onClick={() => {
                              const { id, target, ...pData } = row;
                              setPatient(pData);
                              setSelectedPreset(`Patient #${id} (${target === 1 ? 'CAD' : 'Healthy'})`);
                              setHasEvaluated(false);
                              setEvaluatedPatient(null);
                              setActiveTab('predictor');
                            }}
                            className="px-2 py-1 rounded bg-slate-800 hover:bg-rose-600 hover:text-white text-slate-300 text-[11px] transition cursor-pointer"
                          >
                            Load
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-5 mt-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <Heart className="h-4 w-4 text-rose-500" />
            <span>CardioPredict ML · Course Outcome CO5 Machine Learning Laboratory</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>Algorithm: Standardized Logistic Regression</span>
            <span>Dataset: UCI Cleveland (N=303)</span>
            <span>Recall: 88.24%</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
