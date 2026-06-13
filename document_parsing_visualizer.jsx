import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

// Static assets served from /public/resources/
const BASE = import.meta.env.BASE_URL;
const IMG_CLEAN   = `${BASE}resources/output.jpeg`;       // clean physics paper (input)
const IMG_BOXED   = `${BASE}resources/input.jpeg`;        // physics paper with detection boxes
const IMG_NEMO     = `${BASE}resources/nemo_ocr_input.jpg`;   // Chinese financial report — clean
const IMG_NEMO_OUT = `${BASE}resources/nemo_ocr_output.webp`; // same page with real detection boxes
const IMG_PIZZA    = `${BASE}resources/pizza.jpg`;            // menu photo — perspective + rotation
const IMG_MIXED    = `${BASE}resources/mixed_orientation.png`; // supplement label — 4 text orientations
const IMG_CROP_1   = `${BASE}resources/crop_1_page.png`;       // page with rotated text + detected polygon
const IMG_CROP_2   = `${BASE}resources/crop_2_extracted.png`;  // cropped rotated text
const IMG_CROP_3   = `${BASE}resources/crop_3_deskewed.png`;   // de-skewed
const IMG_CROP_4   = `${BASE}resources/crop_4_normalized.png`; // resized to recognizer input

// ─── Chapter config ───────────────────────────────────────────────────────────

const pages = [
  { id: 'tasks',      title: 'What can you do with a document?',     subtitle: 'A taxonomy of document understanding' },
  { id: 'ocr',        title: 'Classical OCR: detect then recognize', subtitle: 'The two-stage pipeline that ruled the field' },
  { id: 'vlm',        title: 'Enter the vision-language model',      subtitle: 'End-to-end, image-in text-out' },
  { id: 'failures',   title: 'Where VLMs still fail',                 subtitle: 'Hallucination, repetition, malformed output, latency' },
  { id: 'training',   title: 'Training & data',                       subtitle: 'How modern doc models are getting smarter' },
  { id: 'inference',  title: 'Inference & architecture',              subtitle: 'VRFM, diffusion decoding, and the speed race' },
  { id: 'landscape',  title: 'The current model landscape',           subtitle: 'MinerU · olmOCR · PaddleOCR-VL · Nemotron · Chandra' },
];

// ─── Shared helpers ───────────────────────────────────────────────────────────

function SectionLabel({ color = '#2563eb', children }) {
  return (
    <span className="inline-block rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wider text-white mb-2" style={{ background: color }}>
      {children}
    </span>
  );
}

function InfoBox({ color = '#2563eb', title, children }) {
  return (
    <div className="rounded-xl border-l-4 p-4" style={{ borderColor: color, background: color + '10' }}>
      {title && <div className="font-semibold mb-1" style={{ color }}>{title}</div>}
      <div className="text-sm opacity-85">{children}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 1 — What can you do with a document?
// ═══════════════════════════════════════════════════════════════════════════════

const TASKS = [
  {
    id: 'ocr',
    label: 'Text Extraction',
    emoji: '📄',
    color: '#2563eb',
    desc: 'The most fundamental task: convert pixels of printed or handwritten text into a machine-readable string. Works across fonts, sizes, and languages.',
    output: 'Q3 2024 Earnings Report\n\nRevenue grew 18% year-over-year,\ndriven by enterprise adoption in\nNorth America and APAC regions.\n\nOperating margin improved to 24.3%,\nup from 21.1% in the prior year.',
    outputLabel: 'Extracted plain text',
    models: ['Tesseract 5', 'EasyOCR', 'PaddleOCR', 'GOT-OCR 2.0'],
  },
  {
    id: 'layout',
    label: 'Layout Analysis',
    emoji: '🗂',
    color: '#0d9488',
    desc: 'Identify the reading order and semantic role of each region. This is a real physics preprint — a two-column paper with inline LaTeX formulas, density plots, and figure captions. The model must handle all of them simultaneously.',
    output: null,
    outputLabel: 'Detected regions — actual model output',
    models: ['DocLayout-YOLO', 'PP-DocLayoutV3', 'Surya 2', 'LayoutLMv3'],
  },
  {
    id: 'table',
    label: 'Table Extraction',
    emoji: '📊',
    color: '#d97706',
    desc: 'Recover rows, columns, spanning cells, and headers from a table image — outputting structured HTML, LaTeX, or JSON. Handles merged cells and borderless grids.',
    output: '| Product  |  Q3   |  Q4   |\n|----------|------:|------:|\n| Widget A | 1,240 | 1,890 |\n| Widget B |   870 | 1,100 |\n| Widget C | 2,300 | 2,750 |\n| Total    | 4,410 | 5,740 |',
    outputLabel: 'Extracted as Markdown table',
    models: ['TableTransformer', 'Nougat', 'Qwen2-VL', 'GPT-4o'],
  },
  {
    id: 'chart',
    label: 'Chart Understanding',
    emoji: '📈',
    color: '#7c3aed',
    desc: 'Read a chart\'s axes, labels, and bar/line values, then return the underlying data as CSV or answer questions about trends. Requires understanding both visual encoding and domain semantics.',
    output: 'Year,Revenue ($M)\n2021,15.2\n2022,19.7\n2023,24.1\n2024,29.8',
    outputLabel: 'Data recovered as CSV',
    models: ['DePlot', 'ChartQA fine-tunes', 'Qwen2-VL', 'Gemini 2.0'],
  },
  {
    id: 'kv',
    label: 'Key-Value Extraction',
    emoji: '🧾',
    color: '#16a34a',
    desc: 'Map labeled fields to their values on invoices, forms, receipts, and ID documents. Enables zero-touch data entry and automated document workflows.',
    output: '{\n  "vendor": "Acme Corp Ltd",\n  "invoice_number": "INV-2024-0391",\n  "date": "2024-10-01",\n  "due_date": "2024-10-31",\n  "total_due": "$4,536.00"\n}',
    outputLabel: 'Extracted as structured JSON',
    models: ['LayoutLM', 'Donut', 'InternVL2', 'Claude 3.5'],
  },
  {
    id: 'classify',
    label: 'Classification & Routing',
    emoji: '🏷',
    color: '#dc2626',
    desc: 'Identify the document type (invoice, contract, passport, lab report) and route it to the right downstream workflow — without reading every word. Often the first step in a document pipeline.',
    output: '{\n  "doc_type": "signed_contract",\n  "confidence": 0.97,\n  "has_signature": true,\n  "language": "en",\n  "page_count": 12\n}',
    outputLabel: 'Classification metadata',
    models: ['ViT classifiers', 'LayoutLM', 'Donut', 'GPT-4o Vision'],
  },
];

// ─── Compact per-task document SVGs (viewBox 0 0 260 200) ────────────────────

function OcrDocSVGTask() {
  return (
    <svg viewBox="0 0 260 200" className="w-full rounded-xl border bg-white" style={{ display: 'block' }}>
      <rect width="260" height="200" fill="white" />
      <rect x="12" y="12" width="236" height="176" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      {/* Title bar */}
      <rect x="22" y="22" width="140" height="11" rx="2" fill="#1e293b" />
      <rect x="22" y="39" width="100" height="7" rx="2" fill="#94a3b8" />
      <line x1="22" y1="52" x2="238" y2="52" stroke="#e2e8f0" strokeWidth="0.8" />
      {/* Body lines — groups of 3 with a gap between paragraphs */}
      {[
        [22, 60, 210], [22, 71, 190], [22, 82, 216],
        [22, 97, 200], [22, 108, 180], [22, 119, 208],
        [22, 134, 195], [22, 145, 170], [22, 156, 205],
      ].map(([x, y, w], i) => (
        <rect key={i} x={x} y={y} width={w} height="6" rx="2" fill="#cbd5e1" />
      ))}
      {/* Blue highlight boxes over lines 1-6 showing what's being extracted */}
      <rect x="20" y="57" width="218" height="69" rx="3"
        fill="#2563eb0a" stroke="#2563eb" strokeWidth="1.2" strokeDasharray="5 3" />
      {/* Small "OCR" badge inside the highlight — no rotation */}
      <rect x="188" y="58" width="48" height="14" rx="3" fill="#2563eb" />
      <text x="212" y="68" textAnchor="middle" fontSize="7.5" fontWeight="700" fill="white">OCR ↓</text>
    </svg>
  );
}

const LAYOUT_REGIONS_COMPACT = [
  { x: 22, y: 20, w: 216, h: 14, label: 'Title', color: '#2563eb' },
  { x: 22, y: 39, w: 216, h: 28, label: 'Abstract', color: '#0d9488' },
  { x: 22, y: 73, w: 102, h: 60, label: 'Body col 1', color: '#16a34a' },
  { x: 132, y: 73, w: 106, h: 60, label: 'Body col 2', color: '#16a34a' },
  { x: 22, y: 139, w: 102, h: 38, label: 'Figure', color: '#7c3aed' },
  { x: 132, y: 139, w: 106, h: 38, label: 'References', color: '#dc2626' },
];

function LayoutDocSVGTask() {
  const [hovered, setHovered] = useState(null);
  return (
    <svg viewBox="0 0 260 190" className="w-full rounded-xl border bg-white" style={{ display: 'block', cursor: 'default' }}>
      <rect width="260" height="190" fill="white" />
      <rect x="12" y="10" width="236" height="170" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      {/* Faint content lines behind regions */}
      {[80, 89, 98, 107, 116, 125].map(y => (
        <rect key={y} x="26" y={y} width="94" height="4" rx="1" fill="#e2e8f0" />
      ))}
      {[80, 89, 98, 107, 116, 125].map(y => (
        <rect key={y} x="136" y={y} width="98" height="4" rx="1" fill="#e2e8f0" />
      ))}
      <rect x="26" y="143" width="94" height="30" rx="2" fill="#ede9fe" />
      <text x="73" y="162" textAnchor="middle" fontSize="7" fill="#7c3aed" opacity="0.5">[ fig ]</text>
      {[143, 152, 161, 166].map(y => (
        <rect key={y} x="136" y={y} width={y % 9 === 0 ? 90 : 75} height="4" rx="1" fill="#e2e8f0" />
      ))}
      {/* Region overlays */}
      {LAYOUT_REGIONS_COMPACT.map(r => (
        <g key={r.label}
          onMouseEnter={() => setHovered(r.label)}
          onMouseLeave={() => setHovered(null)}
        >
          <rect x={r.x} y={r.y} width={r.w} height={r.h} rx="2"
            fill={hovered === r.label ? r.color + '30' : r.color + '18'}
            stroke={r.color}
            strokeWidth={hovered === r.label ? 2 : 1}
            style={{ transition: 'all 0.12s' }}
          />
          <rect x={r.x + 2} y={r.y + 2} width={Math.min(r.label.length * 5, r.w - 4)} height="10" rx="2" fill={r.color} />
          <text x={r.x + 5} y={r.y + 10} fontSize="6.5" fontWeight="700" fill="white">{r.label}</text>
        </g>
      ))}
    </svg>
  );
}

function TableDocSVGTask() {
  const [activeRow, setActiveRow] = useState(null);
  const rows = [
    { label: 'Widget A', q3: '1,240', q4: '1,890' },
    { label: 'Widget B', q3: '870',   q4: '1,100' },
    { label: 'Widget C', q3: '2,300', q4: '2,750' },
  ];
  const C = '#d97706';
  return (
    <svg viewBox="0 0 260 190" className="w-full rounded-xl border bg-white" style={{ display: 'block' }}>
      <rect width="260" height="190" fill="white" />
      <rect x="12" y="10" width="236" height="170" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      {/* Doc title lines */}
      <rect x="22" y="18" width="120" height="9" rx="2" fill="#1e293b" />
      <rect x="22" y="32" width="80" height="6" rx="2" fill="#94a3b8" />
      {/* Table header */}
      <rect x="22" y="46" width="226" height="18" rx="3" fill={C} />
      <text x="66" y="58" textAnchor="middle" fontSize="8" fontWeight="700" fill="white">Product</text>
      <text x="152" y="58" textAnchor="middle" fontSize="8" fontWeight="700" fill="white">Q3 Sales</text>
      <text x="222" y="58" textAnchor="middle" fontSize="8" fontWeight="700" fill="white">Q4 Sales</text>
      {/* Col dividers */}
      <line x1="108" y1="46" x2="108" y2="152" stroke="white" strokeWidth="0.5" opacity="0.4" />
      <line x1="196" y1="46" x2="196" y2="152" stroke="white" strokeWidth="0.5" opacity="0.4" />
      {/* Data rows */}
      {rows.map((row, i) => {
        const y = 64 + i * 22;
        const active = activeRow === i;
        return (
          <g key={i} onMouseEnter={() => setActiveRow(i)} onMouseLeave={() => setActiveRow(null)}>
            <rect x="22" y={y} width="226" height="21"
              fill={active ? C + '22' : i % 2 === 0 ? '#fafafa' : 'white'}
              stroke={active ? C : '#e2e8f0'} strokeWidth={active ? 1.5 : 0.5}
              style={{ transition: 'all 0.12s' }}
            />
            <line x1="108" y1={y} x2="108" y2={y + 21} stroke="#e2e8f0" strokeWidth="0.5" />
            <line x1="196" y1={y} x2="196" y2={y + 21} stroke="#e2e8f0" strokeWidth="0.5" />
            <text x="66"  y={y + 13} textAnchor="middle" fontSize="8" fill="#334155">{row.label}</text>
            <text x="152" y={y + 13} textAnchor="middle" fontSize="8" fill="#334155">{row.q3}</text>
            <text x="222" y={y + 13} textAnchor="middle" fontSize="8" fill="#334155">{row.q4}</text>
          </g>
        );
      })}
      {/* Total row */}
      <rect x="22" y="130" width="226" height="20" rx="0" fill={C + '15'} stroke={C} strokeWidth="1" />
      <line x1="108" y1="130" x2="108" y2="150" stroke={C} strokeWidth="0.5" opacity="0.5" />
      <line x1="196" y1="130" x2="196" y2="150" stroke={C} strokeWidth="0.5" opacity="0.5" />
      <text x="66"  y="143" textAnchor="middle" fontSize="8" fontWeight="700" fill="#92400e">Total</text>
      <text x="152" y="143" textAnchor="middle" fontSize="8" fontWeight="700" fill="#92400e">4,410</text>
      <text x="222" y="143" textAnchor="middle" fontSize="8" fontWeight="700" fill="#92400e">5,740</text>
      <rect x="22" y="46" width="226" height="104" rx="3" fill="none" stroke={C} strokeWidth="1.5" />
    </svg>
  );
}

const CHART_BARS_COMPACT = [
  { year: '2021', val: 15.2, h: 52 },
  { year: '2022', val: 19.7, h: 68 },
  { year: '2023', val: 24.1, h: 83 },
  { year: '2024', val: 29.8, h: 103 },
];

function ChartDocSVGTask() {
  const [activeBar, setActiveBar] = useState(null);
  const C = '#7c3aed';
  const baseY = 162;
  const barW = 34;
  const gap = 16;
  const startX = 54;
  return (
    <svg viewBox="0 0 260 190" className="w-full rounded-xl border bg-white" style={{ display: 'block' }}>
      <rect width="260" height="190" fill="white" />
      <rect x="12" y="10" width="236" height="170" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      {/* Title */}
      <rect x="22" y="18" width="160" height="8" rx="2" fill="#1e293b" />
      {/* Grid lines + Y labels */}
      {[0, 10, 20, 30].map(v => {
        const y = baseY - (v / 30) * 110;
        return (
          <g key={v}>
            <line x1="44" y1={y} x2="244" y2={y} stroke="#f1f5f9" strokeWidth="1" />
            <text x="42" y={y + 3} textAnchor="end" fontSize="6.5" fill="#94a3b8">{v}M</text>
          </g>
        );
      })}
      {/* Axes */}
      <line x1="44" y1="32" x2="44" y2={baseY} stroke="#cbd5e1" strokeWidth="1" />
      <line x1="44" y1={baseY} x2="244" y2={baseY} stroke="#cbd5e1" strokeWidth="1" />
      {/* Bars */}
      {CHART_BARS_COMPACT.map((b, i) => {
        const x = startX + i * (barW + gap);
        const active = activeBar === i;
        return (
          <g key={b.year} onMouseEnter={() => setActiveBar(i)} onMouseLeave={() => setActiveBar(null)}>
            <rect x={x} y={baseY - b.h} width={barW} height={b.h} rx="3"
              fill={active ? C : C + '66'} stroke={C} strokeWidth="1"
              style={{ transition: 'all 0.12s' }}
            />
            {active && (
              <>
                <rect x={x - 4} y={baseY - b.h - 17} width={barW + 8} height="14" rx="3" fill={C} />
                <text x={x + barW / 2} y={baseY - b.h - 6} textAnchor="middle" fontSize="7.5" fontWeight="700" fill="white">${b.val}M</text>
                <line x1="44" y1={baseY - b.h} x2={x} y2={baseY - b.h} stroke={C} strokeWidth="1" strokeDasharray="3 2" />
              </>
            )}
            <text x={x + barW / 2} y={baseY + 11} textAnchor="middle" fontSize="7" fill="#64748b">{b.year}</text>
          </g>
        );
      })}
    </svg>
  );
}

const KV_INVOICE_FIELDS = [
  { key: 'Vendor',    value: 'Acme Corp Ltd',  x: 22, y: 52, w: 112, h: 18 },
  { key: 'Invoice #', value: 'INV-2024-0391',  x: 144, y: 52, w: 104, h: 18 },
  { key: 'Date',      value: '2024-10-01',      x: 22, y: 76, w: 112, h: 18 },
  { key: 'Due Date',  value: '2024-10-31',      x: 144, y: 76, w: 104, h: 18 },
  { key: 'Subtotal',  value: '$4,200.00',       x: 144, y: 126, w: 104, h: 16 },
  { key: 'Tax (8%)',  value: '$336.00',          x: 144, y: 146, w: 104, h: 16 },
  { key: 'Total Due', value: '$4,536.00',        x: 144, y: 166, w: 104, h: 18 },
];

function KvDocSVGTask({ onHover, hovered }) {
  const C = '#16a34a';
  return (
    <svg viewBox="0 0 260 196" className="w-full rounded-xl border bg-white" style={{ display: 'block', cursor: 'default' }}>
      <rect width="260" height="196" fill="white" />
      <rect x="12" y="10" width="236" height="176" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      {/* Header */}
      <rect x="22" y="18" width="216" height="22" rx="3" fill="#1e293b" />
      <text x="130" y="33" textAnchor="middle" fontSize="11" fontWeight="800" fill="white" letterSpacing="3">INVOICE</text>
      {/* Field boxes */}
      {KV_INVOICE_FIELDS.map(f => {
        const active = hovered === f.key;
        return (
          <g key={f.key}
            onMouseEnter={() => onHover(f.key)}
            onMouseLeave={() => onHover(null)}
          >
            <rect x={f.x} y={f.y} width={f.w} height={f.h} rx="3"
              fill={active ? C + '28' : C + '0e'}
              stroke={active ? C : C + '55'}
              strokeWidth={active ? 2 : 1}
              style={{ transition: 'all 0.12s' }}
            />
            <text x={f.x + 4} y={f.y + 7} fontSize="5.5" fill={C} fontWeight="700" opacity="0.8">{f.key}</text>
            <text x={f.x + 4} y={f.y + 15} fontSize="7.5" fill="#1e293b" fontWeight={active ? '700' : '400'}>{f.value}</text>
          </g>
        );
      })}
      {/* Line items header */}
      <rect x="22" y="100" width="216" height="14" rx="2" fill="#f1f5f9" />
      <text x="26" y="110" fontSize="6" fill="#64748b" fontWeight="600">Description</text>
      <text x="192" y="110" fontSize="6" fill="#64748b" fontWeight="600">Amount</text>
      {[['Consulting (Oct) · 20h', '$3,000.00', 120], ['Software license (annual)', '$1,200.00', 132]].map(([d, a, y]) => (
        <g key={y}>
          <text x="26" y={y} fontSize="7" fill="#334155">{d}</text>
          <text x="192" y={y} fontSize="7" fill="#334155">{a}</text>
        </g>
      ))}
      <line x1="22" y1="138" x2="238" y2="138" stroke="#e2e8f0" strokeWidth="0.8" />
    </svg>
  );
}

function ClassifyDocSVGTask() {
  const C = '#dc2626';
  return (
    <svg viewBox="0 0 260 190" className="w-full rounded-xl border bg-white" style={{ display: 'block' }}>
      <rect width="260" height="190" fill="white" />
      <rect x="12" y="10" width="236" height="170" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
      <text x="130" y="30" textAnchor="middle" fontSize="11" fontWeight="800" fill="#1e293b" letterSpacing="3">SERVICE AGREEMENT</text>
      <line x1="22" y1="36" x2="238" y2="36" stroke="#1e293b" strokeWidth="0.8" />
      {[44, 54, 64, 74, 84].map(y => (
        <rect key={y} x="22" y={y} width={y % 20 === 0 ? 200 : 214} height="6" rx="2" fill="#e2e8f0" />
      ))}
      {/* Signature box */}
      <rect x="22" y="102" width="100" height="42" rx="4" fill="#fff7ed" stroke={C} strokeWidth="1.5" strokeDasharray="4 2" />
      <text x="72" y="115" textAnchor="middle" fontSize="6" fill={C} fontWeight="600">AUTHORIZED SIGNATURE</text>
      <path d="M32,136 C40,126 50,145 60,133 S78,122 88,131 S104,138 115,130" stroke="#1e293b" strokeWidth="1.8" fill="none" strokeLinecap="round" />
      {/* Stamp */}
      <circle cx="192" cy="124" r="30" fill="none" stroke={C} strokeWidth="1.8" opacity="0.7" />
      <circle cx="192" cy="124" r="23" fill="none" stroke={C} strokeWidth="1" opacity="0.45" />
      <text x="192" y="119" textAnchor="middle" fontSize="6.5" fill={C} fontWeight="700" opacity="0.8">CERTIFIED</text>
      <text x="192" y="128" textAnchor="middle" fontSize="6" fill={C} opacity="0.7">ACME CORP</text>
      <text x="192" y="136" textAnchor="middle" fontSize="5.5" fill={C} opacity="0.6">2024-10-01</text>
      {/* Classification result badge at bottom */}
      <rect x="72" y="160" width="116" height="14" rx="7" fill={C} />
      <text x="130" y="170" textAnchor="middle" fontSize="7" fontWeight="700" fill="white">Signed Contract ✓</text>
    </svg>
  );
}

// ─── Layout analysis: real image panels ──────────────────────────────────────

// Color legend matching the actual detection output in IMG_BOXED
const LAYOUT_LEGEND = [
  { label: 'Text / Paragraph', color: '#16a34a' },
  { label: 'Formula',          color: '#dc2626' },
  { label: 'Figure / Picture', color: '#2563eb' },
  { label: 'Caption',          color: '#d97706' },
  { label: 'Section header',   color: '#7c3aed' },
  { label: 'Page header',      color: '#64748b' },
];

function LayoutDocRealImage() {
  return (
    <div className="rounded-xl border overflow-hidden bg-white" style={{ height: 260 }}>
      <img
        src={IMG_CLEAN}
        alt="Clean physics preprint — input to layout model"
        style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top center', display: 'block' }}
      />
    </div>
  );
}

function LayoutOutputRealImage() {
  return (
    <div className="flex flex-col gap-2 h-full">
      <div className="rounded-xl border overflow-hidden bg-white flex-1" style={{ minHeight: 0 }}>
        <img
          src={IMG_BOXED}
          alt="Layout analysis output — colored bounding boxes per region class"
          style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top center', display: 'block' }}
        />
      </div>
      {/* Color legend */}
      <div className="rounded-xl border p-2 flex flex-wrap gap-x-3 gap-y-1">
        {LAYOUT_LEGEND.map(l => (
          <div key={l.label} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: l.color }} />
            <span className="text-xs">{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── KV output panel with bidirectional hover ─────────────────────────────────

function KvOutputPanel({ hovered, onHover, color }) {
  return (
    <div className="rounded-xl border h-full flex flex-col" style={{ background: '#0f172a', borderColor: '#1e293b' }}>
      <div className="px-3 pt-3 pb-1 text-xs font-semibold uppercase tracking-wider" style={{ color: '#64748b' }}>
        Extracted JSON
      </div>
      <div className="flex-1 overflow-auto px-3 pb-3 space-y-0.5 font-mono text-xs">
        <div style={{ color: '#94a3b8' }}>{`{`}</div>
        {KV_INVOICE_FIELDS.map(f => {
          const key = f.key.toLowerCase().replace(/\s+/g, '_').replace('#', 'number').replace(/[()%]/g, '');
          const active = hovered === f.key;
          return (
            <div key={f.key}
              className="pl-3 rounded py-0.5 transition-all"
              style={{ background: active ? color + '28' : 'transparent', color: active ? '#fff' : '#e2e8f0' }}
              onMouseEnter={() => onHover(f.key)}
              onMouseLeave={() => onHover(null)}
            >
              <span style={{ color: active ? '#93c5fd' : '#7dd3fc' }}>"{key}"</span>
              <span style={{ color: '#94a3b8' }}>: </span>
              <span style={{ color: active ? '#bbf7d0' : '#86efac' }}>"{f.value}"</span>
            </div>
          );
        })}
        <div style={{ color: '#94a3b8' }}>{`}`}</div>
      </div>
    </div>
  );
}

// ─── Generic text output panel ────────────────────────────────────────────────

function TextOutputPanel({ label, content }) {
  return (
    <div className="rounded-xl border h-full flex flex-col" style={{ background: '#0f172a', borderColor: '#1e293b' }}>
      <div className="px-3 pt-3 pb-1 text-xs font-semibold uppercase tracking-wider" style={{ color: '#64748b' }}>
        {label}
      </div>
      <pre className="flex-1 overflow-auto px-3 pb-3 text-xs font-mono leading-relaxed"
        style={{ color: '#e2e8f0', whiteSpace: 'pre-wrap', margin: 0 }}>
        {content}
      </pre>
    </div>
  );
}

// ─── TasksPage ─────────────────────────────────────────────────────────────────

function TasksPage() {
  const [activeTask, setActiveTask] = useState('ocr');
  const [kvHovered, setKvHovered] = useState(null);
  const active = TASKS.find(t => t.id === activeTask);

  function renderDoc() {
    switch (activeTask) {
      case 'ocr':      return <OcrDocSVGTask />;
      case 'layout':   return <LayoutDocRealImage />;
      case 'table':    return <TableDocSVGTask />;
      case 'chart':    return <ChartDocSVGTask />;
      case 'kv':       return <KvDocSVGTask onHover={setKvHovered} hovered={kvHovered} />;
      case 'classify': return <ClassifyDocSVGTask />;
      default:         return null;
    }
  }

  function renderOutput() {
    if (activeTask === 'layout') return <LayoutOutputRealImage />;
    if (activeTask === 'kv') return <KvOutputPanel hovered={kvHovered} onHover={setKvHovered} color={active.color} />;
    if (active.output) return <TextOutputPanel label={active.outputLabel} content={active.output} />;
    return null;
  }

  const docHint = {
    table: 'hover rows',
    chart: 'hover bars',
    kv: 'hover fields',
  }[activeTask];

  return (
    <div className="space-y-4">
      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-5 md:p-6 space-y-4">

          {/* Task selector */}
          <div className="flex flex-wrap gap-2">
            {TASKS.map(t => (
              <button
                key={t.id}
                onClick={() => { setActiveTask(t.id); setKvHovered(null); }}
                className="rounded-xl px-3 py-1.5 text-sm font-semibold transition-all flex items-center gap-1.5"
                style={{
                  background: activeTask === t.id ? t.color : 'transparent',
                  color: activeTask === t.id ? 'white' : '#64748b',
                  border: `1.5px solid ${activeTask === t.id ? t.color : '#e2e8f0'}`,
                }}
              >
                <span>{t.emoji}</span>
                <span>{t.label}</span>
              </button>
            ))}
          </div>

          {/* Document | Output side by side */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTask}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.16 }}
            >
              <div className="grid grid-cols-2 gap-3" style={{ minHeight: 200 }}>
                {/* Left: document */}
                <div className="flex flex-col gap-1">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                    Input document
                    {docHint && <span className="normal-case font-normal opacity-60">— {docHint}</span>}
                  </div>
                  {renderDoc()}
                </div>

                {/* Right: output */}
                <div className="flex flex-col gap-1">
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {active.outputLabel || 'Extracted output'}
                  </div>
                  <div className="flex-1">
                    {renderOutput()}
                  </div>
                </div>
              </div>

              {/* Description + models below */}
              <div className="mt-4 grid md:grid-cols-[1fr_auto] gap-4 items-start">
                <p className="text-sm opacity-80 leading-relaxed">{active.desc}</p>
                <div className="flex flex-wrap gap-1.5 md:justify-end">
                  {active.models.map(m => (
                    <span key={m} className="rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap"
                      style={{ borderColor: active.color, color: active.color }}>
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

        </CardContent>
      </Card>

      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-5">
          <InfoBox color="#2563eb" title="The key insight">
            Every document task requires some combination of <strong>localization</strong> (where is the content?),
            <strong> recognition</strong> (what does it say?), and <strong>structure recovery</strong> (how does it relate to other content?).
            Old-school pipelines handled each separately. Modern VLMs attempt all three in a single forward pass.
          </InfoBox>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 2 — Classical OCR
// ═══════════════════════════════════════════════════════════════════════════════

const OCR_STEPS = [
  {
    id: 0,
    label: 'Raw Document Image',
    desc: 'Input: a raster image (scan, photo, PDF render). Pixels only — no semantic information. This is what every OCR system starts with.',
  },
  {
    id: 1,
    label: 'Text Detection',
    desc: 'A detection model (CRAFT, DBNet, TextSnake) finds bounding polygons around text regions. It learns to identify text-like textures and edge patterns. Outputs: a set of boxes/polygons, no characters yet.',
    sublabel: 'CRAFT · DBNet · TextSnake',
  },
  {
    id: 2,
    label: 'Region Cropping',
    desc: 'Each detected bounding box is cropped and optionally rectified (if rotated). These fixed-size strips are then fed to the recognizer independently — the two stages do not share weights.',
    sublabel: 'TPS perspective transform',
  },
  {
    id: 3,
    label: 'Text Recognition',
    desc: 'A CRNN (CNN feature extractor → BiLSTM sequence model → CTC decoder) reads characters from each strip. CTC allows alignment-free training between image columns and output characters.',
    sublabel: 'CRNN · CTC · Attention-seq2seq',
  },
  {
    id: 4,
    label: 'Post-processing',
    desc: 'Detected strings are sorted by reading order (top-left to bottom-right or layout-aware), optionally spell-corrected, and merged into a full-document transcript or structured output.',
    sublabel: 'Reading order · spell check · heuristics',
  },
];

const OCR_LIMITATIONS = [
  { title: 'Multi-column layouts', desc: 'Heuristic reading-order fails on newspapers, academic PDFs, and multi-panel documents where columns interleave.', color: '#dc2626' },
  { title: 'Dense / borderless tables', desc: 'Row and column structure is invisible to a detector trained on text regions — it sees a grid of text boxes with no notion of spanning cells or header rows.', color: '#dc2626' },
  { title: 'Rotated & perspective text', desc: 'TPS rectification helps but requires a good initial detection polygon. Extreme rotation, fisheye distortion, or curved text surfaces cause alignment errors.', color: '#dc2626' },
  { title: 'Two-stage error cascade', desc: 'A recognition error cannot be fixed by looking at the full page — the crop threw that context away. A bad detection box means the recognizer never sees the text at all.', color: '#dc2626' },
];

// (overlay approach removed — real annotated image used instead)

// Curated excerpt from nemo_ocr.txt — flat stream showing lost table structure
const NEMO_FLAT_OUTPUT = `财信证券
CHASING SECURITIES
公司研究报告
财务预测摘要
人民币位
2020A  2021A  2022E  2023E  2024E
利润表
净利息收入
2,533.78
2,693.82
2,837.71
3,041.79
3,284.97
净利润增速
5.42%
18.96%
13.93%
14.48%
15.68%
手续费及佣金
164.95
220.07
300.85
387.35
496.56
营业收入
2,862.21
3,187.64
3,495.00
3,834.68
4,239.78
资产负债表
贷款
55,124  62,372  69,610  78,330  86,546
同业资产
5,520   6,361   7,077   7,821   8,626
...`;

// What the table actually looks like (structured)
const NEMO_STRUCTURED = [
  ['指标', '2020A', '2021A', '2022E', '2023E', '2024E'],
  ['净利息收入', '2,533', '2,694', '2,838', '3,042', '3,285'],
  ['净利润增速', '5.42%', '18.96%', '13.93%', '14.48%', '15.68%'],
  ['手续费及佣金', '165', '220', '301', '387', '497'],
  ['营业收入', '2,862', '3,188', '3,495', '3,835', '4,240'],
];

// Which image to show at each step
// Steps 0: clean input; steps 1-3: real annotated output; step 4: clean input again (for the output comparison)
function OcrDocReal({ step }) {
  const useAnnotated = step >= 1;
  const src = useAnnotated ? IMG_NEMO_OUT : IMG_NEMO;
  const alt = useAnnotated
    ? 'Model detection output — real bounding boxes on the financial report'
    : '财信证券 financial report — Chasing Securities research document (clean input)';

  return (
    <div className="space-y-1.5">
      <div className="text-xs text-muted-foreground font-medium">
        {step === 0 ? '↓ Input document' : '↓ Actual model detection output'}
      </div>
      <div className="rounded-xl border overflow-hidden bg-white">
        <img
          src={src}
          alt={alt}
          style={{ width: '100%', display: 'block', maxHeight: 720, objectFit: 'contain', objectPosition: 'top center' }}
        />
      </div>
    </div>
  );
}

// 4-stage crop pipeline used in Chapter 2 step 2 (replaces the document image in the main panel)
function CropPipeline() {
  const STAGES = [
    { n: 1, title: 'Detection polygon',     color: '#dc2626', src: IMG_CROP_1, caption: 'Detector outputs a 4-point polygon around the text region. The polygon is rotated, not axis-aligned — the model preserved the angle.',                        transition: 'Crop pixels inside the polygon' },
    { n: 2, title: 'Cropped polygon',       color: '#dc2626', src: IMG_CROP_2, caption: 'Just the pixels inside the polygon are extracted. Still rotated. The recognizer cannot consume this directly — sequence models expect horizontal scanlines.',     transition: 'Apply TPS or affine warp' },
    { n: 3, title: 'Rectified (de-skewed)', color: '#d97706', src: IMG_CROP_3, caption: 'Thin-plate spline or affine transform rotates the crop back to flat horizontal. Letter shapes are now aligned to a single baseline.',                              transition: 'Resize to recognizer input size' },
    { n: 4, title: 'Normalized strip',      color: '#16a34a', src: IMG_CROP_4, caption: 'Resized to a fixed 200×32 px input. The CRNN expects this exact shape — fixed height (32), variable width (typically padded/truncated to 200). Ready for inference.' },
  ];

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Region cropping — what happens to one detected polygon
      </div>
      {STAGES.map((s, i) => (
        <div key={s.n}>
          <div className="rounded-2xl border overflow-hidden" style={{ borderColor: s.color + '55' }}>
            <div className="px-4 py-2.5 flex items-center gap-3" style={{ background: s.color + '12' }}>
              <div className="flex-shrink-0 flex items-center justify-center font-bold text-sm rounded-lg"
                style={{ background: s.color, color: 'white', width: 28, height: 28 }}
              >
                {s.n}
              </div>
              <div className="text-sm font-bold" style={{ color: s.color }}>{s.title}</div>
            </div>
            <div className="grid md:grid-cols-[1fr_1fr] gap-4 p-4 items-center">
              <div className="flex items-center justify-center">
                <img src={s.src} alt={s.title}
                  style={{
                    maxHeight: s.n === 1 ? 160 : 110,
                    maxWidth: '100%',
                    display: 'block',
                    borderRadius: 4,
                  }}
                />
              </div>
              <p className="text-xs opacity-85 leading-relaxed">{s.caption}</p>
            </div>
          </div>
          {/* Transition arrow */}
          {s.transition && (
            <div className="flex flex-col items-center py-2">
              <div className="text-xs italic text-muted-foreground">↓ {s.transition}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function OcrPage() {
  const [step, setStep] = useState(0);
  const current = OCR_STEPS[step];

  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-6 md:p-8">
          {/* 1/4 left sidebar | 3/4 document image */}
          <div className="grid xl:grid-cols-[1fr_3fr] gap-6 items-start">

            {/* Col 1 — step nav + description */}
            <div className="space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Pipeline steps</div>

              {OCR_STEPS.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setStep(s.id)}
                  className="w-full text-left rounded-lg px-3 py-2 text-xs font-medium transition-all"
                  style={{
                    background: step === s.id ? '#2563eb' : '#f0f4ff',
                    color: step === s.id ? 'white' : '#2563eb',
                    border: `1px solid ${step === s.id ? '#2563eb' : '#dbeafe'}`,
                  }}
                >
                  <span className="opacity-60 mr-1">{s.id}.</span> {s.label}
                </button>
              ))}

              <div className="flex gap-1.5">
                <Button variant="outline" size="sm" className="flex-1" disabled={step === 0} onClick={() => setStep(s => s - 1)}>← Prev</Button>
                <Button size="sm" className="flex-1" disabled={step === OCR_STEPS.length - 1} onClick={() => setStep(s => s + 1)}>Next →</Button>
              </div>

              <div className="border-t pt-3 space-y-3">
                <SectionLabel color="#2563eb">Step {step} of {OCR_STEPS.length - 1}</SectionLabel>
                <h3 className="text-sm font-bold">{current.label}</h3>
                {current.sublabel && <p className="text-xs text-muted-foreground">{current.sublabel}</p>}
                <p className="text-xs opacity-80 leading-relaxed">{current.desc}</p>

                {step === 0 && (
                  <InfoBox color="#2563eb" title="Why this document?">
                    A real report from 财信证券 (Chasing Securities) — dense multi-column Chinese tables,
                    mixed scripts, no visible cell borders. The ideal stress-test for classical OCR.
                  </InfoBox>
                )}

                {step === 1 && (
                  <InfoBox color="#2563eb" title="Real detection output">
                    Each colored box is a text region the model found. It fires per line or cell —
                    <strong> locations only, no structure yet.</strong>
                  </InfoBox>
                )}

                {step === 2 && (
                  <InfoBox color="#d97706" title="Crop → de-skew → normalize">
                    Detection polygons are rotated. Each region must be cropped, rectified to a flat
                    horizontal strip (via TPS or affine transform), then resized to a fixed recognizer input.
                    <strong> All page context is discarded at this point.</strong>
                  </InfoBox>
                )}

                {step === 3 && (
                  <div className="rounded-xl border p-3" style={{ background: '#fef3c7' }}>
                    <div className="font-semibold text-amber-700 mb-1.5 text-xs">CTC decoding</div>
                    <div className="font-mono text-xs space-y-1 text-amber-800">
                      <div>cols: 利利-润润-表表</div>
                      <div>collapse: 利 润 表</div>
                      <div>→ "利润表"</div>
                    </div>
                  </div>
                )}

                {step === 4 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Original structure</div>
                    <div className="rounded-xl border overflow-x-auto">
                      <table className="text-xs w-full min-w-max">
                        <thead>
                          <tr className="bg-slate-100">
                            {NEMO_STRUCTURED[0].map((h, i) => (
                              <th key={i} className="px-2 py-1 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {NEMO_STRUCTURED.slice(1).map((row, i) => (
                            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                              {row.map((cell, j) => (
                                <td key={j} className="px-2 py-1 border-b whitespace-nowrap">{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="text-xs font-semibold text-red-600 uppercase tracking-wider">OCR output — structure lost</div>
                    <pre className="rounded-xl p-2 text-xs font-mono overflow-auto"
                      style={{ background: '#0f172a', color: '#fca5a5', maxHeight: 140, whiteSpace: 'pre-wrap' }}>
                      {NEMO_FLAT_OUTPUT}
                    </pre>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      Characters correct, grid gone.
                      <strong className="text-red-600"> This is what VLMs solve.</strong>
                    </p>
                  </div>
                )}

                {step < 4 && (
                  <div className="rounded-xl border p-3">
                    <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Full pipeline</div>
                    <div className="flex flex-wrap gap-1 text-xs">
                      {['Image', '→', 'Detector', '→', 'Crop', '→', 'CNN', '→', 'BiLSTM', '→', 'CTC', '→', 'Text'].map((s, i) => (
                        <span key={i} className={s === '→' ? 'text-muted-foreground' : 'rounded px-1.5 py-0.5 font-mono font-semibold'}
                          style={s !== '→' ? { background: '#2563eb15', color: '#2563eb' } : {}}>
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Col 2 — document image, OR the crop pipeline at step 2 */}
            <div>
              {step === 2 ? <CropPipeline /> : <OcrDocReal step={step} />}
            </div>

          </div>
        </CardContent>
      </Card>

      {/* Limitations */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Limitations of the two-stage approach</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          {/* Text-only limitation cards */}
          <div className="grid md:grid-cols-2 gap-3">
            {OCR_LIMITATIONS.map(l => (
              <div key={l.title} className="rounded-xl border-l-4 p-4" style={{ borderColor: l.color, background: l.color + '08' }}>
                <div className="font-semibold text-sm mb-1" style={{ color: l.color }}>{l.title}</div>
                <p className="text-xs opacity-80 leading-relaxed">{l.desc}</p>
              </div>
            ))}
          </div>

          {/* Image-backed examples — click to reveal */}
          {(() => {
            const [openExample, setOpenExample] = useState(null);
            const examples = [
              {
                id: 'pizza',
                title: 'Rotated & perspective text',
                src: IMG_PIZZA,
                alt: 'Restaurant menu photographed at an angle',
                imgStyle: { objectFit: 'cover', objectPosition: 'top center' },
                desc: 'This menu is photographed at an angle — the page recedes with perspective distortion. A naive reading-order algorithm sorts boxes by their top-left coordinate, so columns interleave: a heading from the left column, then a price from the right, then back. TPS rectification helps only if detection found a clean polygon to warp.',
              },
              {
                id: 'mixed',
                title: 'Multi-orientation text',
                src: IMG_MIXED,
                alt: 'Supplement label with text in four different orientations',
                imgStyle: { objectFit: 'contain' },
                desc: 'This Nutrilite supplement label has text at 0°, 90°, 180°, and 270° — all on the same image. Classical OCR expects one dominant orientation per page. Each rotated block must be detected, classified, individually rectified, then recognized — and results reassembled into coherent reading order. Errors at any step cascade.',
              },
            ];
            return (
              <div className="grid md:grid-cols-2 gap-4">
                {examples.map(ex => {
                  const isOpen = openExample === ex.id;
                  return (
                    <div key={ex.id} className="rounded-2xl border overflow-hidden" style={{ borderColor: '#dc262655' }}>
                      {/* Clickable header */}
                      <button
                        onClick={() => setOpenExample(isOpen ? null : ex.id)}
                        className="w-full text-left p-4 flex items-center justify-between transition-colors"
                        style={{ background: isOpen ? '#dc262612' : 'transparent' }}
                      >
                        <div className="space-y-1">
                          <div className="font-semibold text-sm" style={{ color: '#dc2626' }}>{ex.title}</div>
                          <p className="text-xs opacity-75 leading-relaxed">{ex.desc}</p>
                        </div>
                        <span className="ml-3 flex-shrink-0 text-lg font-light" style={{ color: '#dc2626' }}>
                          {isOpen ? '−' : '+ see example'}
                        </span>
                      </button>

                      {/* Expandable image */}
                      {isOpen && (
                        <div className="border-t overflow-hidden bg-white" style={{ borderColor: '#dc262633' }}>
                          <img
                            src={ex.src}
                            alt={ex.alt}
                            style={{ width: '100%', display: 'block', maxHeight: 340, ...ex.imgStyle }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })()}

          <InfoBox color="#16a34a" title="What still works great">
            For clean, single-column documents with standard fonts (receipts, passports, license plates),
            classical OCR remains fast, cheap, and highly accurate. Tesseract 5 achieves &gt;99%
            character accuracy on clean Latin-script documents. PaddleOCR extends this to 80+ languages.
          </InfoBox>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 3 — Vision-Language Models
// ═══════════════════════════════════════════════════════════════════════════════

const VLM_MODELS = [
  { name: 'Donut',         year: 2022, type: 'Encoder-Decoder',  specialty: 'Document understanding without OCR',        color: '#7c3aed', url: 'https://github.com/clovaai/donut' },
  { name: 'Nougat',        year: 2023, type: 'Encoder-Decoder',  specialty: 'Academic PDF → LaTeX/Markdown',              color: '#7c3aed', url: 'https://github.com/facebookresearch/nougat' },
  { name: 'GOT-OCR 2.0',   year: 2024, type: 'Encoder-Decoder',  specialty: 'High-res OCR, formulas, tables',             color: '#2563eb', url: 'https://huggingface.co/stepfun-ai/GOT-OCR2_0' },
  { name: 'Qwen2-VL',      year: 2024, type: 'Decoder-only VLM', specialty: 'Dynamic resolution, instruction following', color: '#d97706', url: 'https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct' },
  { name: 'InternVL2',     year: 2024, type: 'Decoder-only VLM', specialty: 'Strong at dense text & charts',              color: '#d97706', url: 'https://huggingface.co/OpenGVLab/InternVL2-8B' },
  { name: 'Phi-3-Vision',  year: 2024, type: 'Decoder-only VLM', specialty: 'Efficient on-device document QA',            color: '#16a34a', url: 'https://huggingface.co/microsoft/Phi-3-vision-128k-instruct' },
  { name: 'Gemini 2.0',    year: 2025, type: 'Frontier VLM',     specialty: 'Multi-modal, multi-page, grounding',         color: '#dc2626', url: 'https://deepmind.google/technologies/gemini/' },
  { name: 'GPT-4o',        year: 2024, type: 'Frontier VLM',     specialty: 'General document + chart + form tasks',      color: '#dc2626', url: 'https://openai.com/index/hello-gpt-4o/' },
  { name: 'NuExtract 3',   year: 2025, type: 'Doc-extraction VLM', specialty: 'Schema-prompted structured extraction (4B)', color: '#0d9488', url: 'https://huggingface.co/numind/NuExtract3' },
];

// Images are TRUE SQUARES so math is exact.
// ViT patch size = 16×16 px always.
// Focus patch anchored at col=3, row=3 in 224px image (pixel 48,48).
// Scales exactly: ×2 → pixel 96,96 in 448px; ×4 → pixel 192,192 in 896px.
// Same content percentage (21.4%, 21.4%) across all three images.
const PATCH_SIZE = 16;
const FOCUS_COL = 3; // patch column index in 224px grid (0-indexed)
const FOCUS_ROW = 3; // patch row index

const PATCH_CONFIGS = [
  {
    label: '224 px',
    docSrc:   `${BASE}resources/nemo_224.jpg`,   // 224×224 square document
    patchSrc: `${BASE}resources/patch_224.png`,  // 16×16 crop → 200×200 (NEAREST, blocky)
    size: 224, gridCols: 14, tokens: 196, factor: 1,
    patchLabel: '1×1 patch — 16×16 px source',
    note: 'ONE patch covers this area. The model receives just 256 pixels — text is an unrecognisable smear.',
  },
  {
    label: '448 px',
    docSrc:   `${BASE}resources/nemo_448.jpg`,
    patchSrc: `${BASE}resources/patch_448.png`,  // 32×32 crop → 200×200 (NEAREST, less blocky)
    size: 448, gridCols: 28, tokens: 784, factor: 2,
    patchLabel: '2×2 patches — 32×32 px source',
    note: 'Same content now spans 4 patches (32×32 px). Characters are starting to form but still blurry.',
  },
  {
    label: '896 px',
    docSrc:   `${BASE}resources/nemo_896.jpg`,
    patchSrc: `${BASE}resources/patch_896.png`,  // 64×64 crop → 200×200 (BILINEAR, sharp)
    size: 896, gridCols: 56, tokens: 3136, factor: 4,
    patchLabel: '4×4 patches — 64×64 px source',
    note: '16 patches cover this area (64×64 px). Numbers and Chinese characters are clearly legible.',
  },
];

function ImagePatchExplorer({ config }) {
  const { docSrc, patchSrc, size, gridCols, tokens, factor, note, label, patchLabel } = config;

  const DISPLAY = 360;
  const cellPx  = DISPLAY / gridCols;
  // Focus pixel in source image scales with factor: 48 → 96 → 192
  // Divided by `size` (224 → 448 → 896) this is always 21.4% — same screen position.
  const focusPct  = (FOCUS_COL * PATCH_SIZE * factor) / size;
  // Focus region is `factor` patches wide at source pixel scale.
  // Divided by `size` this is always 7.1% (1×16/224), 14.3% (2×16/448 = 32/448), 28.6% (4×16/896 = 64/896)
  // Wait — same fraction issue: PATCH_SIZE * factor / size = 16 * factor / (224 * factor) = 16/224 = 7.1% ALWAYS.
  // So box stays SAME screen size across resolutions. That's intentional — what changes is patches *inside* it.
  const boxOffset = focusPct * DISPLAY;
  const boxSize   = (PATCH_SIZE * factor / size) * DISPLAY;

  const verdict = factor === 1
    ? { text: '❌ 16×16 px — text unreadable',  color: '#dc2626' }
    : factor === 2
    ? { text: '⚠️ 32×32 px — barely legible',   color: '#d97706' }
    : { text: '✓ 64×64 px — clearly legible',   color: '#16a34a' };

  return (
    <div className="grid md:grid-cols-2 gap-6 items-start">
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {label} — {gridCols}×{gridCols} grid — {tokens.toLocaleString()} tokens
        </div>
        <div style={{
          position: 'relative', width: DISPLAY, height: DISPLAY,
          border: '1px solid #e2e8f0', borderRadius: 8, overflow: 'hidden',
        }}>
          <img src={docSrc} style={{ width: DISPLAY, height: DISPLAY, display: 'block' }} />
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            backgroundImage: [
              'linear-gradient(to right, rgba(124,58,237,0.4) 1px, transparent 1px)',
              'linear-gradient(to bottom, rgba(124,58,237,0.4) 1px, transparent 1px)',
            ].join(','),
            backgroundSize: `${cellPx}px ${cellPx}px`,
          }} />
          <div style={{
            position: 'absolute',
            left: boxOffset, top: boxOffset,
            width: boxSize, height: boxSize,
            border: '3px solid #ef4444',
            background: 'rgba(239,68,68,0.2)',
            pointerEvents: 'none',
          }}>
            <div style={{
              position: 'absolute', bottom: '100%', left: 0,
              background: '#ef4444', color: 'white',
              fontSize: 8, fontWeight: 800, padding: '1px 5px',
              borderRadius: '3px 3px 0 0', whiteSpace: 'nowrap',
            }}>
              {factor}×{factor} patch{factor > 1 ? 'es' : ''}
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Content inside red box — {patchLabel}
        </div>
        <div style={{ border: '3px solid #ef4444', borderRadius: 8, overflow: 'hidden', width: 200, height: 200 }}>
          <img
            src={patchSrc}
            alt={patchLabel}
            style={{ width: 200, height: 200, display: 'block', imageRendering: factor <= 2 ? 'pixelated' : 'auto' }}
          />
        </div>
        <div className="rounded-xl border p-3 space-y-1.5">
          <div className="font-bold text-sm" style={{ color: verdict.color }}>{verdict.text}</div>
          <p className="text-xs opacity-75 leading-relaxed">{note}</p>
        </div>
      </div>
    </div>
  );
}

// ─── Chapter 3 — visual token compression + schema-prompted decoder ────────

const EXTRACTION_MODES = [
  {
    id: 'invoice',
    label: 'Invoice schema',
    color: '#16a34a',
    desc: 'NuExtract-style JSON template — leaf values declare types.',
    promptTokens: 120,
    prompt: `{
  "invoice_number": "verbatim-string",
  "invoice_date": "date",
  "total_amount": "number",
  "currency": "currency",
  "line_items": [{
    "description": "verbatim-string",
    "quantity": "integer",
    "unit_price": "number"
  }]
}`,
    outputTokens: 220,
    output: `{
  "invoice_number": "INV-2024-0391",
  "invoice_date": "2024-10-01",
  "total_amount": 4536.00,
  "currency": "USD",
  "line_items": [
    {
      "description": "Consulting services (Oct)",
      "quantity": 20,
      "unit_price": 150.00
    },
    {
      "description": "Software license (annual)",
      "quantity": 1,
      "unit_price": 1200.00
    }
  ]
}`,
  },
  {
    id: 'markdown',
    label: 'Doc → Markdown',
    color: '#2563eb',
    desc: 'No schema. Just convert the whole document to clean Markdown / HTML tables.',
    promptTokens: 15,
    prompt: 'Convert the document to clean Markdown. Preserve tables as HTML and equations as LaTeX.',
    outputTokens: 820,
    output: `# Quarterly Report — Q3 2024

Revenue grew 18% year-over-year, driven by
enterprise adoption in North America and APAC.

## Sales by Product

<table>
  <thead>
    <tr>
      <th>Product</th>
      <th>Q3 Sales</th>
      <th>Q4 Forecast</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Widget A</td><td>1,240</td><td>1,890</td></tr>
    <tr><td>Widget B</td><td>870</td><td>1,100</td></tr>
    <tr><td>Widget C</td><td>2,300</td><td>2,750</td></tr>
  </tbody>
</table>

Operating margin improved to **24.3%**, up
from 21.1% in the prior year. ...`,
  },
  {
    id: 'qa',
    label: 'Free-form Q&A',
    color: '#d97706',
    desc: 'Ask a single question. Get a short grounded answer.',
    promptTokens: 11,
    prompt: 'What is the total amount due on this invoice?',
    outputTokens: 8,
    output: `$4,536.00`,
  },
  {
    id: 'multi',
    label: 'Multi-schema (NuExtract feature)',
    color: '#7c3aed',
    desc: 'Combine extraction + summarisation + risk flags in one schema. One forward pass, multiple structured outputs.',
    promptTokens: 195,
    prompt: `{
  "extraction": {
    "invoice_number": "verbatim-string",
    "total_amount": "number",
    "due_date": "date"
  },
  "summary": "string",
  "payment_terms": "string",
  "risk_flags": [["overdue", "missing_signature",
                   "unusual_amount", "vendor_mismatch"]],
  "language": "country"
}`,
    outputTokens: 280,
    output: `{
  "extraction": {
    "invoice_number": "INV-2024-0391",
    "total_amount": 4536.00,
    "due_date": "2024-10-31"
  },
  "summary": "Standard consulting + software license invoice from Acme Corp, due net-30, no irregularities detected.",
  "payment_terms": "Net 30",
  "risk_flags": [],
  "language": "US"
}`,
  },
];

function TokenPipelineDiagram({ mode }) {
  const VISUAL_TOKENS_RAW = 3136;
  const VISUAL_TOKENS_COMPRESSED = 784;
  const inputToDecoder = VISUAL_TOKENS_COMPRESSED + mode.promptTokens;

  return (
    <svg viewBox="0 0 760 220" className="w-full" style={{ minWidth: 680 }}>
      {/* Stage 1: input image */}
      <g>
        <rect x={8} y={70} width={76} height={76} rx={6} fill="#e2e8f0" stroke="#64748b" strokeWidth={1.2} />
        <text x={46} y={102} textAnchor="middle" fontSize={9} fontWeight={700} fill="#475569">896×896</text>
        <text x={46} y={116} textAnchor="middle" fontSize={9} fill="#475569" opacity={0.75}>image</text>
        <text x={46} y={158} textAnchor="middle" fontSize={8} fill="#64748b" opacity={0.7}>input pixels</text>
      </g>

      {/* Arrow 1: ViT patchify */}
      <line x1={88} y1={108} x2={130} y2={108} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#tp-arr)" />
      <text x={109} y={102} textAnchor="middle" fontSize={8} fill="#94a3b8">ViT-B/16</text>
      <text x={109} y={116} textAnchor="middle" fontSize={8} fill="#94a3b8">56×56 patches</text>

      {/* Stage 2: 3136 visual tokens */}
      <g>
        <rect x={134} y={62} width={120} height={92} rx={6} fill="#7c3aed12" stroke="#7c3aed" strokeWidth={1.4} />
        <text x={194} y={82} textAnchor="middle" fontSize={11} fontWeight={700} fill="#7c3aed">3,136</text>
        <text x={194} y={96} textAnchor="middle" fontSize={9} fill="#7c3aed" opacity={0.85}>visual tokens</text>
        {/* Mini token grid */}
        {Array.from({ length: 6 }).flatMap((_, r) =>
          Array.from({ length: 16 }).map((_, c) => (
            <rect key={`r${r}c${c}`} x={140 + c * 6.6} y={108 + r * 6} width={5.5} height={5} rx={1} fill="#7c3aed" opacity={0.4 + (r + c) * 0.005} />
          ))
        )}
      </g>

      {/* Arrow 2: 2×2 compression */}
      <line x1={258} y1={108} x2={302} y2={108} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#tp-arr)" />
      <text x={280} y={102} textAnchor="middle" fontSize={8} fill="#94a3b8" fontWeight={700}>2×2 compress</text>
      <text x={280} y={116} textAnchor="middle" fontSize={8} fill="#94a3b8" opacity={0.8}>(pixel shuffle)</text>

      {/* Stage 3: 784 compressed tokens */}
      <g>
        <rect x={306} y={62} width={120} height={92} rx={6} fill="#0d948812" stroke="#0d9488" strokeWidth={1.4} />
        <text x={366} y={82} textAnchor="middle" fontSize={11} fontWeight={700} fill="#0d9488">784</text>
        <text x={366} y={96} textAnchor="middle" fontSize={9} fill="#0d9488" opacity={0.85}>compressed tokens</text>
        {/* Smaller token grid */}
        {Array.from({ length: 4 }).flatMap((_, r) =>
          Array.from({ length: 8 }).map((_, c) => (
            <rect key={`cr${r}c${c}`} x={316 + c * 13} y={108 + r * 10} width={11} height={9} rx={1} fill="#0d9488" opacity={0.5 + (r + c) * 0.02} />
          ))
        )}
      </g>

      {/* Schema prompt (lower path) */}
      <g>
        <rect x={306} y={172} width={120} height={36} rx={6} fill={mode.color + '12'} stroke={mode.color} strokeWidth={1.4} />
        <text x={366} y={188} textAnchor="middle" fontSize={10} fontWeight={700} fill={mode.color}>
          {mode.promptTokens} tokens
        </text>
        <text x={366} y={200} textAnchor="middle" fontSize={8} fill={mode.color} opacity={0.85}>
          {mode.label}
        </text>
      </g>

      {/* Combine arrows into decoder */}
      <line x1={430} y1={108} x2={486} y2={140} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#tp-arr)" />
      <line x1={430} y1={190} x2={486} y2={158} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#tp-arr)" />

      {/* Combined input label */}
      <text x={460} y={108} fontSize={8} fill="#64748b">  {VISUAL_TOKENS_COMPRESSED} visual</text>
      <text x={460} y={196} fontSize={8} fill="#64748b">  +{mode.promptTokens} prompt</text>

      {/* Stage 4: decoder */}
      <g>
        <rect x={490} y={108} width={140} height={70} rx={6} fill="#16a34a15" stroke="#16a34a" strokeWidth={1.4} />
        <text x={560} y={132} textAnchor="middle" fontSize={11} fontWeight={700} fill="#16a34a">LLM decoder</text>
        <text x={560} y={146} textAnchor="middle" fontSize={9} fill="#16a34a" opacity={0.8}>
          input = {inputToDecoder} tokens
        </text>
        <text x={560} y={159} textAnchor="middle" fontSize={8} fill="#16a34a" opacity={0.7}>
          (NuExtract3 4B · Qwen3.5-style)
        </text>
      </g>

      {/* Arrow to output */}
      <line x1={630} y1={143} x2={668} y2={143} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#tp-arr)" />

      {/* Stage 5: output */}
      <g>
        <rect x={672} y={114} width={80} height={56} rx={6} fill={mode.color + '15'} stroke={mode.color} strokeWidth={1.4} />
        <text x={712} y={137} textAnchor="middle" fontSize={11} fontWeight={700} fill={mode.color}>
          {mode.outputTokens}
        </text>
        <text x={712} y={150} textAnchor="middle" fontSize={9} fill={mode.color} opacity={0.85}>output</text>
        <text x={712} y={161} textAnchor="middle" fontSize={9} fill={mode.color} opacity={0.85}>tokens</text>
      </g>

      <defs>
        <marker id="tp-arr" markerWidth="6" markerHeight="6" refX={5} refY={3} orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  );
}

function VlmPage() {
  const [patchIdx, setPatchIdx] = useState(1);
  const config = PATCH_CONFIGS[patchIdx];
  const [extractMode, setExtractMode] = useState('invoice');
  const mode = EXTRACTION_MODES.find(m => m.id === extractMode);

  return (
    <div className="space-y-6">
      {/* Architecture */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">End-to-end architecture</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div className="overflow-x-auto">
            <svg viewBox="0 0 700 160" className="w-full" style={{ minWidth: 560 }}>
              {/* Document image box */}
              <rect x="10" y="50" width="100" height="60" rx="8" fill="#f0f4ff" stroke="#2563eb" strokeWidth="1.5" />
              <text x="60" y="78" textAnchor="middle" fontSize="10" fill="#2563eb" fontWeight="600">Document</text>
              <text x="60" y="93" textAnchor="middle" fontSize="9" fill="#2563eb" opacity="0.7">Image</text>

              {/* Arrow */}
              <line x1="110" y1="80" x2="145" y2="80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#vlmarr)" />
              <text x="127" y="73" textAnchor="middle" fontSize="8" fill="#94a3b8">patch</text>

              {/* ViT patch encoder */}
              <rect x="145" y="30" width="130" height="100" rx="8" fill="#ede9fe" stroke="#7c3aed" strokeWidth="1.5" />
              <text x="210" y="55" textAnchor="middle" fontSize="10" fill="#7c3aed" fontWeight="600">Vision Encoder</text>
              <text x="210" y="70" textAnchor="middle" fontSize="8.5" fill="#7c3aed" opacity="0.8">(ViT / SigLIP / CLIP)</text>
              {[85, 100, 115].map((y, i) => (
                <rect key={i} x="155" y={y} width="110" height="8" rx="2" fill="#7c3aed33" />
              ))}
              <text x="210" y="140" textAnchor="middle" fontSize="8" fill="#7c3aed" opacity="0.7">patch embeddings</text>

              {/* Arrow */}
              <line x1="275" y1="80" x2="310" y2="80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#vlmarr)" />
              <text x="292" y="73" textAnchor="middle" fontSize="8" fill="#94a3b8">tokens</text>

              {/* Projector / connector */}
              <rect x="310" y="55" width="80" height="50" rx="8" fill="#fef3c7" stroke="#d97706" strokeWidth="1.5" />
              <text x="350" y="78" textAnchor="middle" fontSize="9.5" fill="#d97706" fontWeight="600">Projector</text>
              <text x="350" y="93" textAnchor="middle" fontSize="8" fill="#d97706" opacity="0.7">MLP / cross-attn</text>

              {/* Arrow */}
              <line x1="390" y1="80" x2="425" y2="80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#vlmarr)" />
              <text x="407" y="73" textAnchor="middle" fontSize="8" fill="#94a3b8">fused</text>

              {/* LLM */}
              <rect x="425" y="20" width="150" height="120" rx="8" fill="#f0fdf4" stroke="#16a34a" strokeWidth="1.5" />
              <text x="500" y="45" textAnchor="middle" fontSize="10" fill="#16a34a" fontWeight="600">Language Model</text>
              <text x="500" y="60" textAnchor="middle" fontSize="8.5" fill="#16a34a" opacity="0.8">(Qwen2 / Phi / Llama)</text>
              {[70, 85, 100, 115].map((y, i) => (
                <rect key={i} x="435" y={y} width="130" height="8" rx="2" fill="#16a34a22" />
              ))}

              {/* Arrow */}
              <line x1="575" y1="80" x2="610" y2="80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#vlmarr)" />

              {/* Output */}
              <rect x="610" y="50" width="80" height="60" rx="8" fill="#fff7ed" stroke="#ea580c" strokeWidth="1.5" />
              <text x="650" y="73" textAnchor="middle" fontSize="9.5" fill="#ea580c" fontWeight="600">Output</text>
              <text x="650" y="87" textAnchor="middle" fontSize="8" fill="#ea580c" opacity="0.7">text / JSON</text>
              <text x="650" y="100" textAnchor="middle" fontSize="8" fill="#ea580c" opacity="0.7">LaTeX / CSV</text>

              <defs>
                <marker id="vlmarr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
                </marker>
              </defs>
            </svg>
          </div>

          <div className="mt-4 grid md:grid-cols-3 gap-3">
            <InfoBox color="#7c3aed" title="Vision Encoder">
              A ViT splits the image into fixed-size patches (e.g. 14×14 or 16×16 pixels), adds position embeddings, and produces one vector per patch. SigLIP and CLIP variants are pre-trained on image-text pairs for richer visual semantics.
            </InfoBox>
            <InfoBox color="#d97706" title="Projector">
              A lightweight MLP or cross-attention layer adapts the vision encoder's embedding dimension to the LLM's hidden size. This is where the modalities are "stitched" together — and where most fine-tuning happens for doc tasks.
            </InfoBox>
            <InfoBox color="#16a34a" title="Language Model">
              A causal decoder-only LLM (or encoder-decoder like T5/Bart in older VLMs) generates the output token by token, conditioned on both the visual tokens and any text prompt. This enables instruction following: "Extract the table as JSON."
            </InfoBox>
          </div>
        </CardContent>
      </Card>

      {/* Patch resolution explorer */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">The resolution–token count problem</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div className="space-y-4">
            <p className="text-sm opacity-80">
              A ViT carves the image into a fixed-size patch grid (typically 16×16 px each).
              Higher resolution = more patches = more tokens. Select a resolution to see
              the real document split up — and zoom in to see what a single patch actually contains.
            </p>
            {/* Resolution picker */}
            <div className="flex flex-wrap gap-2">
              {PATCH_CONFIGS.map((c, i) => (
                <button
                  key={i}
                  onClick={() => setPatchIdx(i)}
                  className="rounded-lg border px-4 py-2 text-sm transition-all"
                  style={{
                    borderColor: patchIdx === i ? '#7c3aed' : '#e5e5e5',
                    background: patchIdx === i ? '#ede9fe' : 'white',
                    fontWeight: patchIdx === i ? 700 : 400,
                    color: patchIdx === i ? '#7c3aed' : 'inherit',
                  }}
                >
                  {c.label}
                </button>
              ))}
            </div>
            {/* Image patch explorer */}
            <ImagePatchExplorer config={config} />
          </div>
        </CardContent>
      </Card>

      {/* Full token pipeline + schema-prompted extraction */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">The full token pipeline — visual tokens, compression, prompt, output</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">
          <p className="text-sm opacity-80 leading-relaxed">
            ViTs produce thousands of visual tokens per page. Feeding all of them to the LLM is expensive — so modern
            doc VLMs add a <strong>compression module</strong> (2×2 pixel-shuffle / pooling) between the encoder and decoder.
            The decoder then receives those compressed tokens <em>plus a text prompt</em> that defines what to extract.
            Pick a schema below to see how the prompt shapes the output.
          </p>

          {/* Schema selector tabs */}
          <div className="flex flex-wrap gap-2">
            {EXTRACTION_MODES.map(m => (
              <button
                key={m.id}
                onClick={() => setExtractMode(m.id)}
                className="rounded-xl px-3 py-2 text-xs font-semibold transition-all"
                style={{
                  background: extractMode === m.id ? m.color : 'transparent',
                  color: extractMode === m.id ? 'white' : m.color,
                  border: `1.5px solid ${extractMode === m.id ? m.color : m.color + '55'}`,
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Pipeline diagram */}
          <div className="rounded-xl border p-4 overflow-x-auto">
            <TokenPipelineDiagram mode={mode} />
          </div>

          {/* Mode description + side-by-side prompt + output */}
          <div className="text-sm opacity-85 leading-relaxed">{mode.desc}</div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                Prompt <span className="font-mono opacity-70">({mode.promptTokens} tokens)</span>
              </div>
              <pre className="rounded-xl p-3 text-xs font-mono leading-relaxed overflow-x-auto"
                style={{ background: '#0f172a', color: '#e2e8f0', whiteSpace: 'pre-wrap', maxHeight: 280 }}
              >
                {mode.prompt}
              </pre>
            </div>
            <div className="space-y-1.5">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                Output <span className="font-mono opacity-70">({mode.outputTokens} tokens)</span>
              </div>
              <pre className="rounded-xl p-3 text-xs font-mono leading-relaxed overflow-x-auto"
                style={{ background: '#0f172a', color: '#86efac', whiteSpace: 'pre-wrap', maxHeight: 280 }}
              >
                {mode.output}
              </pre>
            </div>
          </div>

          {extractMode === 'multi' && (
            <InfoBox color="#7c3aed" title="Multi-schema in one pass — the NuExtract 3 trick">
              <span className="text-xs">
                NuExtract 3 lets you compose <strong>multiple</strong> extraction targets into a single template:
                structured field extraction, free-form summary, enum-based risk flags, and language detection,
                all in one forward pass. The model emits a single JSON that satisfies the whole template.
                For downstream pipelines this collapses 3–4 separate model calls into one.
              </span>
            </InfoBox>
          )}

          <InfoBox color="#0d9488" title="Why compression matters">
            <span className="text-xs">
              Without the 2×2 compression step, 3,136 visual tokens would dominate the decoder's context.
              With it, only 784 visual tokens reach the LLM — leaving plenty of context budget for the schema
              prompt and the output. This is what makes long, prompt-heavy extraction feasible on small models
              like the 0.9B PaddleOCR-VL and 4B NuExtract 3.
            </span>
          </InfoBox>
        </CardContent>
      </Card>

      {/* Model family */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">The model family</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {VLM_MODELS.map(m => (
              <div key={m.name} className="rounded-xl border p-4 space-y-1" style={{ borderColor: m.color + '55' }}>
                <a href={m.url} target="_blank" rel="noopener noreferrer"
                  className="font-bold text-sm hover:underline inline-flex items-center gap-1"
                  style={{ color: m.color }}
                >
                  {m.name} <span className="text-xs">↗</span>
                </a>
                <div className="text-xs text-muted-foreground">{m.year} · {m.type}</div>
                <div className="text-xs opacity-80 mt-1">{m.specialty}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTERS 4–7 — placeholders, filled in below
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Chapter 4 data ──────────────────────────────────────────────────────────

const FAILURE_MODES = [
  {
    id: 'hallucination',
    label: 'Hallucinated text',
    color: '#dc2626',
    title: 'The model invents content',
    desc: 'When a region is hard to read — low contrast, occlusion, unusual fonts, photographic noise — the model fills in plausible content rather than expressing uncertainty. The output looks confident and well-formed but is silently wrong.',
    fix: 'Multi-model labeling + cross-validation filtering (PaddleOCR-VL discards labels where multiple labelers disagree)',
    fixChapter: 5,
    fixColor: '#2563eb',
  },
  {
    id: 'unparseable',
    label: 'Unparseable outputs',
    color: '#d97706',
    title: 'Arbitrary text generators emit broken structure',
    desc: 'A VLM is at heart a next-token predictor. When asked for JSON, LaTeX, HTML, or markdown tables, nothing guarantees the output is syntactically valid — it can prepend stray tokens, wrap output in code fences, mismatch braces, or produce rows that do not align.',
    fix: 'RL with verifiable rewards (olmOCR-2 GRPO) directly penalizes outputs that fail programmatic checks: does the JSON parse? does the LaTeX compile? does the table have the right shape?',
    fixChapter: 5,
    fixColor: '#2563eb',
  },
  {
    id: 'repetition',
    label: 'Repetition loops',
    color: '#7c3aed',
    title: 'Autoregressive decoders get stuck',
    desc: 'Token-by-token generation can collapse into a repeating loop on long or out-of-domain documents. Once the model is in the loop, every subsequent token reinforces it. Nougat ships with a built-in repetition detector. olmOCR has GitHub issues filed for "page processing loops endlessly".',
    fix: 'Diffusion decoding (MinerU-Diffusion, PA-BDM) replaces left-to-right generation with parallel masked denoising — no single token can lock the model into a degenerate trajectory',
    fixChapter: 6,
    fixColor: '#16a34a',
  },
  {
    id: 'latency',
    label: 'Slow & memory hungry',
    color: '#0d9488',
    title: 'A page can take a minute',
    desc: 'Autoregressive VLMs need one forward pass per output token. A dense page produces 3–5K tokens. At realistic throughput (MinerU 2.5 = 52 tokens/sec, PaddleOCR-VL = 40 tokens/sec) one page can take 60–125 seconds. KV cache grows with sequence length. olmOCR-2 caps output at 8K tokens per page just to stay tractable.',
    fix: 'VRFM coarse-to-fine architecture (process only valid regions, ~39% of the page) and diffusion decoding (parallel token generation, 3–5× speedup)',
    fixChapter: 6,
    fixColor: '#16a34a',
  },
];

// Real-example panels for each failure mode (placeholders user can swap with real screenshots)

function HallucinationExample() {
  return (
    <div className="rounded-xl border overflow-hidden bg-white">
      <div className="px-4 py-2 text-xs font-semibold uppercase tracking-wider" style={{ background: '#fef2f2', color: '#dc2626' }}>
        Real-world example — invoice extraction
      </div>
      <div className="p-5 space-y-3 font-mono text-xs">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <span className="text-muted-foreground">Vendor:</span>          <span>Acme Industries Ltd</span>
          <span className="text-muted-foreground">Invoice number:</span>  <span>INV-2024-0391</span>
          <span className="text-muted-foreground">Date:</span>             <span>2024-10-01</span>
          <span className="text-muted-foreground">Subtotal:</span>         <span>$4,200.00</span>
          <span className="text-muted-foreground">Tax (8%):</span>         <span>$336.00</span>
          <span className="text-muted-foreground">Total due:</span>
          <span>
            <span style={{ background: '#fee2e2', color: '#dc2626', padding: '0 4px', borderRadius: 3, fontWeight: 700 }}>
              $4,536.56
            </span>
            <span className="text-muted-foreground ml-2">← model output</span>
          </span>
        </div>
        <div className="border-t pt-3 mt-3 space-y-1">
          <div className="text-muted-foreground">Actual scan:</div>
          <div style={{ color: '#16a34a' }}>
            ✓ Total due: <strong>$4,536.00</strong>
            <span className="text-muted-foreground ml-2">(.56 was JPEG noise on the decimal)</span>
          </div>
        </div>
      </div>
      <div className="px-5 pb-4 text-xs text-muted-foreground italic">
        Placeholder — to be replaced with real production example.
      </div>
    </div>
  );
}

function UnparseableExample() {
  return (
    <div className="rounded-xl border overflow-hidden bg-white">
      <div className="px-4 py-2 text-xs font-semibold uppercase tracking-wider" style={{ background: '#fef3c7', color: '#92400e' }}>
        Real bug reports — VLM structured output failures
      </div>
      <div className="p-4 space-y-3">
        {/* Issue 1: extra prefix */}
        <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
          <div className="text-xs text-slate-400">
            <a href="https://github.com/vllm-project/vllm/issues/31858" target="_blank" rel="noopener noreferrer"
              className="hover:underline" style={{ color: '#fbbf24' }}
            >
              vllm-project/vllm #31858 ↗
            </a>
            <span className="ml-2">— "extra tokens appear before JSON"</span>
          </div>
          <div className="font-mono text-xs leading-relaxed">
            <span style={{ color: '#94a3b8' }}># Expected:</span><br/>
            <span style={{ color: '#86efac' }}>{`{"title": "..."}`}</span><br/>
            <span style={{ color: '#94a3b8' }}># Actual:</span><br/>
            <span style={{ color: '#fca5a5' }}>{`A{"title": "..."}`}</span>
            <span className="text-slate-500"> ← stray 'A' token</span>
          </div>
        </div>

        {/* Issue 2: markdown fence around JSON */}
        <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
          <div className="text-xs text-slate-400">
            <a href="https://github.com/vllm-project/vllm/issues/35700" target="_blank" rel="noopener noreferrer"
              className="hover:underline" style={{ color: '#fbbf24' }}
            >
              vllm-project/vllm #35700 ↗
            </a>
            <span className="ml-2">— "result wrapped with markdown code block"</span>
          </div>
          <div className="font-mono text-xs leading-relaxed">
            <span style={{ color: '#94a3b8' }}># Expected raw JSON, got:</span><br/>
            <span style={{ color: '#fca5a5' }}>{'```json'}</span><br/>
            <span style={{ color: '#fca5a5' }}>{`{"author": "..."}`}</span><br/>
            <span style={{ color: '#fca5a5' }}>{'```'}</span>
          </div>
        </div>

        {/* Issue 3: malformed table crashes parser */}
        <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
          <div className="text-xs text-slate-400">
            <a href="https://github.com/docling-project/docling/issues/2467" target="_blank" rel="noopener noreferrer"
              className="hover:underline" style={{ color: '#fbbf24' }}
            >
              docling-project/docling #2467 ↗
            </a>
            <span className="ml-2">— "malformed table tokens crash the parser"</span>
          </div>
          <div className="font-mono text-xs leading-relaxed">
            <span style={{ color: '#fca5a5' }}>IndexError: list index out of range</span><br/>
            <span style={{ color: '#94a3b8' }}>  at otsl_parse_texts → tokens[r_idx][c_idx]</span><br/>
            <span style={{ color: '#94a3b8' }}>  (VLM emitted unequal row lengths)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function RepetitionExample() {
  return (
    <div className="rounded-xl border overflow-hidden bg-white">
      <div className="px-4 py-2 text-xs font-semibold uppercase tracking-wider" style={{ background: '#f3e8ff', color: '#7c3aed' }}>
        Real example — Nougat / olmOCR repetition loop
      </div>
      <div className="p-4 space-y-3">
        <div className="rounded-lg p-3" style={{ background: '#0f172a' }}>
          <div className="text-xs text-slate-400 mb-2">
            <a href="https://github.com/facebookresearch/nougat/issues/11" target="_blank" rel="noopener noreferrer"
              className="hover:underline" style={{ color: '#a78bfa' }}
            >
              facebookresearch/nougat #11 ↗
            </a>
            <span className="ml-2">— built-in detection heuristic</span>
          </div>
          <div className="font-mono text-xs space-y-0.5">
            <div style={{ color: '#86efac' }}>The Hamiltonian is given by</div>
            <div style={{ color: '#86efac' }}>H = T + V, where T is the kinetic energy</div>
            <div style={{ color: '#86efac' }}>and V is the potential energy of the system.</div>
            <div style={{ color: '#fca5a5' }}>The system is the system is the system is</div>
            <div style={{ color: '#fca5a5' }}>the system is the system is the system is</div>
            <div style={{ color: '#fca5a5' }}>the system is the system is the system is</div>
            <div style={{ color: '#fca5a5' }}>the system is the system is the system is</div>
            <div style={{ color: '#fbbf24' }}>{'>>> WARNING:root:Found repetitions in sample 0'}</div>
            <div style={{ color: '#94a3b8' }}>{'>>> truncating output, marking page as failed'}</div>
          </div>
        </div>
        <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
          <div className="text-xs text-slate-400">
            <a href="https://github.com/allenai/olmocr/issues/110" target="_blank" rel="noopener noreferrer"
              className="hover:underline" style={{ color: '#a78bfa' }}
            >
              allenai/olmocr #110 ↗
            </a>
            <span className="ml-2">— "Page processing loops endlessly"</span>
          </div>
          <div className="font-mono text-xs" style={{ color: '#94a3b8' }}>
            "I left it running for two hours" — reporter
          </div>
        </div>
      </div>
    </div>
  );
}

function LatencyExample() {
  return (
    <div className="rounded-xl border overflow-hidden bg-white">
      <div className="px-4 py-2 text-xs font-semibold uppercase tracking-wider" style={{ background: '#ccfbf1', color: '#0d9488' }}>
        Real throughput numbers — autoregressive VLMs on doc parsing
      </div>
      <div className="p-4">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-xs uppercase tracking-wider text-muted-foreground">
              <th className="text-left py-2 pr-3 font-semibold">Model</th>
              <th className="text-right py-2 pr-3 font-semibold">Tokens/sec</th>
              <th className="text-right py-2 font-semibold">~Time per 5K-token page</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            <tr>
              <td className="py-2 pr-3 font-medium">PaddleOCR-VL (static decode, 32 steps)</td>
              <td className="py-2 pr-3 text-right font-mono">22 TPS</td>
              <td className="py-2 text-right font-mono" style={{ color: '#dc2626' }}>≈ 227 s</td>
            </tr>
            <tr>
              <td className="py-2 pr-3 font-medium">PaddleOCR-VL (default)</td>
              <td className="py-2 pr-3 text-right font-mono">40 TPS</td>
              <td className="py-2 text-right font-mono" style={{ color: '#dc2626' }}>≈ 125 s</td>
            </tr>
            <tr>
              <td className="py-2 pr-3 font-medium">MinerU 2.5 (AR baseline)</td>
              <td className="py-2 pr-3 text-right font-mono">52 TPS</td>
              <td className="py-2 text-right font-mono" style={{ color: '#dc2626' }}>≈ 96 s</td>
            </tr>
            <tr>
              <td className="py-2 pr-3 font-medium text-muted-foreground">— for reference —</td>
              <td className="py-2 pr-3 text-right font-mono text-muted-foreground">—</td>
              <td className="py-2 text-right font-mono text-muted-foreground">—</td>
            </tr>
            <tr>
              <td className="py-2 pr-3 font-medium" style={{ color: '#16a34a' }}>MinerU-Diffusion (τ=0.6, ch6)</td>
              <td className="py-2 pr-3 text-right font-mono" style={{ color: '#16a34a' }}>165 TPS</td>
              <td className="py-2 text-right font-mono" style={{ color: '#16a34a' }}>≈ 30 s</td>
            </tr>
            <tr>
              <td className="py-2 pr-3 font-medium" style={{ color: '#16a34a' }}>PA-BDM (block diffusion, ch6)</td>
              <td className="py-2 pr-3 text-right font-mono" style={{ color: '#16a34a' }}>267 TPS</td>
              <td className="py-2 text-right font-mono" style={{ color: '#16a34a' }}>≈ 19 s</td>
            </tr>
          </tbody>
        </table>
        <div className="mt-4 rounded-lg p-3 text-xs" style={{ background: '#0f172a', color: '#e2e8f0' }}>
          <div className="space-y-1 font-mono">
            <div><span className="text-slate-400"># olmOCR-2-7B token budget (per page):</span></div>
            <div>~1,000 tokens encode the page image</div>
            <div>~1,800 tokens for document anchoring</div>
            <div>~200 tokens for the prompt</div>
            <div style={{ color: '#fbbf24' }}>→ only ~5,000 tokens left for actual text output (8K cap)</div>
          </div>
        </div>
        <div className="mt-3 text-xs text-muted-foreground leading-relaxed">
          <strong>Sources:</strong>{' '}
          <a href="https://arxiv.org/abs/2603.22458" target="_blank" rel="noopener noreferrer" className="underline hover:opacity-100 opacity-80">MinerU-Diffusion paper ↗</a>{' '}
          for MinerU 2.5 baseline, PaddleOCR-VL throughput, and τ-presets;{' '}
          <a href="https://arxiv.org/abs/2605.16861" target="_blank" rel="noopener noreferrer" className="underline hover:opacity-100 opacity-80">PA-BDM paper ↗</a>{' '}
          for block-diffusion numbers;{' '}
          <a href="https://github.com/allenai/olmocr/issues/102" target="_blank" rel="noopener noreferrer" className="underline hover:opacity-100 opacity-80">olmOCR issue #102 ↗</a>{' '}
          for the token-budget breakdown.
        </div>
      </div>
    </div>
  );
}

function FailureExample({ modeId }) {
  switch (modeId) {
    case 'hallucination': return <HallucinationExample />;
    case 'unparseable':   return <UnparseableExample />;
    case 'repetition':    return <RepetitionExample />;
    case 'latency':       return <LatencyExample />;
    default:              return null;
  }
}

function FailuresPage() {
  const [activeMode, setActiveMode] = useState('hallucination');
  const mode = FAILURE_MODES.find(m => m.id === activeMode);

  return (
    <div className="space-y-6">

      {/* Main: mode selector + example */}
      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-6 md:p-8">
          <div className="grid xl:grid-cols-[1fr_2fr] gap-6 items-start">

            {/* Left — mode selector + description */}
            <div className="space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Failure modes</div>
              {FAILURE_MODES.map(m => (
                <button
                  key={m.id}
                  onClick={() => setActiveMode(m.id)}
                  className="w-full text-left rounded-lg px-3 py-2.5 text-sm transition-all flex items-center gap-2.5"
                  style={{
                    background: activeMode === m.id ? m.color + '15' : 'transparent',
                    border: `1.5px solid ${activeMode === m.id ? m.color : '#e2e8f0'}`,
                    color: activeMode === m.id ? m.color : 'inherit',
                    fontWeight: activeMode === m.id ? 600 : 500,
                  }}
                >
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: m.color }} />
                  {m.label}
                </button>
              ))}

              <div className="border-t pt-4 space-y-3">
                <div>
                  <SectionLabel color={mode.color}>{mode.label}</SectionLabel>
                  <h3 className="text-sm font-bold mt-1">{mode.title}</h3>
                  <p className="text-xs opacity-80 mt-2 leading-relaxed">{mode.desc}</p>
                </div>

                <div className="rounded-xl border-l-4 p-3" style={{ borderColor: mode.fixColor, background: mode.fixColor + '08' }}>
                  <div className="text-xs font-semibold mb-1" style={{ color: mode.fixColor }}>
                    Fix in Chapter {mode.fixChapter} →
                  </div>
                  <p className="text-xs opacity-85 leading-relaxed">{mode.fix}</p>
                </div>
              </div>
            </div>

            {/* Right — concrete real-world example */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeMode}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.16 }}
              >
                <FailureExample modeId={activeMode} />
              </motion.div>
            </AnimatePresence>

          </div>
        </CardContent>
      </Card>

      {/* The through-line — failures map to fixes in coming chapters */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">The through-line — each failure has a corresponding fix</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div className="grid md:grid-cols-2 gap-3">
            {FAILURE_MODES.map(m => (
              <div key={m.id} className="rounded-xl border p-3 text-xs flex items-center gap-3">
                <span className="rounded px-2 py-1 font-semibold whitespace-nowrap" style={{ background: m.color + '15', color: m.color }}>
                  {m.label}
                </span>
                <span className="text-muted-foreground">→</span>
                <span className="rounded px-2 py-1 font-semibold whitespace-nowrap" style={{ background: m.fixColor + '15', color: m.fixColor }}>
                  Ch {m.fixChapter}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

    </div>
  );
}

// ─── Chapter 5 data ──────────────────────────────────────────────────────────

const DATA_ACQUISITION_METHODS = [
  {
    id: 'public',
    label: 'Public academic datasets',
    icon: '📚',
    color: '#2563eb',
    pros: ['Free to use', 'Standard benchmarks', 'Reproducible'],
    cons: ['Small (millions, not billions)', 'Narrow distribution', 'Often saturated', 'Stale visual styles'],
    examples: [
      { name: 'PubLayNet',  url: 'https://github.com/ibm-aur-nlp/PubLayNet' },
      { name: 'DocVQA',     url: 'https://www.docvqa.org/' },
      { name: 'ChartQA',    url: 'https://github.com/vis-nlp/ChartQA' },
      { name: 'RVL-CDIP',   url: 'https://www.cs.cmu.edu/~aharley/rvl-cdip/' },
      { name: 'TableBank',  url: 'https://github.com/doc-analysis/TableBank' },
    ],
  },
  {
    id: 'commercial',
    label: 'Purchase labeled data',
    icon: '💳',
    color: '#0d9488',
    pros: ['High quality', 'Ready to use', 'Custom labeling possible'],
    cons: ['Expensive ($0.10–$5/label)', 'Licensing constraints', 'Slow turnaround', 'Vendor lock-in'],
    examples: [
      { name: 'Scale AI', url: 'https://scale.com/' },
      { name: 'Surge',    url: 'https://www.surgehq.ai/' },
      { name: 'LabelBox', url: 'https://labelbox.com/' },
      { name: 'iMerit',   url: 'https://imerit.net/' },
      { name: 'CloudFactory', url: 'https://www.cloudfactory.com/' },
    ],
  },
  {
    id: 'inhouse',
    label: 'Label in-house',
    icon: '👥',
    color: '#d97706',
    pros: ['Full quality control', 'Domain expertise stays internal', 'Iterate fast'],
    cons: ['Headcount cost', 'Slow at scale', 'Annotator drift', 'Hard to ramp'],
    examples: [
      { name: 'Internal annotation teams' },
      { name: 'Domain experts (legal, medical)' },
      { name: 'Bug-bash style review' },
    ],
  },
  {
    id: 'synthetic',
    label: 'Generate synthetic data',
    icon: '🏭',
    color: '#7c3aed',
    pros: ['Infinite scale', 'Perfect labels (by construction)', 'Controlled diversity', 'Privacy-safe'],
    cons: ['Distribution gap vs real docs', 'Hard to match real noise', 'Risk of model memorising templates'],
    examples: [
      { name: 'olmOCR-synthmix', url: 'https://huggingface.co/datasets/allenai/olmOCR-synthmix-1025' },
      { name: 'SynthDog',        url: 'https://github.com/clovaai/donut' },
      { name: 'Docmatix',        url: 'https://huggingface.co/datasets/HuggingFaceM4/Docmatix' },
      { name: 'Synthetic LaTeX papers' },
      { name: 'Generated invoices' },
    ],
  },
  {
    id: 'pseudo',
    label: 'Pseudo-label with a frontier model',
    icon: '🤖',
    color: '#dc2626',
    pros: ['Cheapest scaling path', 'Near-frontier quality', 'Distill into small student'],
    cons: ['Inherits teacher biases', 'Teacher hallucinations leak in', 'Needs filtering step'],
    examples: [
      { name: 'olmOCR-mix (GPT-4o labels)', url: 'https://huggingface.co/datasets/allenai/olmOCR-mix-0225' },
      { name: 'PaddleOCR-VL (ERNIE + Qwen2.5)', url: 'https://arxiv.org/abs/2510.14528' },
      { name: 'Anthropic-labeled corpora' },
    ],
  },
];

const SYNTHETIC_METHODS = [
  {
    id: 'html',
    label: 'HTML / CSS → PDF',
    color: '#2563eb',
    desc: 'Write templates in HTML/CSS, render to PDF (or screenshot). Perfect ground truth by construction — you know every element\'s text, position, and role.',
    usedBy: [
      { name: 'olmOCR-2 (Claude generates HTML templates)', url: 'https://arxiv.org/abs/2510.19817' },
      { name: 'Docmatix', url: 'https://huggingface.co/datasets/HuggingFaceM4/Docmatix' },
    ],
    seed: 'mid',
  },
  {
    id: 'latex',
    label: 'LaTeX → compiled PDF',
    color: '#7c3aed',
    desc: 'Sample equations, tables, sections from a LaTeX grammar, compile with pdflatex. Heart of academic doc parsing — formulas, captions, citations, multi-column layout.',
    usedBy: [
      { name: 'Nougat', url: 'https://github.com/facebookresearch/nougat' },
      { name: 'GOT-OCR 2.0', url: 'https://huggingface.co/stepfun-ai/GOT-OCR2_0' },
    ],
    seed: 'low',
  },
  {
    id: 'pixel',
    label: 'Raw pixel placement',
    color: '#d97706',
    desc: 'Programmatically place glyphs and text blocks onto background images. Old-school but works — generates massive volumes of text-on-natural-image data.',
    usedBy: [
      { name: 'SynthText', url: 'https://github.com/ankush-me/SynthText' },
      { name: 'Most scene-text recognizers' },
    ],
    seed: 'low',
  },
  {
    id: 'pdf_seeded',
    label: 'Real PDFs + augmentation',
    color: '#16a34a',
    desc: 'Take real documents, apply realistic distortion: skew, JPEG noise, ink bleed, fold marks, scan artefacts. Distribution stays real but labels are inherited from layout-perfect originals.',
    usedBy: [
      { name: 'Most production training pipelines' },
    ],
    seed: 'high',
  },
];

const TABLE_GEN_EXAMPLE = `# Synthetic table sample
schema = sample_schema(
  num_cols=random(3, 8),
  num_rows=random(4, 30),
  merged_header=random(0.3),
  merged_cells=random(0.15),
)
cells = sample_values(schema, distributions={
  'currency': dollar_amount,
  'percent':  percent_value,
  'date':     iso_date,
})
html = render_table(schema, cells)
pdf  = render_pdf(html, style=random_style())
# (pdf, html, schema) is the training triple
`;

const CHART_GEN_EXAMPLE = `# Synthetic chart sample (matplotlib)
data = sample_series(
  n_points=random(5, 30),
  trend='increasing' | 'noisy' | 'cyclic',
)
fig, ax = plt.subplots()
ax.bar(data.x, data.y)
ax.set_title(random_title())
ax.set_ylabel(random_unit_label())

# Save image + ground-truth CSV pair
fig.savefig('chart.png')
data.to_csv('chart.csv')
# Train: chart.png → chart.csv
`;

function TrainingPage() {
  return (
    <div className="space-y-6">

      {/* Section 1: Five ways to get data */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Five ways to get training data</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <p className="text-sm opacity-80 mb-5 leading-relaxed">
            Modern doc parsing models are not bottlenecked by architecture — they're bottlenecked by data.
            The single biggest determinant of model quality is which combination of these five sources you choose,
            and how you clean them.
          </p>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {DATA_ACQUISITION_METHODS.map(m => (
              <div key={m.id} className="rounded-2xl border overflow-hidden flex flex-col"
                style={{ borderColor: m.color + '55' }}
              >
                <div className="px-4 py-2.5 flex items-center gap-2"
                  style={{ background: m.color + '10' }}
                >
                  <span className="text-base">{m.icon}</span>
                  <span className="text-sm font-bold" style={{ color: m.color }}>{m.label}</span>
                </div>
                <div className="p-4 space-y-3 flex-1">
                  <div className="space-y-1">
                    {m.pros.map((p, i) => (
                      <div key={i} className="text-xs flex items-start gap-1.5">
                        <span className="font-bold flex-shrink-0" style={{ color: '#16a34a' }}>✓</span>
                        <span className="opacity-85">{p}</span>
                      </div>
                    ))}
                    {m.cons.map((c, i) => (
                      <div key={i} className="text-xs flex items-start gap-1.5">
                        <span className="font-bold flex-shrink-0" style={{ color: '#dc2626' }}>✗</span>
                        <span className="opacity-70">{c}</span>
                      </div>
                    ))}
                  </div>
                  <div className="border-t pt-2.5">
                    <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1.5">Examples</div>
                    <div className="flex flex-wrap gap-1">
                      {m.examples.map(e => (
                        e.url ? (
                          <a key={e.name} href={e.url} target="_blank" rel="noopener noreferrer"
                            className="rounded-full px-2 py-0.5 text-xs hover:underline"
                            style={{ background: m.color + '12', color: m.color }}
                          >
                            {e.name} ↗
                          </a>
                        ) : (
                          <span key={e.name} className="rounded-full px-2 py-0.5 text-xs"
                            style={{ background: m.color + '12', color: m.color }}
                          >
                            {e.name}
                          </span>
                        )
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <InfoBox color="#2563eb" title="The combination matters">
            <span className="text-xs">
              Every leading doc model uses a <strong>mix</strong> — typically synthetic + pseudo-labeled + a smaller human-labeled gold set.
              PaddleOCR-VL combined synthesis + multi-model pseudo-labeling + filtering to assemble 30M+ training samples
              and hit 96.33% on OmniDocBench v1.6.
            </span>
          </InfoBox>
        </CardContent>
      </Card>

      {/* Section 2: Inside synthetic generation */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Inside synthetic generation</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">
          <p className="text-sm opacity-80 leading-relaxed">
            "Synthetic" covers a wide spectrum. At one end, pure procedural generation from random seeds. At the other,
            real documents lightly perturbed. The sweet spot depends on what you want the model to learn.
          </p>

          {/* Spectrum bar */}
          <div className="rounded-xl border p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
              From-scratch ↔ seeded by real data
            </div>
            <div style={{ position: 'relative', height: 56 }}>
              <div style={{
                position: 'absolute', left: 0, right: 0, top: 25,
                height: 6, borderRadius: 3,
                background: 'linear-gradient(to right, #7c3aed, #d97706, #2563eb, #16a34a)',
              }} />
              {[
                { x: '4%',  label: 'LaTeX', color: '#7c3aed' },
                { x: '30%', label: 'Pixel placement', color: '#d97706' },
                { x: '60%', label: 'HTML / CSS', color: '#2563eb' },
                { x: '92%', label: 'Real PDFs + aug.', color: '#16a34a' },
              ].map((p, i) => (
                <div key={i} style={{
                  position: 'absolute', left: p.x,
                  transform: 'translateX(-50%)',
                  top: 16,
                }}>
                  <div style={{
                    width: 18, height: 18, borderRadius: '50%',
                    background: p.color, border: '3px solid white',
                    boxShadow: '0 0 0 2px ' + p.color,
                  }} />
                  <div style={{
                    position: 'absolute', top: 24, left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: 10, fontWeight: 700,
                    whiteSpace: 'nowrap', color: p.color,
                  }}>
                    {p.label}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex justify-between text-xs text-muted-foreground mt-2">
              <span>← maximum control, maximum distribution gap</span>
              <span>maximum realism, minimum scale →</span>
            </div>
          </div>

          {/* Generation methods grid */}
          <div className="grid md:grid-cols-2 gap-3">
            {SYNTHETIC_METHODS.map(m => (
              <div key={m.id} className="rounded-2xl border p-4 space-y-2"
                style={{ borderColor: m.color + '55' }}
              >
                <div className="text-sm font-bold" style={{ color: m.color }}>{m.label}</div>
                <p className="text-xs opacity-80 leading-relaxed">{m.desc}</p>
                <div className="text-xs">
                  <span className="text-muted-foreground">Used by: </span>
                  {(m.usedBy || []).map((u, i) => (
                    <span key={u.name}>
                      {i > 0 && <span className="text-muted-foreground">, </span>}
                      {u.url ? (
                        <a href={u.url} target="_blank" rel="noopener noreferrer"
                          className="hover:underline" style={{ color: m.color }}
                        >
                          {u.name} ↗
                        </a>
                      ) : (
                        <span style={{ color: m.color }}>{u.name}</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Subsection: synthetic tables */}
          <div className="rounded-2xl border" style={{ borderColor: '#d9770644' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#d9770612', color: '#d97706' }}
            >
              <span>📊</span> Synthetic tables — procedural schema + sampled values
            </div>
            <div className="p-4 grid md:grid-cols-2 gap-4">
              <div>
                <pre className="rounded-xl p-3 text-xs font-mono leading-relaxed overflow-x-auto"
                  style={{ background: '#0f172a', color: '#e2e8f0', whiteSpace: 'pre-wrap' }}
                >
{TABLE_GEN_EXAMPLE}
                </pre>
              </div>
              <div className="space-y-2 text-xs leading-relaxed">
                <p className="opacity-85">
                  Sample a schema (column count, types, header structure), then fill cells from typed distributions
                  (currencies, dates, percentages). Render to HTML, then to PDF with a random style.
                </p>
                <p className="opacity-85">
                  <strong style={{ color: '#d97706' }}>Why it works:</strong> every cell's row, column, span, and value
                  are known by construction. The model never trains against a label noisier than the schema.
                </p>
                <p className="opacity-85">
                  <strong style={{ color: '#d97706' }}>Used by:</strong> nearly every modern table-extraction
                  pipeline — TableTransformer's training data, Nougat's table augmentation, PaddleOCR-VL's table corpus.
                </p>
              </div>
            </div>
          </div>

          {/* Subsection: synthetic charts */}
          <div className="rounded-2xl border" style={{ borderColor: '#7c3aed44' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#7c3aed12', color: '#7c3aed' }}
            >
              <span>📈</span> Synthetic charts — render matplotlib with known data
            </div>
            <div className="p-4 grid md:grid-cols-2 gap-4">
              <div>
                <pre className="rounded-xl p-3 text-xs font-mono leading-relaxed overflow-x-auto"
                  style={{ background: '#0f172a', color: '#e2e8f0', whiteSpace: 'pre-wrap' }}
                >
{CHART_GEN_EXAMPLE}
                </pre>
              </div>
              <div className="space-y-2 text-xs leading-relaxed">
                <p className="opacity-85">
                  Sample numerical series with controlled properties (trend, noise, distribution), render to PNG with matplotlib
                  or seaborn, save the underlying values as ground-truth CSV.
                </p>
                <p className="opacity-85">
                  <strong style={{ color: '#7c3aed' }}>Why it works:</strong> chart-to-CSV is a notoriously brittle task —
                  reading axes, labels, and visual heights is error-prone. Synthetic pairs give you billions of
                  (chart image, exact CSV) examples that no human team could ever label.
                </p>
                <p className="opacity-85">
                  <strong style={{ color: '#7c3aed' }}>Used by:</strong> DePlot, ChartQA fine-tunes, MatCha, FigureQA
                  — synthetic data is the entire reason chart understanding became tractable.
                </p>
              </div>
            </div>
          </div>

        </CardContent>
      </Card>

      {/* Section 3: Pseudo-labeling at scale */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Pseudo-labeling at scale — the olmOCR story</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">
          <p className="text-sm opacity-80 leading-relaxed">
            The recent unlock: use a frontier model (GPT-4o, Claude, Gemini) as the teacher, label a curated subset of
            crawled PDFs, then distill the labels into a small specialist that runs cheaply forever.
          </p>

          {/* Pipeline diagram */}
          <div className="rounded-xl border p-4 overflow-x-auto">
            <svg viewBox="0 0 800 110" className="w-full" style={{ minWidth: 680 }}>
              {[
                { x: 20,  label: '240M+ PDFs',     sub: 'public web crawl',     color: '#94a3b8' },
                { x: 175, label: '~250K pages',     sub: 'curated sample',       color: '#2563eb' },
                { x: 335, label: 'GPT-4o labels',  sub: '$6,240 / M pages',     color: '#dc2626' },
                { x: 495, label: '7B student',     sub: 'distilled',            color: '#7c3aed' },
                { x: 655, label: '$176 / M pages', sub: '35× cheaper',          color: '#16a34a' },
              ].map((s, i, arr) => (
                <g key={i}>
                  <rect x={s.x} y={25} width={130} height={50} rx={8}
                    fill={s.color + '15'} stroke={s.color} strokeWidth={1.5}
                  />
                  <text x={s.x + 65} y={48} textAnchor="middle" fontSize={11} fontWeight={700} fill={s.color}>
                    {s.label}
                  </text>
                  <text x={s.x + 65} y={64} textAnchor="middle" fontSize={9} fill={s.color} opacity={0.75}>
                    {s.sub}
                  </text>
                  {i < arr.length - 1 && (
                    <g>
                      <line x1={s.x + 130} y1={50} x2={arr[i+1].x - 4} y2={50}
                        stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#ptr-arrow)" />
                    </g>
                  )}
                </g>
              ))}
              <defs>
                <marker id="ptr-arrow" markerWidth="6" markerHeight="6" refX={5} refY={3} orient="auto">
                  <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
                </marker>
              </defs>
            </svg>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <InfoBox color="#16a34a" title="The economics flip">
              <span className="text-xs">
                Pay GPT-4o <strong>once</strong> to label 250K pages. The 7B distilled student matches GPT-4o
                quality at <strong>1/35th</strong> the per-page cost — and you ship the weights, not an API dependency.
                The capex is the label budget. The opex is essentially free.
              </span>
            </InfoBox>
            <InfoBox color="#7c3aed" title="It's not zero-sum">
              <span className="text-xs">
                The teacher's hallucinations leak into the student. olmOCR-2 added an RL stage on top
                (covered in the next section). PaddleOCR-VL added <strong>multi-model cross-check filtering</strong>
                to catch teacher errors before they enter training.
              </span>
            </InfoBox>
          </div>

          {/* PaddleOCR-VL filtering subsection */}
          <div className="rounded-2xl border" style={{ borderColor: '#d9770644' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#d9770612', color: '#d97706' }}
            >
              <span>🔍</span> Multi-model labeling + filtering (PaddleOCR-VL)
            </div>
            <div className="p-4 space-y-3">
              <p className="text-xs opacity-85 leading-relaxed">
                Don't trust a single teacher. PaddleOCR-VL labels each image with multiple models, then discards
                samples where they disagree — a kind of cheap consensus filter.
              </p>
              <div className="overflow-x-auto">
                <svg viewBox="0 0 720 130" className="w-full" style={{ minWidth: 620 }}>
                  {/* Stage 1: PP-StructureV3 */}
                  <rect x={20}  y={20} width={130} height={36} rx={6} fill="#2563eb15" stroke="#2563eb" />
                  <text x={85} y={36} textAnchor="middle" fontSize={10} fontWeight={700} fill="#2563eb">PP-StructureV3</text>
                  <text x={85} y={48} textAnchor="middle" fontSize={8} fill="#2563eb" opacity={0.7}>(expert model: initial labels)</text>

                  <line x1={150} y1={38} x2={180} y2={38} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#flt-arr)" />

                  {/* Stage 2: ERNIE-4.5-VL */}
                  <rect x={185} y={20} width={130} height={36} rx={6} fill="#7c3aed15" stroke="#7c3aed" />
                  <text x={250} y={36} textAnchor="middle" fontSize={10} fontWeight={700} fill="#7c3aed">ERNIE-4.5-VL</text>
                  <text x={250} y={48} textAnchor="middle" fontSize={8} fill="#7c3aed" opacity={0.7}>(refines with prompt)</text>

                  <line x1={315} y1={38} x2={345} y2={38} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#flt-arr)" />

                  {/* Stage 3: Qwen2.5-VL */}
                  <rect x={350} y={20} width={130} height={36} rx={6} fill="#d9770615" stroke="#d97706" />
                  <text x={415} y={36} textAnchor="middle" fontSize={10} fontWeight={700} fill="#d97706">Qwen2.5-VL</text>
                  <text x={415} y={48} textAnchor="middle" fontSize={8} fill="#d97706" opacity={0.7}>(second refinement)</text>

                  <line x1={480} y1={38} x2={510} y2={38} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#flt-arr)" />

                  {/* Filter step */}
                  <rect x={515} y={10} width={180} height={56} rx={6} fill="#dc262615" stroke="#dc2626" strokeWidth={2} />
                  <text x={605} y={28} textAnchor="middle" fontSize={10} fontWeight={700} fill="#dc2626">Hallucination filter</text>
                  <text x={605} y={42} textAnchor="middle" fontSize={8} fill="#dc2626" opacity={0.8}>discard if labels disagree</text>
                  <text x={605} y={54} textAnchor="middle" fontSize={8} fill="#dc2626" opacity={0.8}>or fail consistency checks</text>

                  {/* Final */}
                  <text x={360} y={100} textAnchor="middle" fontSize={11} fontWeight={700} fill="#16a34a">
                    → 30M+ high-quality training samples (filtered)
                  </text>

                  <defs>
                    <marker id="flt-arr" markerWidth="6" markerHeight="6" refX={5} refY={3} orient="auto">
                      <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
                    </marker>
                  </defs>
                </svg>
              </div>
            </div>
          </div>

        </CardContent>
      </Card>

      {/* Section 4: CE vs RL */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Cross-entropy vs RL — the alignment story</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">

          <p className="text-sm opacity-80 leading-relaxed">
            How a model is trained matters more than what it's trained on. The training objective decides which kinds of
            mistakes get penalised. Cross-entropy is brutally blunt — every token error counts equally. RL with
            verifiable rewards can target exactly what you care about.
          </p>

          {/* Side-by-side comparison */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="rounded-2xl border overflow-hidden" style={{ borderColor: '#dc262644' }}>
              <div className="px-4 py-2.5 text-sm font-bold" style={{ background: '#dc262612', color: '#dc2626' }}>
                Cross-entropy training
              </div>
              <div className="p-4 space-y-3 text-xs">
                <p className="opacity-85 leading-relaxed">
                  Maximise <code className="font-mono px-1" style={{ background: '#fee2e2' }}>log P(target token)</code>
                  &nbsp;over every position. Every token error counts the same: a typo and a hallucinated number both shift the loss by the same amount.
                </p>
                <div className="space-y-1.5 mt-2">
                  <div className="font-semibold" style={{ color: '#dc2626' }}>What it doesn't penalise:</div>
                  <ul className="space-y-1 ml-3 opacity-85">
                    <li>• Confidently wrong (vs appropriately unsure)</li>
                    <li>• JSON that doesn't parse</li>
                    <li>• LaTeX that doesn't compile</li>
                    <li>• Table rows that don't align</li>
                    <li>• Wrong totals that look plausible</li>
                  </ul>
                </div>
                <p className="opacity-85 leading-relaxed mt-3 italic">
                  You can hit 99.9% character accuracy and still ship a model that hallucinates,
                  because hallucination isn't a character error.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border overflow-hidden" style={{ borderColor: '#16a34a44' }}>
              <div className="px-4 py-2.5 text-sm font-bold" style={{ background: '#16a34a12', color: '#16a34a' }}>
                RL with verifiable rewards
              </div>
              <div className="p-4 space-y-3 text-xs">
                <p className="opacity-85 leading-relaxed">
                  Train against a reward function that directly measures what we want: structurally valid output,
                  correct values, faithful structure. The model learns to optimise the downstream metric, not a proxy.
                </p>
                <div className="space-y-1.5 mt-2">
                  <div className="font-semibold" style={{ color: '#16a34a' }}>Verifiable rewards in OCR:</div>
                  <ul className="space-y-1 ml-3 opacity-85 font-mono">
                    <li>• <code style={{ background: '#dcfce7' }}>JSON.parse(output)</code> — does it parse?</li>
                    <li>• <code style={{ background: '#dcfce7' }}>render(latex)</code> — does it compile?</li>
                    <li>• <code style={{ background: '#dcfce7' }}>table.cols == N</code> — right shape?</li>
                    <li>• <code style={{ background: '#dcfce7' }}>sum(items) == total</code> — math checks?</li>
                    <li>• <code style={{ background: '#dcfce7' }}>html_to_dom(out)</code> — valid markup?</li>
                  </ul>
                </div>
                <p className="opacity-85 leading-relaxed mt-3 italic">
                  Same idea as math/code RL — programmatic graders reward correctness, penalize hallucination directly.
                </p>
              </div>
            </div>
          </div>

          {/* Reading-order alignment subsection */}
          <div className="rounded-2xl border" style={{ borderColor: '#d9770644' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#d9770612', color: '#d97706' }}
            >
              <span>🔀</span> The reading-order alignment problem — the strongest argument against CE
            </div>
            <div className="p-4 space-y-3 text-xs">
              <p className="opacity-85 leading-relaxed">
                Here's the failure that most clearly exposes cross-entropy's blind spot. On a two-column paper or a
                document with sidebars, the model might emit the <em>correct words</em> but in a slightly different
                order — left column then sidebar, instead of left column then right column. Or it jumps a paragraph
                early. Content is perfect. Sequence is shifted.
              </p>

              <div className="grid md:grid-cols-2 gap-3 mt-3">
                {/* Ground truth */}
                <div className="rounded-lg p-3" style={{ background: '#0f172a' }}>
                  <div className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>Ground truth (correct order)</div>
                  <div className="font-mono text-xs leading-relaxed">
                    <div style={{ color: '#86efac' }}>1. Title: Quarterly Report</div>
                    <div style={{ color: '#86efac' }}>2. Body left col, para 1</div>
                    <div style={{ color: '#86efac' }}>3. Body left col, para 2</div>
                    <div style={{ color: '#86efac' }}>4. Body right col, para 1</div>
                    <div style={{ color: '#86efac' }}>5. Body right col, para 2</div>
                    <div style={{ color: '#86efac' }}>6. Footnote</div>
                  </div>
                </div>

                {/* Model output */}
                <div className="rounded-lg p-3" style={{ background: '#0f172a' }}>
                  <div className="text-xs font-semibold mb-2" style={{ color: '#94a3b8' }}>Model output — all content right, order shifted</div>
                  <div className="font-mono text-xs leading-relaxed">
                    <div style={{ color: '#86efac' }}>1. Title: Quarterly Report</div>
                    <div style={{ color: '#86efac' }}>2. Body left col, para 1</div>
                    <div style={{ color: '#fde68a' }}>3. Body right col, para 1</div>
                    <div style={{ color: '#fde68a' }}>4. Body left col, para 2</div>
                    <div style={{ color: '#86efac' }}>5. Body right col, para 2</div>
                    <div style={{ color: '#86efac' }}>6. Footnote</div>
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-3 mt-3">
                <div className="rounded-xl border-l-4 p-3" style={{ borderColor: '#dc2626', background: '#dc262608' }}>
                  <div className="font-semibold text-xs mb-1" style={{ color: '#dc2626' }}>Cross-entropy view</div>
                  <p className="text-xs opacity-85 leading-relaxed">
                    Every token from paragraph 3 onward is misaligned with the target sequence.
                    Loss spikes <strong>as if the model hallucinated everything</strong>. Two paragraphs of perfect content
                    score the same as two paragraphs of nonsense.
                  </p>
                </div>
                <div className="rounded-xl border-l-4 p-3" style={{ borderColor: '#16a34a', background: '#16a34a08' }}>
                  <div className="font-semibold text-xs mb-1" style={{ color: '#16a34a' }}>RL view — graded by content tests</div>
                  <p className="text-xs opacity-85 leading-relaxed">
                    Unit tests like "<code className="font-mono">'para 1 of left col' in output</code>" pass.
                    "<code className="font-mono">reading_order_distance(out, gt) &lt; threshold</code>" fails — but with much smaller
                    penalty than a content error. <strong>We grade what we care about.</strong>
                  </p>
                </div>
              </div>

              <p className="opacity-85 leading-relaxed italic mt-2">
                On long, complex layouts (papers, magazines, multi-language docs) this is the dominant failure
                pattern that CE training amplifies. Modern benchmarks (covered in Chapter 7) have moved to
                content-faithfulness metrics partly to stop punishing models for harmless order shifts.
              </p>
            </div>
          </div>

          {/* olmOCR-2 GRPO callout */}
          <div className="rounded-2xl border" style={{ borderColor: '#0d948844' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#0d948812', color: '#0d9488' }}
            >
              <span>🎯</span> olmOCR-2 — GRPO RL on synthetic unit tests
            </div>
            <div className="p-4 grid md:grid-cols-[1fr_1fr] gap-4 text-xs">
              <div className="space-y-2">
                <p className="opacity-85 leading-relaxed">
                  olmOCR-2's recipe: <strong>Claude generates HTML templates → render to PDF → build programmatic unit tests
                  against the templates → GRPO trains the model to pass them.</strong>
                </p>
                <p className="opacity-85 leading-relaxed">
                  Because the data is synthetic, the unit tests are essentially free to create. Because the tests are
                  programmatic, the reward signal is exact and non-gameable. Because GRPO is on-policy,
                  the model directly learns to produce extracts that pass.
                </p>
              </div>
              <div className="rounded-xl p-3" style={{ background: '#0f172a' }}>
                <div className="text-xs text-slate-400 mb-2 font-mono"># Example unit tests</div>
                <div className="font-mono text-xs leading-relaxed space-y-1">
                  <div style={{ color: '#86efac' }}>✓ table.row_count == 5</div>
                  <div style={{ color: '#86efac' }}>✓ "Q3 2024" in output</div>
                  <div style={{ color: '#86efac' }}>✓ formula compiles</div>
                  <div style={{ color: '#86efac' }}>✓ output.endswith("∎")</div>
                  <div style={{ color: '#86efac' }}>✓ JSON.parse(output)</div>
                  <div style={{ color: '#fca5a5' }}>✗ "footnote 3" missing</div>
                  <div style={{ color: '#94a3b8' }}>--&nbsp;reward = 5/6 = 0.833</div>
                </div>
              </div>
            </div>
          </div>

          <InfoBox color="#dc2626" title="This closes the loop on Chapter 4">
            <span className="text-xs">
              Remember "Unparseable outputs" from the failure modes? RL with verifiable rewards is the direct
              answer — the training objective <strong>literally</strong> penalises outputs that fail to parse. Cross-entropy
              can't tell those apart from valid outputs. The shift to RL is one of the bigger recent wins in doc parsing.
            </span>
          </InfoBox>

        </CardContent>
      </Card>

    </div>
  );
}

// ─── Chapter 6 data ──────────────────────────────────────────────────────────

const THROUGHPUT_DATA = [
  { model: 'PaddleOCR-VL (static decode, 32 steps)', tps: 22,  type: 'ar',   color: '#dc2626' },
  { model: 'PaddleOCR-VL (autoregressive)',           tps: 40,  type: 'ar',   color: '#dc2626' },
  { model: 'MinerU 2.5 (autoregressive baseline)',    tps: 52,  type: 'ar',   color: '#dc2626' },
  { model: 'MinerU-Diffusion (τ=0.95, no acc. loss)', tps: 109, type: 'diff', color: '#16a34a' },
  { model: 'MinerU-Diffusion (τ=0.6, ~99% acc.)',     tps: 165, type: 'diff', color: '#16a34a' },
  { model: 'PA-BDM (prefix-adaptive block diffusion)', tps: 267, type: 'diff', color: '#16a34a' },
];

const TAU_PRESETS = [
  { tau: 0.99, tps: 60,  acc: '~100%', label: 'High confidence — almost AR' },
  { tau: 0.95, tps: 109, acc: '99.9%', label: 'Sweet spot — 2.1× faster, no accuracy loss' },
  { tau: 0.80, tps: 140, acc: '99.4%', label: 'Aggressive — 2.7× faster' },
  { tau: 0.60, tps: 165, acc: '~99%',  label: 'Peak speed — 3.2× faster, ~1% drop' },
];

function ARBottleneckViz() {
  const TOKENS = ['The', ' quarterly', ' report', ' shows', ' a', ' 18', '%', ' increase'];
  return (
    <div className="rounded-xl border p-4 space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Autoregressive: 1 token per forward pass
      </div>
      <div className="space-y-1.5 font-mono text-xs">
        {TOKENS.map((tok, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-muted-foreground" style={{ minWidth: 50 }}>t = {i + 1}</span>
            <div className="flex gap-0.5 items-center">
              {TOKENS.slice(0, i).map((t, j) => (
                <span key={j} className="px-1 py-0.5 rounded" style={{ background: '#dcfce7', color: '#16a34a' }}>
                  {t.trim() || ' '}
                </span>
              ))}
              <span className="px-1 py-0.5 rounded font-bold" style={{ background: '#fef3c7', color: '#d97706' }}>
                {tok.trim() || ' '}
              </span>
              {Array.from({ length: TOKENS.length - i - 1 }).map((_, j) => (
                <span key={j} className="px-1 py-0.5 rounded" style={{ background: '#f1f5f9', color: '#94a3b8' }}>···</span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="text-xs text-muted-foreground">
        Each row is one full LLM forward pass. <strong style={{ color: '#dc2626' }}>5,000 tokens = 5,000 passes.</strong>
      </div>
    </div>
  );
}

function DiffusionViz() {
  const FINAL = ['The', ' quarterly', ' report', ' shows', ' a', ' 18', '%', ' increase'];
  const STEPS = [
    [null, null, null, null, null, null, null, null],
    [null, null, null, 'shows', null, null, null, null],
    [null, ' quarterly', null, 'shows', null, ' 18', null, null],
    ['The', ' quarterly', ' report', 'shows', null, ' 18', '%', ' increase'],
    ['The', ' quarterly', ' report', 'shows', ' a', ' 18', '%', ' increase'],
  ];
  return (
    <div className="rounded-xl border p-4 space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Diffusion: parallel masked denoising
      </div>
      <div className="space-y-1.5 font-mono text-xs">
        {STEPS.map((step, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-muted-foreground" style={{ minWidth: 50 }}>step {i}</span>
            <div className="flex gap-0.5 items-center">
              {step.map((t, j) => {
                const isNew = t !== null && (i === 0 || STEPS[i-1][j] === null);
                if (t === null) {
                  return (
                    <span key={j} className="px-1 py-0.5 rounded font-bold" style={{ background: '#1e293b', color: '#94a3b8' }}>
                      [MASK]
                    </span>
                  );
                }
                return (
                  <span key={j} className="px-1 py-0.5 rounded"
                    style={{
                      background: isNew ? '#fef3c7' : '#dcfce7',
                      color: isNew ? '#d97706' : '#16a34a',
                      fontWeight: isNew ? 700 : 400,
                    }}
                  >
                    {t.trim() || ' '}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="text-xs text-muted-foreground">
        Multiple tokens unmasked per pass — high-confidence ones first.
        <strong style={{ color: '#16a34a' }}> 5 steps instead of 8. ~3× fewer forward passes.</strong>
      </div>
    </div>
  );
}

function ThroughputChart() {
  const max = Math.max(...THROUGHPUT_DATA.map(d => d.tps));
  return (
    <div className="space-y-2">
      {THROUGHPUT_DATA.map(d => (
        <div key={d.model} className="grid grid-cols-[1fr_3fr_64px] gap-3 items-center text-sm">
          <div className="text-xs font-medium truncate" title={d.model}>{d.model}</div>
          <div className="rounded-full overflow-hidden" style={{ background: '#f1f5f9', height: 22 }}>
            <div
              className="h-full rounded-full flex items-center justify-end pr-2 transition-all duration-700"
              style={{ width: `${(d.tps / max) * 100}%`, background: d.color }}
            >
              <span className="text-xs font-bold text-white">{d.tps}</span>
            </div>
          </div>
          <div className="text-xs font-mono text-muted-foreground text-right">
            ≈{Math.round(5000 / d.tps)} s/pg
          </div>
        </div>
      ))}
    </div>
  );
}

function InferencePage() {
  const [tauIdx, setTauIdx] = useState(1);
  const tau = TAU_PRESETS[tauIdx];

  return (
    <div className="space-y-6">

      {/* Section 1: AR bottleneck */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">The autoregressive bottleneck</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          <p className="text-sm opacity-80 leading-relaxed">
            Every VLM we've seen so far decodes one token at a time. Each token requires a full pass through the LLM —
            attention over every previous token, every layer. Dense documents produce thousands of output tokens.
            The wall is unavoidable as long as you stay autoregressive.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <ARBottleneckViz />
            <div className="rounded-xl border p-4 space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Why does it have to be this way?
              </div>
              <p className="text-xs opacity-85 leading-relaxed">
                It doesn't. Causal left-to-right generation is what makes text models <em>creative</em> — but
                document parsing is different.
              </p>
              <p className="text-xs opacity-85 leading-relaxed">
                <strong>OCR is deterministic.</strong> Given an image, there is exactly one correct text output.
                There is no ambiguity to model, no probability mass to spread. The image fully constrains the answer.
              </p>
              <p className="text-xs opacity-85 leading-relaxed">
                <strong>So why are we generating token-by-token?</strong> Left-to-right is a serialization artefact
                inherited from language modelling — it has nothing to do with the task. Tokens can be recovered in
                parallel. That insight underpins everything in the rest of this chapter.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Section 2: VRFM coarse-to-fine */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">VRFM — process only the valid regions (PaddleOCR-VL)</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          <p className="text-sm opacity-80 leading-relaxed">
            Before the VLM even starts decoding, you can save 60% of the work by simply <em>not looking at most of the page.</em>
            PaddleOCR-VL's Valid Region Focus Module measured that on real documents, only ~39% of the image area
            contains actual content worth recognising. The other 61% is background, margins, decorative elements, watermarks.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            {/* Diagram */}
            <div className="rounded-xl border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                The coarse-to-fine pipeline
              </div>
              <svg viewBox="0 0 360 220" className="w-full">
                {/* Input */}
                <rect x={10}  y={90} width={70} height={40} rx={6} fill="#f1f5f9" stroke="#94a3b8" />
                <text x={45} y={108} textAnchor="middle" fontSize={9} fontWeight={700}>Full image</text>
                <text x={45} y={120} textAnchor="middle" fontSize={8} opacity={0.6}>(100% pixels)</text>

                {/* Arrow to VRFM */}
                <line x1={80} y1={110} x2={108} y2={110} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#vr-arr)" />

                {/* VRFM box (coarse) */}
                <rect x={110} y={70} width={100} height={80} rx={6} fill="#2563eb15" stroke="#2563eb" strokeWidth={1.5} />
                <text x={160} y={90} textAnchor="middle" fontSize={10} fontWeight={700} fill="#2563eb">VRFM</text>
                <text x={160} y={102} textAnchor="middle" fontSize={8} fill="#2563eb" opacity={0.85}>(lightweight)</text>
                <text x={160} y={116} textAnchor="middle" fontSize={8} fill="#2563eb" opacity={0.7}>RT-DETR</text>
                <text x={160} y={128} textAnchor="middle" fontSize={8} fill="#2563eb" opacity={0.7}>+ Pointer Net</text>
                <text x={160} y={142} textAnchor="middle" fontSize={8} fill="#2563eb" opacity={0.7}>(reading order)</text>

                {/* Arrow to crops */}
                <line x1={210} y1={110} x2={238} y2={110} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#vr-arr)" />

                {/* Cropped regions */}
                <rect x={240} y={75} width={50} height={70} rx={4} fill="#7c3aed15" stroke="#7c3aed" strokeDasharray="3 2" />
                <text x={265} y={108} textAnchor="middle" fontSize={9} fontWeight={700} fill="#7c3aed">39%</text>
                <text x={265} y={120} textAnchor="middle" fontSize={7} fill="#7c3aed" opacity={0.7}>valid only</text>

                {/* Arrow to VLM */}
                <line x1={290} y1={110} x2={310} y2={110} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#vr-arr)" />

                {/* 0.9B VLM */}
                <rect x={312} y={85} width={42} height={50} rx={6} fill="#16a34a15" stroke="#16a34a" strokeWidth={1.5} />
                <text x={333} y={104} textAnchor="middle" fontSize={9} fontWeight={700} fill="#16a34a">0.9B</text>
                <text x={333} y={116} textAnchor="middle" fontSize={8} fill="#16a34a" opacity={0.7}>VLM</text>
                <text x={333} y={128} textAnchor="middle" fontSize={7} fill="#16a34a" opacity={0.7}>(fine)</text>

                {/* Below: skipped region */}
                <text x={180} y={180} textAnchor="middle" fontSize={9} fontWeight={700} fill="#dc2626">
                  61% of pixels never reach the VLM
                </text>
                <text x={180} y={195} textAnchor="middle" fontSize={8} fill="#dc2626" opacity={0.7}>
                  (margins, watermarks, blanks, decorative elements)
                </text>

                <defs>
                  <marker id="vr-arr" markerWidth="6" markerHeight="6" refX={5} refY={3} orient="auto">
                    <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
                  </marker>
                </defs>
              </svg>
            </div>

            {/* Why it wins */}
            <div className="space-y-3">
              <InfoBox color="#2563eb" title="Two wins from one architectural change">
                <ul className="text-xs space-y-1.5 list-none">
                  <li>
                    <strong className="text-blue-600">Faster:</strong> the lightweight detector runs in ~10ms; the
                    expensive 0.9B VLM only processes 39% of the image area.
                  </li>
                  <li>
                    <strong className="text-blue-600">Less hallucination:</strong> the VLM literally never sees the
                    irrelevant 61% — it cannot accidentally transcribe a watermark as body text.
                  </li>
                </ul>
              </InfoBox>
              <InfoBox color="#16a34a" title="Where this maps back to Chapter 4">
                <span className="text-xs">
                  This is the direct fix for "Slow & memory hungry" — fewer pixels in means fewer
                  output tokens out, means faster inference. It also addresses background-bleed
                  hallucination, where models confidently transcribe page headers or watermarks into the main output.
                </span>
              </InfoBox>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Section 3: Diffusion decoding */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Diffusion decoding — parallel by design (MinerU-Diffusion)</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          <p className="text-sm opacity-80 leading-relaxed">
            If OCR is deterministic, you can generate multiple tokens at once. MinerU-Diffusion starts with the entire
            output as <code className="font-mono px-1 py-0.5 rounded" style={{ background: '#1e293b', color: '#e2e8f0' }}>[MASK]</code> tokens
            and progressively unmasks them based on confidence. Each pass commits the high-confidence tokens and
            re-queries the masked ones.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <DiffusionViz />
            <div className="space-y-3">
              <InfoBox color="#7c3aed" title="The confidence threshold τ">
                <span className="text-xs">
                  At each step, only tokens with model confidence above <code className="font-mono">τ</code> get
                  committed. Higher <code className="font-mono">τ</code> = fewer commits per pass = closer to
                  autoregressive but safer. Lower <code className="font-mono">τ</code> = more parallelism but
                  more risk of errors that can't be undone.
                </span>
              </InfoBox>

              {/* τ slider */}
              <div className="rounded-xl border p-4 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Pick a confidence threshold
                </div>
                <div className="flex gap-1.5 flex-wrap">
                  {TAU_PRESETS.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => setTauIdx(i)}
                      className="rounded-lg px-2.5 py-1 text-xs font-mono font-semibold transition-all"
                      style={{
                        background: tauIdx === i ? '#7c3aed' : '#f1f5f9',
                        color: tauIdx === i ? 'white' : '#7c3aed',
                        border: `1px solid ${tauIdx === i ? '#7c3aed' : '#e2e8f0'}`,
                      }}
                    >
                      τ = {p.tau}
                    </button>
                  ))}
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Throughput</span>
                    <span className="font-mono font-bold" style={{ color: '#7c3aed' }}>{tau.tps} TPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Relative accuracy</span>
                    <span className="font-mono font-bold" style={{ color: '#7c3aed' }}>{tau.acc}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Speedup vs MinerU 2.5 baseline</span>
                    <span className="font-mono font-bold" style={{ color: '#7c3aed' }}>
                      {(tau.tps / 52).toFixed(2)}×
                    </span>
                  </div>
                  <div className="border-t pt-1.5 mt-1.5 text-xs opacity-80 italic">{tau.label}</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Section 4: PA-BDM + throughput chart */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Block diffusion — keeping structure intact (PA-BDM)</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          <p className="text-sm opacity-80 leading-relaxed">
            Pure global diffusion has a problem: tokens are denoised independently, but document structure (tables,
            formulas, JSON) demands joint constraints. PA-BDM (Prefix-Adaptive Block Diffusion) decodes in <em>blocks</em>:
            tokens within a block are denoised in parallel, but blocks are committed in order. You get parallelism
            within local context, autoregressive coherence between blocks. Best of both.
          </p>

          <div className="rounded-xl border p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
              Throughput comparison — same task, different decoders
            </div>
            <ThroughputChart />
            <div className="text-xs text-muted-foreground mt-4 leading-relaxed space-y-1">
              <div>
                "≈s/pg" assumes a 5,000-output-token page. Real-world latency depends on hardware and batch size.
              </div>
              <div>
                <strong>Sources:</strong>{' '}
                <a href="https://arxiv.org/abs/2603.22458" target="_blank" rel="noopener noreferrer" className="underline hover:opacity-100 opacity-80">MinerU-Diffusion paper ↗</a>{' '}
                for MinerU 2.5 baseline (52 TPS), PaddleOCR-VL throughput (22 / 40 TPS), and MinerU-Diffusion τ-thresholds (109 / 165 TPS);{' '}
                <a href="https://arxiv.org/abs/2605.16861" target="_blank" rel="noopener noreferrer" className="underline hover:opacity-100 opacity-80">PA-BDM paper ↗</a>{' '}
                reports 267 TPS at 3B params.
              </div>
            </div>
          </div>

          <InfoBox color="#16a34a" title="The headline">
            <span className="text-xs">
              <strong>PA-BDM at 267 TPS is ~5× faster than the MinerU 2.5 autoregressive baseline at 52 TPS</strong> — with
              comparable or better quality on structured outputs. We went from
              "one page per 100 seconds" to "five pages per second" in under a year by changing the decoder, not the model.
            </span>
          </InfoBox>
        </CardContent>
      </Card>

    </div>
  );
}

// ─── Chapter 7 data ──────────────────────────────────────────────────────────

const LANDSCAPE_MODELS = [
  {
    id: 'mineru',
    name: 'MinerU 2.5 / MinerU-Diffusion',
    org: 'Shanghai AI Lab · OpenDataLab',
    released: '2024 → 2026',
    params: 'Pipeline + 2.5B diffusion',
    license: 'Modified Apache 2.0',
    color: '#2563eb',
    headline: 'Diffusion decoding pioneer for document parsing',
    technique: ['Pipeline + VLM hybrid backends', 'Block-wise masked diffusion decoder', 'Sliding-window memory for long docs'],
    bestFor: 'Scientific literature, long PDFs, CJK content',
    benchmark: '86.2 on OmniDocBench v1.5 · 3.2× speedup over AR',
    chapterTieIn: 'ch6 — first to ship diffusion decoding for OCR',
    links: [
      { label: 'GitHub', url: 'https://github.com/opendatalab/MinerU' },
      { label: 'MinerU-Diffusion paper', url: 'https://arxiv.org/abs/2603.22458' },
      { label: 'PA-BDM paper', url: 'https://arxiv.org/abs/2605.16861' },
    ],
  },
  {
    id: 'olmocr',
    name: 'olmOCR-2',
    org: 'Allen AI (AI2)',
    released: '2025 (10/25)',
    params: '7B (FP8 quantized)',
    license: 'Apache 2.0',
    color: '#dc2626',
    headline: 'Pseudo-labeling at scale + GRPO RL on synthetic unit tests',
    technique: ['260K pages labeled by GPT-4o → distilled 7B', 'Synthetic HTML templates → unit-test rewards', 'GRPO RL training'],
    bestFor: 'Cost-sensitive PDF→text at scale, training data preparation',
    benchmark: '$176 per 1M pages (35× cheaper than GPT-4o direct)',
    chapterTieIn: 'ch5 — defined the modern pseudo-labeling playbook',
    links: [
      { label: 'GitHub', url: 'https://github.com/allenai/olmocr' },
      { label: 'olmOCR-mix dataset', url: 'https://huggingface.co/datasets/allenai/olmOCR-mix-0225' },
      { label: 'olmOCR-2 paper', url: 'https://arxiv.org/abs/2510.19817' },
    ],
  },
  {
    id: 'paddle',
    name: 'PaddleOCR-VL-1.6',
    org: 'Baidu PaddlePaddle',
    released: '2026 (06/26)',
    params: '0.9B',
    license: 'Apache 2.0',
    color: '#d97706',
    headline: '0.9B model holding state-of-the-art via coarse-to-fine VRFM',
    technique: ['VRFM coarse-to-fine architecture', 'Multi-model labeling + hallucination filtering', 'CPT → SFT → RL progressive training', '109-language coverage'],
    bestFor: 'SOTA accuracy at small model size, CJK docs, complex tables',
    benchmark: '96.33% on OmniDocBench v1.6 (current SOTA)',
    chapterTieIn: 'ch5 + ch6 — VRFM for inference, multi-model filtering for training',
    links: [
      { label: 'GitHub', url: 'https://github.com/PaddlePaddle/PaddleOCR' },
      { label: 'Hugging Face', url: 'https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6' },
      { label: 'Paper (VRFM)', url: 'https://arxiv.org/abs/2510.14528' },
    ],
  },
  {
    id: 'nemotron',
    name: 'Nemotron Parse 1.1',
    org: 'NVIDIA',
    released: '2025 (11/25)',
    params: '885M',
    license: 'NVIDIA license · NIM-deployable',
    color: '#7c3aed',
    headline: 'Heavy encoder, light decoder — built for grounded extraction',
    technique: ['C-RADIO heavy vision encoder + mBART decoder', 'Three modes: markdown_bbox, markdown_no_bbox, detection_only', 'Token-compressed (TC) variant for throughput', 'TensorRT-LLM backend'],
    bestFor: 'Enterprise deployments, grounded extraction with bboxes',
    benchmark: 'Near-perfect on GOT Dense OCR · strong on RD-TableBench',
    chapterTieIn: 'ch5 — released alongside Nemotron-VLM-Dataset-v2 (synthetic + human-labeled training corpus)',
    links: [
      { label: 'Blog', url: 'https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/' },
      { label: 'API reference', url: 'https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse' },
      { label: 'NIM docs', url: 'https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html' },
    ],
  },
  {
    id: 'hunyuan',
    name: 'HunyuanOCR',
    org: 'Tencent Hunyuan Vision Team',
    released: '2025 (11/25)',
    params: '1B (Hunyuan-ViT + 0.5B LM)',
    license: 'Apache 2.0',
    color: '#0d9488',
    headline: 'Sub-3B SOTA on OCRBench — unified spotting + parsing + IE + VQA + translation',
    technique: ['Hunyuan-ViT (SigLIP-v2 backbone) + Adaptive MLP + Hunyuan-0.5B LM', 'XD-RoPE decomposes positions into text/height/width/time subspaces', 'End-to-end — no layout pre-processor', 'Reinforcement learning during post-training', 'Text-image translation for 14+ source languages'],
    bestFor: 'Multi-task OCR on a budget · text-image translation · subtitle extraction · scene-text VQA',
    benchmark: 'SOTA OCRBench (<3B params) · won ICDAR 2025 DIMT Challenge (Small Model Track)',
    chapterTieIn: 'ch5 + ch6 — RL for OCR + end-to-end design with no pipeline error propagation',
    links: [
      { label: 'Hugging Face', url: 'https://huggingface.co/tencent/HunyuanOCR' },
      { label: 'GitHub', url: 'https://github.com/Tencent-Hunyuan/HunyuanOCR' },
      { label: 'Tech report', url: 'https://arxiv.org/abs/2511.19575' },
    ],
  },
  {
    id: 'chandra',
    name: 'Chandra OCR 2',
    org: 'Datalab · Vik Paruchuri',
    released: '2026 (03/26)',
    params: '5B (8B & 2B quantized variants)',
    license: 'Apache 2.0 code · OpenRAIL-M weights',
    color: '#16a34a',
    headline: 'Best-in-class handwriting + math + multilingual OCR',
    technique: ['Layout-aware extraction with checkbox reconstruction', 'Heavy investment in handwriting + math data', '90+ language coverage'],
    bestFor: 'Handwriting, math-heavy docs, forms, broad multilingual',
    benchmark: '85.9% on olmOCR-bench (#1 open source) · 77.8% on 43-lang multilingual bench',
    chapterTieIn: 'ch5 — exemplifies what synthetic data + careful labeling unlocks for hard subdomains',
    links: [
      { label: 'GitHub', url: 'https://github.com/datalab-to/chandra' },
      { label: 'Hugging Face', url: 'https://huggingface.co/datalab-to/chandra-ocr-2' },
      { label: 'Launch blog', url: 'https://landing.datalab.to/blog/introducing-chandra' },
    ],
  },
];

// ─── Chapter 7 multilingual section data ─────────────────────────────────────

// Language coverage at a glance
const MULTILINGUAL_COVERAGE = [
  {
    model: 'PaddleOCR-VL-1.6',
    count: 109,
    notes: 'Largest claimed language coverage; CJK + ancient scripts a focus.',
    color: '#d97706',
    url: 'https://github.com/PaddlePaddle/PaddleOCR',
  },
  {
    model: 'Chandra OCR 2',
    count: 90,
    notes: 'Best open-source on tough low-resource langs (Amharic, Khmer, Burmese).',
    color: '#dc2626',
    url: 'https://github.com/datalab-to/chandra',
  },
  {
    model: 'Surya 2',
    count: 90,
    notes: '650M VLM, multilingual sweet spot for self-hosting.',
    color: '#2563eb',
    url: 'https://github.com/datalab-to/surya',
  },
  {
    model: 'Tesseract 5',
    count: 100,
    notes: 'Classical LSTM-based OCR. No layout / structure recovery.',
    color: '#64748b',
    url: 'https://github.com/tesseract-ocr/tesseract',
  },
  {
    model: 'HunyuanOCR',
    count: 14,
    notes: '14+ source languages for translation; broader recognition.',
    color: '#0d9488',
    url: 'https://huggingface.co/tencent/HunyuanOCR',
  },
  {
    model: 'Frontier APIs (GPT-4o / Gemini)',
    count: 100,
    notes: 'Effectively all major languages; multilingual handled implicitly.',
    color: '#7c3aed',
    url: null,
  },
];

// Selected 20-language subset from Chandra 2's 90-language benchmark
// Source: https://github.com/datalab-to/chandra/blob/master/FULL_BENCHMARKS.md
const CHANDRA_LANG_BENCHMARK = [
  // High-resource Latin / Western
  { code: 'en', name: 'English',     chandra: 96.6, gemini: 90.3, family: 'Latin', color: '#2563eb' },
  { code: 'de', name: 'German',      chandra: 94.8, gemini: 88.3, family: 'Latin', color: '#2563eb' },
  { code: 'fr', name: 'French',      chandra: 93.7, gemini: 86.1, family: 'Latin', color: '#2563eb' },
  { code: 'es', name: 'Spanish',     chandra: 89.3, gemini: 86.8, family: 'Latin', color: '#2563eb' },
  { code: 'pt', name: 'Portuguese',  chandra: 95.2, gemini: 89.4, family: 'Latin', color: '#2563eb' },
  // Slavic / Cyrillic
  { code: 'pl', name: 'Polish',      chandra: 91.5, gemini: 91.1, family: 'Cyrillic', color: '#7c3aed' },
  { code: 'ru', name: 'Russian',     chandra: 85.5, gemini: 82.8, family: 'Cyrillic', color: '#7c3aed' },
  { code: 'uk', name: 'Ukrainian',   chandra: 91.0, gemini: 87.9, family: 'Cyrillic', color: '#7c3aed' },
  // CJK
  { code: 'zh', name: 'Chinese',     chandra: 88.7, gemini: 70.0, family: 'CJK', color: '#dc2626' },
  { code: 'ja', name: 'Japanese',    chandra: 86.9, gemini: 80.0, family: 'CJK', color: '#dc2626' },
  { code: 'ko', name: 'Korean',      chandra: 81.5, gemini: 84.8, family: 'CJK', color: '#dc2626' },
  // Indic / South Asian
  { code: 'hi', name: 'Hindi',       chandra: 78.4, gemini: 82.7, family: 'Indic', color: '#d97706' },
  { code: 'bn', name: 'Bengali',     chandra: 72.8, gemini: 55.3, family: 'Indic', color: '#d97706' },
  { code: 'ta', name: 'Tamil',       chandra: 77.7, gemini: 53.9, family: 'Indic', color: '#d97706' },
  // Middle East / RTL
  { code: 'ar', name: 'Arabic',      chandra: 68.4, gemini: 84.4, family: 'Arabic', color: '#0d9488' },
  { code: 'fa', name: 'Persian',     chandra: 75.1, gemini: 61.8, family: 'Arabic', color: '#0d9488' },
  { code: 'he', name: 'Hebrew',      chandra: 70.4, gemini: 50.9, family: 'Arabic', color: '#0d9488' },
  // Low-resource — Chandra dramatically wins
  { code: 'am', name: 'Amharic',     chandra: 34.4, gemini: 0.5,  family: 'Low-res', color: '#16a34a' },
  { code: 'km', name: 'Khmer',       chandra: 46.1, gemini: 6.3,  family: 'Low-res', color: '#16a34a' },
  { code: 'my', name: 'Burmese',     chandra: 55.9, gemini: 15.8, family: 'Low-res', color: '#16a34a' },
];

function LanguageBenchmarkChart() {
  // Group by family for visual organization
  const families = ['Latin', 'Cyrillic', 'CJK', 'Indic', 'Arabic', 'Low-res'];
  return (
    <div className="space-y-1">
      {families.map(family => {
        const langs = CHANDRA_LANG_BENCHMARK.filter(l => l.family === family);
        const familyColor = langs[0]?.color;
        return (
          <div key={family} className="space-y-0.5">
            <div className="text-xs font-bold uppercase tracking-wider opacity-70 pt-2 first:pt-0"
              style={{ color: familyColor }}>
              {family === 'Low-res' ? 'Lower-resource (where Chandra dramatically wins)' : family}
            </div>
            {langs.map(l => (
              <div key={l.code} className="grid grid-cols-[100px_1fr_1fr] gap-2 items-center text-xs">
                <div className="font-medium truncate" title={l.name}>
                  <span className="font-mono text-muted-foreground mr-1.5">{l.code}</span>
                  {l.name}
                </div>
                {/* Chandra bar */}
                <div className="rounded-full overflow-hidden flex items-center" style={{ background: '#f1f5f9', height: 16 }}>
                  <div className="h-full rounded-full flex items-center justify-end pr-1.5 transition-all duration-500"
                    style={{ width: `${l.chandra}%`, background: '#16a34a', minWidth: 28 }}
                  >
                    <span className="text-xs font-bold text-white">{l.chandra.toFixed(0)}</span>
                  </div>
                </div>
                {/* Gemini bar */}
                <div className="rounded-full overflow-hidden flex items-center" style={{ background: '#f1f5f9', height: 16 }}>
                  <div className="h-full rounded-full flex items-center justify-end pr-1.5 transition-all duration-500"
                    style={{ width: `${l.gemini}%`, background: '#2563eb', minWidth: 28 }}
                  >
                    <span className="text-xs font-bold text-white">{l.gemini.toFixed(0)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function ModelSpotlight({ m }) {
  return (
    <div className="rounded-2xl border overflow-hidden flex flex-col h-full"
      style={{ borderColor: m.color + '55' }}
    >
      {/* Header */}
      <div className="px-4 py-3 space-y-1" style={{ background: m.color + '10' }}>
        <div className="flex items-baseline justify-between gap-2">
          <div className="text-base font-bold" style={{ color: m.color }}>{m.name}</div>
          <div className="text-xs text-muted-foreground whitespace-nowrap">{m.released}</div>
        </div>
        <div className="text-xs text-muted-foreground">
          {m.org} · <span className="font-mono">{m.params}</span>
        </div>
      </div>

      <div className="p-4 space-y-3 flex-1">
        <div className="text-sm font-semibold leading-snug" style={{ color: m.color }}>
          {m.headline}
        </div>

        <div className="space-y-1.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Key techniques</div>
          {m.technique.map((t, i) => (
            <div key={i} className="text-xs flex items-start gap-1.5">
              <span className="font-bold flex-shrink-0" style={{ color: m.color }}>→</span>
              <span className="opacity-85 leading-relaxed">{t}</span>
            </div>
          ))}
        </div>

        <div className="rounded-lg p-2.5 text-xs space-y-0.5" style={{ background: m.color + '08' }}>
          <div className="font-semibold" style={{ color: m.color }}>{m.benchmark}</div>
          <div className="opacity-80">{m.bestFor}</div>
        </div>

        <div className="text-xs italic opacity-75 leading-relaxed">
          <strong>Connects to:</strong> {m.chapterTieIn}
        </div>

        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs pt-1 border-t" style={{ borderColor: m.color + '22' }}>
          {(m.links || []).map(l => (
            <a key={l.url} href={l.url} target="_blank" rel="noopener noreferrer"
              className="hover:underline transition-opacity"
              style={{ color: m.color, opacity: 0.85 }}
            >
              ↗ {l.label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

const FORMULA_EXAMPLES = [
  {
    cls: 'Text',
    color: '#86efac',
    text: 'B(J/ψ→pp̄ω) = (1.10 ± 0.15)·10⁻³, which combined with the PDG J/ψ total width gives an experimental partial width of',
  },
  {
    cls: 'Formula',
    color: '#fca5a5',
    text: '\\(\\Gamma(J/\\psi\\rightarrow p\\bar{p}\\omega)=103\\pm 14\\,\\mathrm{eV}.\\)  (31)',
  },
  {
    cls: 'Text',
    color: '#86efac',
    text: 'The fact that this is approximately equal to the pp̄π⁰ partial width despite the much smaller phase space suggests a robust NNω coupling.',
  },
  {
    cls: 'Formula',
    color: '#fca5a5',
    text: '\\(\\Gamma(J/\\psi\\rightarrow p\\bar{p}\\omega)=\\alpha_\\omega\\cdot\\left(2.468-1.101\\kappa_\\omega+0.886\\kappa_\\omega^2\\right)\\mathrm{eV}.\\)  (32)',
  },
  {
    cls: 'Picture',
    color: '#93c5fd',
    text: '[Figure 5 — density plot, bounding box: (0.52, 0.41) → (0.75, 0.52)]',
  },
  {
    cls: 'Caption',
    color: '#fde68a',
    text: 'FIG. 5: Theoretical J/ψ→pp̄ω DP event density along the diagonal M²_pω = M²_p̄ω, showing the strong κω dependence.',
  },
  {
    cls: 'Section-header',
    color: '#d8b4fe',
    text: '## G. Other Ψ→pp̄V decays',
  },
];

function LandscapePage() {
  return (
    <div className="space-y-6">

      {/* Intro */}
      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-6 md:p-8 space-y-2">
          <p className="text-sm opacity-85 leading-relaxed">
            Five models from five different organisations, each demonstrating a technique we just walked through.
            Together they show the field's rapid 2024–2026 trajectory: smaller, faster, more capable, and increasingly open.
          </p>
          <p className="text-sm opacity-85 leading-relaxed">
            Note how the techniques mix-and-match — pseudo-labeling, synthetic data, RL with verifiable rewards, VRFM,
            diffusion decoding. <strong>No single technique wins.</strong> The best models combine several.
          </p>
        </CardContent>
      </Card>

      {/* Benchmarks Card */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">How do we know which model is best? Benchmarks.</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            End-to-end document parsing is genuinely hard to evaluate. A model can be great at tables and weak on
            formulas. Great at formatting and weak at reading order. Older benchmarks measured text similarity (edit
            distance) on flat strings — but as we just saw, that punishes order shifts as harshly as content errors,
            and ignores structure entirely. The 2024–2026 wave of benchmarks tackles this head-on with
            <strong> per-subcategory scoring</strong> and <strong>rule-based semantic correctness checks</strong>.
          </p>

          {/* Benchmark comparison table */}
          <div className="rounded-xl border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b" style={{ background: '#f8fafc' }}>
                  <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Benchmark</th>
                  <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Year / scale</th>
                  <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Subcategories</th>
                  <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Metric type</th>
                  <th className="text-left py-2.5 px-3 font-semibold text-muted-foreground uppercase text-xs tracking-wider">Current leader</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                <tr>
                  <td className="py-3 px-3">
                    <a href="https://github.com/opendatalab/OmniDocBench" target="_blank" rel="noopener noreferrer"
                      className="font-bold hover:underline" style={{ color: '#2563eb' }}
                    >
                      OmniDocBench v1.5/v1.6 ↗
                    </a>
                    <div className="text-muted-foreground">
                      Shanghai AI Lab · <a href="https://arxiv.org/abs/2412.07626" target="_blank" rel="noopener noreferrer" className="hover:underline">CVPR 2025</a>
                    </div>
                  </td>
                  <td className="py-3 px-3 font-mono">2024–26<br/>1,651 pages</td>
                  <td className="py-3 px-3">Text · Tables · Formulas · Reading order · 28 block / 4 span categories · 10 doc types · 5 languages</td>
                  <td className="py-3 px-3">
                    Continuous metrics:<br/>
                    <span className="font-mono">Norm. edit dist.</span> (text + reading order),<br/>
                    <span className="font-mono">TEDS</span> (tables),<br/>
                    <span className="font-mono">CDM</span> (formulas)
                  </td>
                  <td className="py-3 px-3 font-bold" style={{ color: '#2563eb' }}>PaddleOCR-VL-1.6<br/>96.33%</td>
                </tr>
                <tr style={{ background: '#fafafa' }}>
                  <td className="py-3 px-3">
                    <a href="https://huggingface.co/datasets/allenai/olmOCR-bench" target="_blank" rel="noopener noreferrer"
                      className="font-bold hover:underline" style={{ color: '#dc2626' }}
                    >
                      olmOCR-Bench ↗
                    </a>
                    <div className="text-muted-foreground">
                      Allen AI · <a href="https://olmocr.allenai.org/papers/olmocr.pdf" target="_blank" rel="noopener noreferrer" className="hover:underline">paper</a>
                    </div>
                  </td>
                  <td className="py-3 px-3 font-mono">2025<br/>1,402 PDFs</td>
                  <td className="py-3 px-3">Natural reading order · Table cell existence · Formula symbol positioning · Tiny fonts · Old scans · Headers/footers</td>
                  <td className="py-3 px-3">
                    Binary unit tests:<br/>
                    <span className="font-mono">pass/fail</span> on programmatic rules.<br/>
                    No LLM-as-judge.
                  </td>
                  <td className="py-3 px-3 font-bold" style={{ color: '#dc2626' }}>Chandra OCR 2<br/>85.9%</td>
                </tr>
                <tr>
                  <td className="py-3 px-3">
                    <a href="https://www.parsebench.ai/" target="_blank" rel="noopener noreferrer"
                      className="font-bold hover:underline" style={{ color: '#7c3aed' }}
                    >
                      ParseBench ↗
                    </a>
                    <div className="text-muted-foreground">
                      LlamaIndex · <a href="https://arxiv.org/abs/2604.08538" target="_blank" rel="noopener noreferrer" className="hover:underline">paper</a>
                    </div>
                  </td>
                  <td className="py-3 px-3 font-mono">2026<br/>2,078 pages<br/>169K rules</td>
                  <td className="py-3 px-3">Tables (GTRM) · Charts (data-point match) · Content faithfulness · Semantic formatting · Visual grounding</td>
                  <td className="py-3 px-3">
                    Rule-based semantic:<br/>
                    Structural table match,<br/>
                    chart data-point match,<br/>
                    rule binary tests
                  </td>
                  <td className="py-3 px-3 font-bold" style={{ color: '#7c3aed' }}>LlamaParse Agentic<br/>84.9%</td>
                </tr>
                <tr style={{ background: '#fafafa' }}>
                  <td className="py-3 px-3">
                    <a href="https://github.com/Yuliang-Liu/MultimodalOCR" target="_blank" rel="noopener noreferrer"
                      className="font-bold hover:underline" style={{ color: '#16a34a' }}
                    >
                      OCRBench v2 ↗
                    </a>
                    <div className="text-muted-foreground">2024–25</div>
                  </td>
                  <td className="py-3 px-3 font-mono">2024–25<br/>10,000+ examples</td>
                  <td className="py-3 px-3">23 task types across multilingual OCR, scene text, handwriting, structured</td>
                  <td className="py-3 px-3">
                    Mixed:<br/>
                    <span className="font-mono">TEDS, accuracy, F1</span><br/>per subtask
                  </td>
                  <td className="py-3 px-3 text-muted-foreground italic">Active leaderboard</td>
                </tr>
                <tr>
                  <td className="py-3 px-3">
                    <div className="font-bold" style={{ color: '#d97706' }}>Subtask benchmarks</div>
                    <div className="text-muted-foreground">2019–22</div>
                  </td>
                  <td className="py-3 px-3 font-mono">
                    <a href="https://github.com/ibm-aur-nlp/PubTabNet" target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: '#d97706' }}>PubTabNet ↗</a> (568K)<br/>
                    <a href="https://developer.ibm.com/exchanges/data/all/fintabnet/" target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: '#d97706' }}>FinTabNet ↗</a> (113K)<br/>
                    <a href="https://github.com/vis-nlp/ChartQA" target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: '#d97706' }}>ChartQA ↗</a> (21K)<br/>
                    <a href="https://www.docvqa.org/" target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: '#d97706' }}>DocVQA ↗</a> (50K)
                  </td>
                  <td className="py-3 px-3">Each tests one narrow capability — table structure, chart QA, document VQA</td>
                  <td className="py-3 px-3">
                    Per-task metrics:<br/>
                    <span className="font-mono">TEDS, ANLS,</span><br/><span className="font-mono">relaxed accuracy</span>
                  </td>
                  <td className="py-3 px-3 text-muted-foreground italic">Largely saturated</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Why end-to-end is hard */}
          <div className="grid md:grid-cols-2 gap-4">
            <InfoBox color="#dc2626" title="Why end-to-end is so hard to score">
              <ul className="text-xs space-y-1.5 list-none">
                <li>
                  <strong>Subcategories pull different ways:</strong> a model can be SOTA on tables and middling on formulas.
                  A single overall score hides this.
                </li>
                <li>
                  <strong>Edit distance is a bad proxy:</strong> punishes a model that reordered two correct paragraphs as
                  severely as one that hallucinated both.
                </li>
                <li>
                  <strong>LLM-as-judge is unstable:</strong> evaluator preferences drift over time, and judge models
                  often share biases with the models being evaluated.
                </li>
                <li>
                  <strong>Domain coverage matters:</strong> academic papers, invoices, and government PDFs are very
                  different distributions. A benchmark on one tells you little about the others.
                </li>
              </ul>
            </InfoBox>
            <InfoBox color="#16a34a" title="What the new benchmarks fixed">
              <ul className="text-xs space-y-1.5 list-none">
                <li>
                  <strong>Per-dimension scores:</strong> ParseBench reports 5 axes separately. OmniDocBench breaks
                  down by document type and content category.
                </li>
                <li>
                  <strong>Structural matching for tables:</strong> TEDS and GriTS compare tree edit distance,
                  not flat-text similarity. Cell content + structure both count.
                </li>
                <li>
                  <strong>Binary unit tests:</strong> olmOCR-Bench and ParseBench use programmatic pass/fail tests
                  that reward content correctness even when ordering is imperfect.
                </li>
                <li>
                  <strong>Reading order as its own metric:</strong> OmniDocBench scores reading order separately
                  from content — preventing the CE-style "wrong-order = total failure" trap.
                </li>
              </ul>
            </InfoBox>
          </div>

          <InfoBox color="#7c3aed" title="The headline insight">
            <span className="text-xs">
              No model in the wild is consistently strong on all five dimensions of ParseBench. LlamaParse Agentic
              leads overall at 84.9% but per-dimension breakdowns show <strong>clear capability tradeoffs</strong>.
              When picking a model, do not pick by overall benchmark score — pick by the subcategory that matches
              your workload.
            </span>
          </InfoBox>
        </CardContent>
      </Card>

      {/* Multilingual capability */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">Multilingual capability — how wide do these models go?</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            Training corpora are heavily skewed to English and CJK. Many "multilingual" models score well on top-20
            languages and collapse on the long tail — Amharic, Khmer, Burmese, Sinhala, low-resource African and
            South-East Asian scripts. The 2025–2026 wave of doc models has put serious work into closing this gap,
            and the benchmarks now let us see exactly where.
          </p>

          {/* Coverage at a glance */}
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Language coverage by model
            </div>
            <div className="rounded-xl border overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b" style={{ background: '#f8fafc' }}>
                    <th className="text-left py-2 px-3 font-semibold uppercase text-muted-foreground tracking-wider">Model</th>
                    <th className="text-right py-2 px-3 font-semibold uppercase text-muted-foreground tracking-wider">Languages</th>
                    <th className="text-left py-2 px-3 font-semibold uppercase text-muted-foreground tracking-wider">Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {MULTILINGUAL_COVERAGE.map(c => (
                    <tr key={c.model}>
                      <td className="py-2 px-3 font-semibold">
                        {c.url ? (
                          <a href={c.url} target="_blank" rel="noopener noreferrer"
                            className="hover:underline" style={{ color: c.color }}
                          >
                            {c.model} ↗
                          </a>
                        ) : (
                          <span style={{ color: c.color }}>{c.model}</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-right font-mono font-bold" style={{ color: c.color }}>
                        {c.count}+
                      </td>
                      <td className="py-2 px-3 text-muted-foreground">{c.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Counts approximate "claimed support." Actual quality varies dramatically — see the benchmark below.
            </div>
          </div>

          {/* Chandra 2 vs Gemini 2.5 Flash — 90-language head-to-head */}
          <div>
            <div className="flex items-baseline justify-between flex-wrap gap-2 mb-3">
              <div>
                <div className="text-sm font-bold">Head-to-head — Chandra 2 vs Gemini 2.5 Flash (selected 20 of 90)</div>
                <div className="text-xs text-muted-foreground">
                  Source: <a href="https://github.com/datalab-to/chandra/blob/master/FULL_BENCHMARKS.md" target="_blank" rel="noopener noreferrer" className="underline">Chandra full 90-language benchmark ↗</a> —
                  measured on internal multilingual OCR set
                </div>
              </div>
              <div className="flex gap-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm" style={{ background: '#16a34a' }} />
                  <span>Chandra 2</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm" style={{ background: '#2563eb' }} />
                  <span>Gemini 2.5 Flash</span>
                </div>
              </div>
            </div>
            <div className="rounded-xl border p-4">
              <LanguageBenchmarkChart />
            </div>
            <div className="text-xs text-muted-foreground mt-3 leading-relaxed">
              <strong>Full-set averages:</strong>{' '}
              <span style={{ color: '#16a34a', fontWeight: 700 }}>Chandra 2 = 72.7% ± 1.2%</span> ·{' '}
              <span style={{ color: '#2563eb', fontWeight: 700 }}>Gemini 2.5 Flash = 60.8% ± 1.3%</span>{' '}
              across all 90 languages. The biggest gaps are in low-resource scripts where frontier models score
              near-zero (Amharic, Khmer, Lao, Myanmar) — exactly where specialized OCR training data matters most.
            </div>
          </div>

          {/* HunyuanOCR text-image translation callout */}
          <div className="rounded-2xl border" style={{ borderColor: '#0d948844' }}>
            <div className="px-4 py-2.5 text-sm font-bold flex items-center gap-2"
              style={{ background: '#0d948812', color: '#0d9488' }}
            >
              <span>🌐</span> HunyuanOCR — multilingual goes beyond recognition, into translation
            </div>
            <div className="p-4 grid md:grid-cols-[1.1fr_1fr] gap-4 text-xs">
              <div className="space-y-2.5">
                <p className="opacity-85 leading-relaxed">
                  Most doc OCR models stop at <em>recognising</em> text in a language. HunyuanOCR (1B, Tencent, Nov 2025)
                  adds <strong>text-image translation</strong> as a first-class task. Feed in a scanned page or a street-sign
                  photo in 14+ source languages — Japanese, Korean, French, German, etc. — and get back the same
                  document <em>translated</em> into Chinese or English, with structure preserved (tables stay tables,
                  formulas stay LaTeX).
                </p>
                <p className="opacity-85 leading-relaxed">
                  This took first place in the{' '}
                  <a href="https://arxiv.org/abs/2511.19575" target="_blank" rel="noopener noreferrer"
                    className="underline" style={{ color: '#0d9488' }}>
                    ICDAR 2025 DIMT Challenge (Small Model Track) ↗
                  </a>
                  . It's also the first doc-focused VLM to demonstrate clear RL gains on OCR — closing the loop
                  on the ch5 RL-with-verifiable-rewards story.
                </p>
                <div className="flex flex-wrap gap-2 pt-1">
                  <a href="https://huggingface.co/tencent/HunyuanOCR" target="_blank" rel="noopener noreferrer"
                    className="rounded-full px-3 py-1 text-xs font-semibold hover:opacity-100 opacity-90"
                    style={{ background: '#0d9488', color: 'white' }}
                  >
                    ↗ Hugging Face
                  </a>
                  <a href="https://github.com/Tencent-Hunyuan/HunyuanOCR" target="_blank" rel="noopener noreferrer"
                    className="rounded-full px-3 py-1 text-xs font-semibold hover:opacity-100 opacity-90"
                    style={{ background: '#0d9488', color: 'white' }}
                  >
                    ↗ GitHub
                  </a>
                  <a href="https://arxiv.org/abs/2511.19575" target="_blank" rel="noopener noreferrer"
                    className="rounded-full px-3 py-1 text-xs font-semibold hover:opacity-100 opacity-90"
                    style={{ background: '#0d9488', color: 'white' }}
                  >
                    ↗ Tech report
                  </a>
                </div>
              </div>

              {/* Prompt examples */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Two prompt paradigms
                </div>
                <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
                  <div className="text-xs font-mono leading-relaxed" style={{ color: '#7dd3fc' }}># General-purpose</div>
                  <div className="text-xs font-mono leading-relaxed" style={{ color: '#e2e8f0' }}>
                    "Extract all text from the image<br />
                    and translate it into English."
                  </div>
                </div>
                <div className="rounded-lg p-3 space-y-1.5" style={{ background: '#0f172a' }}>
                  <div className="text-xs font-mono leading-relaxed" style={{ color: '#7dd3fc' }}># Document-oriented</div>
                  <div className="text-xs font-mono leading-relaxed" style={{ color: '#e2e8f0' }}>
                    "First parse the document, then<br />
                    translate its content into Chinese.<br />
                    Represent equations in LaTeX;<br />
                    render tables in HTML format."
                  </div>
                </div>
              </div>
            </div>
          </div>

          <InfoBox color="#7c3aed" title="The practitioner's takeaway">
            <span className="text-xs">
              "Supports 90+ languages" on a model card means little. The Chandra benchmark shows that the same model
              can score 96% on English and 13% on Pashto. <strong>Always check the per-language scores on the long tail
              of languages you actually care about</strong> — and if you need translation, not just recognition, HunyuanOCR
              is the only model in the current open-source landscape designed for it as a first-class task.
            </span>
          </InfoBox>
        </CardContent>
      </Card>

      {/* Model spotlight grid */}
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
        {LANDSCAPE_MODELS.map(m => (
          <ModelSpotlight key={m.id} m={m} />
        ))}
      </div>

      {/* Closing — formula demo (what these techniques unlock) */}
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg">What all of this unlocks — a real example</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0 space-y-4">
          <p className="text-sm opacity-85 leading-relaxed">
            A two-column physics preprint with dense inline equations, density plots, and figure captions —
            historically a hostile input for any OCR pipeline. With layout-aware VLMs trained on the techniques
            we just covered, here's what gets extracted:
          </p>

          <div className="grid xl:grid-cols-[1fr_1fr] gap-6">
            {/* Left: input doc */}
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Input — physics preprint (two-column, dense formulas)
              </div>
              <div className="rounded-xl border overflow-hidden">
                <img
                  src={IMG_CLEAN}
                  alt="Physics preprint input"
                  style={{ width: '100%', height: 380, objectFit: 'cover', objectPosition: 'top center', display: 'block' }}
                />
              </div>
            </div>

            {/* Right: extraction with class labels */}
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Output — text + LaTeX + class labels + bounding boxes
              </div>
              <div className="rounded-xl border overflow-auto" style={{ background: '#0f172a', maxHeight: 380 }}>
                <div className="p-3 space-y-2 font-mono text-xs">
                  {FORMULA_EXAMPLES.map((item, i) => (
                    <div key={i} className="space-y-0.5">
                      <span className="rounded px-1.5 py-0.5 text-xs font-bold inline-block"
                        style={{ background: item.color + '22', color: item.color }}
                      >
                        {item.cls}
                      </span>
                      <div className="pl-2 text-xs leading-relaxed"
                        style={{
                          color: item.cls === 'Formula' ? '#fca5a5'
                               : item.cls === 'Picture' ? '#93c5fd'
                               : item.cls === 'Caption' ? '#fde68a'
                               : item.cls === 'Section-header' ? '#d8b4fe'
                               : '#e2e8f0'
                        }}
                      >
                        {item.text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <InfoBox color="#16a34a" title="The whole stack at work">
            <span className="text-xs">
              Inline formulas are recovered as compileable LaTeX (made possible by synthetic LaTeX training data, ch5).
              Region classes and bounding boxes are predicted by layout-aware architecture (VRFM-style coarse-to-fine, ch6).
              Reading order is preserved across the two-column layout (RL on verifiable structural rewards, ch5).
              And the whole page processes in seconds, not minutes (diffusion decoding, ch6).
            </span>
          </InfoBox>
        </CardContent>
      </Card>

      {/* Closing thought */}
      <Card className="rounded-3xl shadow-sm">
        <CardContent className="p-6 md:p-8">
          <h3 className="text-base font-bold mb-2">Where the field is heading</h3>
          <ul className="space-y-2 text-sm opacity-85 leading-relaxed">
            <li>
              <strong>Smaller models, more compute on inference-time tricks.</strong> 0.9B and sub-1B models matching
              frontier API quality. The trend continues.
            </li>
            <li>
              <strong>Pseudo-labeling and synthetic data become commodities.</strong> Every serious player has a pipeline.
              The differentiation is in filtering and curation, not raw scale.
            </li>
            <li>
              <strong>RL with verifiable rewards extends beyond OCR.</strong> Document parsing was the canary — the same
              recipe (synthetic data + programmatic tests + GRPO) is moving into structured extraction more broadly.
            </li>
            <li>
              <strong>Diffusion decoding is starting to ship.</strong> 3–5× speedups without accuracy loss are too
              valuable to leave on the table. Expect more frontier doc models to adopt block-diffusion variants.
            </li>
          </ul>
        </CardContent>
      </Card>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHELL — Navigation + page transitions
// ═══════════════════════════════════════════════════════════════════════════════

export default function DocumentParsingVisualizer() {
  const [page, setPage] = useState(0);

  const pageComponent = useMemo(() => {
    switch (page) {
      case 0: return <TasksPage />;
      case 1: return <OcrPage />;
      case 2: return <VlmPage />;
      case 3: return <FailuresPage />;
      case 4: return <TrainingPage />;
      case 5: return <InferencePage />;
      case 6: return <LandscapePage />;
      default: return null;
    }
  }, [page]);

  return (
    <div className="min-h-screen bg-white p-4 text-black dark:bg-neutral-950 dark:text-white md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Title */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
            Document Parsing
          </h1>
          <p className="mt-1 text-sm opacity-60">
            From OCR pipelines to vision-language models · San Diego ML Meetup 2026
          </p>
        </div>

        {/* Navigation card */}
        <Card className="rounded-3xl shadow-sm">
          <CardContent className="flex items-center justify-between gap-4 p-4">
            <Button variant="outline" disabled={page === 0} onClick={() => setPage(p => p - 1)}>
              ← Previous
            </Button>
            <div className="text-center min-w-0 flex-1">
              <div className="font-semibold text-sm truncate">{pages[page].title}</div>
              <div className="text-xs text-muted-foreground truncate">{pages[page].subtitle}</div>
            </div>
            <Button disabled={page === pages.length - 1} onClick={() => setPage(p => p + 1)}>
              Next →
            </Button>
          </CardContent>
        </Card>

        {/* Progress dots */}
        <div className="flex justify-center gap-2 flex-wrap">
          {pages.map((p, i) => (
            <button
              key={p.id}
              onClick={() => setPage(i)}
              title={p.title}
              className="rounded-full transition-all text-xs font-medium"
              style={{
                background: page === i ? '#0a0a0a' : '#e5e5e5',
                color: page === i ? 'white' : '#737373',
                padding: page === i ? '4px 12px' : '4px 8px',
                minWidth: page === i ? 80 : 24,
              }}
            >
              {page === i ? `${i + 1}. ${p.title.split(' ').slice(0, 3).join(' ')}…` : i + 1}
            </button>
          ))}
        </div>

        {/* Animated page content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={pages[page].id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.22 }}
          >
            {pageComponent}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
