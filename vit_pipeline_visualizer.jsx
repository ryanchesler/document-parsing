import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

// ═══════════════════════════════════════════════════════════════════════════════
// A VISION TRANSFORMER, SHAPE BY SHAPE
// A variation of the document-parsing talk site, focused ENTIRELY on the tensor
// shapes and transformations from raw pixels → ViT → 2×2 compression → text decoder.
// ═══════════════════════════════════════════════════════════════════════════════

const pages = [
  { id: 'overview', title: 'The whole pipeline',        subtitle: 'Pixels → patches → ViT → compress → decoder, one glance' },
  { id: 'patchify', title: 'Patchify the image',        subtitle: 'How a 2D image becomes a sequence of patch vectors' },
  { id: 'embed',    title: 'Patch & position embedding', subtitle: 'The Conv2d projection and where "order" comes from' },
  { id: 'encoder',  title: 'Inside the ViT encoder',     subtitle: 'Attention + MLP shapes through one transformer block' },
  { id: 'compress', title: '2×2 visual-token compression', subtitle: 'Pixel-shuffle / patch-merge: 4 tokens → 1' },
  { id: 'decoder',  title: 'Into the text decoder',       subtitle: 'Projector, sequence assembly, autoregressive decode' },
  { id: 'playground', title: 'Shape playground',          subtitle: 'Turn every knob, watch every tensor recompute' },
];

// ─── Palette ────────────────────────────────────────────────────────────────
const C_PIX   = '#0ea5e9'; // sky      — pixels / image
const C_PATCH = '#7c3aed'; // violet   — patches / visual tokens
const C_EMB   = '#2563eb'; // blue     — embeddings
const C_ATTN  = '#d97706'; // amber    — attention
const C_COMP  = '#0d9488'; // teal     — compression
const C_TEXT  = '#16a34a'; // green    — text / decoder
const C_GREY  = '#64748b';

// ─── Live shape engine ────────────────────────────────────────────────────────
// Everything downstream is a pure function of this config, so every chapter shows
// numbers that agree with each other and update the instant a knob moves.

function derive(cfg) {
  const { P, grid, C, D, heads, layers, mlpRatio, merge, Dllm, textTokens, cls } = cfg;

  const H = grid * P;                 // image side length (square)
  const W = H;
  const N = grid * grid;              // number of patches
  const patchDim = C * P * P;         // flattened patch dimension
  const headDim = D / heads;          // per-head channel width
  const seqIn = N + (cls ? 1 : 0);    // encoder sequence length (with optional CLS)

  const mergedGrid = Math.floor(grid / merge);
  const Ncomp = mergedGrid * mergedGrid;   // visual tokens after 2×2 merge
  const mergedDim = D * merge * merge;      // concatenated channels before projection
  const visualTokens = Ncomp;
  const seqLen = visualTokens + textTokens; // what the LLM actually sees

  // Parameter estimates (approximate, bias terms folded in loosely)
  const patchEmbedParams = patchDim * D + D;
  const posParams = seqIn * D;
  const perLayer = 4 * D * D + 2 * mlpRatio * D * D; // qkvo + up/down MLP
  const encoderParams = layers * perLayer;
  const projectorParams = mergedDim * Dllm + Dllm * Dllm; // 2-layer MLP connector
  const vitTotal = patchEmbedParams + posParams + encoderParams;

  // FLOP estimate for one encoder forward pass (multiply-adds ×2), in GFLOPs
  const flForward = layers * (
    2 * seqIn * (4 * D * D)          // qkvo projections
    + 2 * 2 * seqIn * seqIn * D      // scores + weighted sum
    + 2 * seqIn * (2 * mlpRatio * D * D) // MLP
  );

  return {
    H, W, N, patchDim, headDim, seqIn,
    mergedGrid, Ncomp, mergedDim, visualTokens, seqLen,
    compressRatio: N / Ncomp,
    patchEmbedParams, posParams, encoderParams, projectorParams, vitTotal,
    gflops: flForward / 1e9,
  };
}

// ─── Presets (internally consistent; "inspired by" real models) ─────────────────
const PRESETS = {
  qwen: {
    label: 'Qwen2-VL-style', tag: '448 · p14 · merge 2',
    cfg: { P: 14, grid: 32, C: 3, D: 1280, heads: 16, layers: 32, mlpRatio: 4, merge: 2, Dllm: 3584, textTokens: 20, cls: false },
  },
  internvl: {
    label: 'InternVL-style', tag: '448 · p14 · pixelshuffle',
    cfg: { P: 14, grid: 32, C: 3, D: 1024, heads: 16, layers: 24, mlpRatio: 4, merge: 2, Dllm: 4096, textTokens: 20, cls: true },
  },
  siglip: {
    label: 'SigLIP-so400m-style', tag: '378 · p14 · no merge',
    cfg: { P: 14, grid: 27, C: 3, D: 1152, heads: 16, layers: 27, mlpRatio: 4, merge: 1, Dllm: 3584, textTokens: 20, cls: false },
  },
  tiny: {
    label: 'Tiny (teaching)', tag: '112 · p16 · merge 2',
    cfg: { P: 16, grid: 7, C: 3, D: 192, heads: 6, layers: 4, mlpRatio: 4, merge: 1, Dllm: 1024, textTokens: 8, cls: true },
  },
};

// ─── Shared UI primitives ───────────────────────────────────────────────────────

function SectionLabel({ color = C_EMB, children }) {
  return (
    <span className="inline-block rounded-full px-2 py-0.5 text-xs font-semibold uppercase tracking-wider text-white mb-2" style={{ background: color }}>
      {children}
    </span>
  );
}

function InfoBox({ color = C_EMB, title, children }) {
  return (
    <div className="rounded-xl border-l-4 p-4" style={{ borderColor: color, background: color + '10' }}>
      {title && <div className="font-semibold mb-1" style={{ color }}>{title}</div>}
      <div className="text-sm opacity-85 leading-relaxed">{children}</div>
    </div>
  );
}

// A tensor-shape pill: [ a × b × c ] with an optional caption above it.
function Shape({ dims, color = C_PATCH, label, sub, big }) {
  return (
    <div className="inline-flex flex-col items-center gap-1">
      {label && <span className="text-[11px] font-semibold uppercase tracking-wide" style={{ color }}>{label}</span>}
      <span
        className="font-mono font-bold rounded-lg border-2 whitespace-nowrap"
        style={{
          color, borderColor: color, background: color + '12',
          padding: big ? '8px 14px' : '4px 10px',
          fontSize: big ? 18 : 13,
        }}
      >
        [ {dims.map((d, i) => (
          <span key={i}>
            {i > 0 && <span style={{ opacity: 0.4 }}> × </span>}
            {typeof d === 'number' ? d.toLocaleString() : d}
          </span>
        ))} ]
      </span>
      {sub && <span className="text-[10px] text-muted-foreground">{sub}</span>}
    </div>
  );
}

function Stat({ label, value, color = C_GREY, mono }) {
  return (
    <div className="rounded-xl border p-3 text-center" style={{ borderColor: color + '40', background: color + '08' }}>
      <div className={`text-lg font-bold ${mono ? 'font-mono' : ''}`} style={{ color }}>{value}</div>
      <div className="text-[11px] text-muted-foreground mt-0.5 leading-tight">{label}</div>
    </div>
  );
}

function Seg({ options, value, onChange }) {
  return (
    <div className="inline-flex rounded-lg border overflow-hidden">
      {options.map(o => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className="px-3 py-1.5 text-xs font-medium transition-colors"
          style={{
            background: value === o.value ? '#0a0a0a' : 'transparent',
            color: value === o.value ? 'white' : '#525252',
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Slider({ label, value, min, max, step = 1, onChange, suffix }) {
  return (
    <label className="block">
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="font-medium text-muted-foreground">{label}</span>
        <span className="font-mono font-bold">{value}{suffix}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-black cursor-pointer"
      />
    </label>
  );
}

function Arrow({ label, sub, color = C_GREY, vertical }) {
  if (vertical) {
    return (
      <div className="flex flex-col items-center py-1">
        <span className="text-[10px] font-semibold" style={{ color }}>{label}</span>
        <span style={{ color, fontSize: 20, lineHeight: 1 }}>↓</span>
        {sub && <span className="text-[9px] text-muted-foreground">{sub}</span>}
      </div>
    );
  }
  return (
    <div className="flex flex-col items-center px-1">
      <span className="text-[10px] font-semibold text-center leading-tight" style={{ color }}>{label}</span>
      <span style={{ color, fontSize: 22, lineHeight: 1 }}>→</span>
      {sub && <span className="text-[9px] text-muted-foreground">{sub}</span>}
    </div>
  );
}

// ─── The compact preset bar shown on every chapter ─────────────────────────────

function PresetBar({ presetKey, setPreset, d, cfg }) {
  return (
    <Card className="rounded-3xl shadow-sm">
      <CardContent className="p-4 flex flex-wrap items-center gap-3 justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mr-1">Config</span>
          {Object.entries(PRESETS).map(([k, p]) => (
            <button
              key={k}
              onClick={() => setPreset(k)}
              className="rounded-lg border px-3 py-1.5 text-xs font-medium transition-all"
              style={{
                borderColor: presetKey === k ? C_PATCH : '#e5e5e5',
                background: presetKey === k ? C_PATCH + '12' : 'transparent',
                color: presetKey === k ? C_PATCH : '#525252',
              }}
              title={p.tag}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
          <span>{d.H}×{d.W}px</span>
          <span>p={cfg.P}</span>
          <span>{cfg.grid}×{cfg.grid} grid</span>
          <span>D={cfg.D}</span>
          <span className="font-bold" style={{ color: C_PATCH }}>{d.N} patches</span>
          <span className="font-bold" style={{ color: C_COMP }}>→ {d.visualTokens} tokens</span>
        </div>
      </CardContent>
    </Card>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 0 — Overview: the whole pipeline
// ═══════════════════════════════════════════════════════════════════════════════

function FlowStage({ color, title, shape, note, x }) {
  return (
    <div className="flex flex-col items-center text-center" style={{ minWidth: 118 }}>
      <div
        className="rounded-2xl border-2 w-full px-2 py-3 flex flex-col items-center gap-1"
        style={{ borderColor: color, background: color + '0e' }}
      >
        <div className="text-[11px] font-bold" style={{ color }}>{title}</div>
        <div className="font-mono text-[11px] font-bold" style={{ color }}>{shape}</div>
      </div>
      {note && <div className="text-[10px] text-muted-foreground mt-1 leading-tight max-w-[130px]">{note}</div>}
    </div>
  );
}

function OverviewPage({ d, cfg }) {
  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_PATCH}>The one-slide mental model</SectionLabel>
          <CardTitle className="text-xl">Every stage is just a reshape or a matmul</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            A vision-language model turns a picture into tokens the text model can read. Nothing mysterious happens —
            the image is <strong>chopped into a grid</strong>, each cell is <strong>projected to a vector</strong>, those
            vectors are <strong>mixed by self-attention</strong>, <strong>squeezed 4-to-1</strong>, and finally
            <strong> handed to the decoder</strong> like any other prompt tokens. Watch the shape travel:
          </p>

          {/* Horizontal flow */}
          <div className="overflow-x-auto pb-2">
            <div className="flex items-center gap-0 min-w-[880px]">
              <FlowStage color={C_PIX} title="Image" shape={`${d.H}×${d.W}×${cfg.C}`} note="raw RGB pixels" />
              <Arrow label="patchify" sub={`${cfg.P}×${cfg.P}`} color={C_PATCH} />
              <FlowStage color={C_PATCH} title="Patches" shape={`${d.N}×${d.patchDim}`} note={`${cfg.grid}×${cfg.grid} flattened cells`} />
              <Arrow label="linear" sub="→ D" color={C_EMB} />
              <FlowStage color={C_EMB} title="Embedded" shape={`${d.seqIn}×${cfg.D}`} note={cfg.cls ? '+CLS +pos' : '+pos emb'} />
              <Arrow label={`${cfg.layers}× blocks`} sub="attn+MLP" color={C_ATTN} />
              <FlowStage color={C_ATTN} title="Encoded" shape={`${d.seqIn}×${cfg.D}`} note="shape preserved" />
              <Arrow label="2×2 merge" sub="4→1" color={C_COMP} />
              <FlowStage color={C_COMP} title="Compressed" shape={`${d.visualTokens}×${cfg.D}`} note={`${d.compressRatio}× fewer tokens`} />
              <Arrow label="projector" sub={`→ ${cfg.Dllm}`} color={C_TEXT} />
              <FlowStage color={C_TEXT} title="LLM input" shape={`${d.seqLen}×${cfg.Dllm}`} note={`${d.visualTokens} vis + ${cfg.textTokens} text`} />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="patches from image" value={d.N.toLocaleString()} color={C_PATCH} mono />
            <Stat label="visual tokens after merge" value={d.visualTokens.toLocaleString()} color={C_COMP} mono />
            <Stat label="token reduction" value={`${d.compressRatio}×`} color={C_COMP} mono />
            <Stat label="tokens the decoder reads" value={d.seqLen.toLocaleString()} color={C_TEXT} mono />
          </div>

          <InfoBox color={C_PATCH} title="Why compression matters">
            The decoder's cost grows with sequence length (roughly quadratically in attention). Shrinking{' '}
            <span className="font-mono">{d.N.toLocaleString()}</span> patch tokens down to{' '}
            <span className="font-mono">{d.visualTokens.toLocaleString()}</span> visual tokens before the LLM is the
            single biggest lever for making high-resolution document VLMs affordable — the whole reason the 2×2 merge exists.
          </InfoBox>
        </CardContent>
      </Card>

      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2"><CardTitle className="text-lg">The three tensor "languages" you'll see</CardTitle></CardHeader>
        <CardContent className="p-6 pt-0 grid md:grid-cols-3 gap-4">
          {[
            { c: C_PIX, t: 'Spatial (image)', s: '[H × W × C]', d: 'A 2D grid of pixels with channels. Convolutions and patchify live here. Order is implicit in the (row, col) position.' },
            { c: C_PATCH, t: 'Sequence (tokens)', s: '[N × D]', d: 'A flat list of N vectors, each D-dimensional. This is what transformers eat. Order must be added back with positional embeddings.' },
            { c: C_ATTN, t: 'Multi-head (attention)', s: '[h × N × d_head]', d: 'The sequence split across h heads. Each head does its own N×N attention, then heads are concatenated back to [N × D].' },
          ].map((x, i) => (
            <div key={i} className="rounded-2xl border p-4 space-y-2" style={{ borderColor: x.c + '40' }}>
              <div className="font-semibold text-sm" style={{ color: x.c }}>{x.t}</div>
              <Shape dims={x.s.replace(/[[\]]/g, '').split('×').map(s => s.trim())} color={x.c} />
              <p className="text-xs opacity-75 leading-relaxed">{x.d}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 1 — Patchify
// ═══════════════════════════════════════════════════════════════════════════════

function PatchGrid({ grid, size = 300, highlight }) {
  const cell = size / grid;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="rounded-lg border" style={{ background: '#f8fafc', maxWidth: '100%' }}>
      {/* faux "image" gradient */}
      <defs>
        <linearGradient id="imgGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#bae6fd" />
          <stop offset="50%" stopColor="#c4b5fd" />
          <stop offset="100%" stopColor="#a7f3d0" />
        </linearGradient>
      </defs>
      <rect x={0} y={0} width={size} height={size} fill="url(#imgGrad)" />
      {Array.from({ length: grid + 1 }).map((_, i) => (
        <g key={i}>
          <line x1={i * cell} y1={0} x2={i * cell} y2={size} stroke={C_PATCH} strokeWidth={0.6} opacity={0.5} />
          <line x1={0} y1={i * cell} x2={size} y2={i * cell} stroke={C_PATCH} strokeWidth={0.6} opacity={0.5} />
        </g>
      ))}
      {highlight != null && (
        <rect
          x={(highlight % grid) * cell} y={Math.floor(highlight / grid) * cell}
          width={cell} height={cell}
          fill={C_PATCH} fillOpacity={0.35} stroke={C_PATCH} strokeWidth={2}
        />
      )}
    </svg>
  );
}

function PatchifyPage({ d, cfg }) {
  const [hi, setHi] = useState(0);
  const row = Math.floor(hi / cfg.grid);
  const col = hi % cfg.grid;

  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_PATCH}>Step 1 — patchify</SectionLabel>
          <CardTitle className="text-xl">From a 2D image to a sequence of flat vectors</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            A transformer has no notion of "2D". So the first job is to slice the image into a{' '}
            <strong>{cfg.grid}×{cfg.grid} grid</strong> of non-overlapping <strong>{cfg.P}×{cfg.P}</strong> patches,
            then flatten each patch into a single long vector. Click a cell to trace one patch through the reshape.
          </p>

          <div className="grid md:grid-cols-2 gap-6 items-start">
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Input image — {d.H}×{d.W}×{cfg.C}
              </div>
              <PatchGrid grid={cfg.grid} highlight={hi} />
              <input
                type="range" min={0} max={d.N - 1} value={hi}
                onChange={e => setHi(Number(e.target.value))}
                className="w-full accent-violet-600 cursor-pointer"
              />
              <div className="text-xs text-muted-foreground text-center">
                patch #{hi} → grid position (row {row}, col {col})
              </div>
            </div>

            <div className="space-y-3">
              <div className="rounded-xl border p-4 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">The reshape, one line at a time</div>
                <div className="flex flex-col gap-2 items-start">
                  <Shape label="1. image" color={C_PIX} dims={[d.H, d.W, cfg.C]} sub="height × width × channels" />
                  <Arrow vertical label="split into P×P blocks" color={C_PATCH} />
                  <Shape label="2. grid of patches" color={C_PATCH} dims={[cfg.grid, cfg.grid, cfg.P, cfg.P, cfg.C]} sub="gh × gw × ph × pw × c" />
                  <Arrow vertical label="flatten spatial → sequence" color={C_PATCH} />
                  <Shape label="3. patch sequence" color={C_PATCH} dims={[d.N, d.patchDim]} sub={`N=${cfg.grid}² patches, each ${cfg.C}·${cfg.P}·${cfg.P} long`} big />
                </div>
              </div>
              <InfoBox color={C_PATCH} title="The key numbers">
                <ul className="space-y-1 text-xs list-disc pl-4">
                  <li><span className="font-mono">N = (H/P)² = ({d.H}/{cfg.P})² = {d.N.toLocaleString()}</span> patches</li>
                  <li><span className="font-mono">patch_dim = C·P·P = {cfg.C}·{cfg.P}·{cfg.P} = {d.patchDim.toLocaleString()}</span></li>
                  <li>Double the image side ⇒ <strong>4×</strong> the patches (quadratic). This is why resolution is expensive.</li>
                </ul>
              </InfoBox>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 2 — Patch & position embedding
// ═══════════════════════════════════════════════════════════════════════════════

function EmbedPage({ d, cfg }) {
  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_EMB}>Step 2 — patch embedding</SectionLabel>
          <CardTitle className="text-xl">One matmul turns raw pixels into learnable tokens</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            Each flattened patch vector of length <span className="font-mono">{d.patchDim.toLocaleString()}</span> is
            multiplied by a shared weight matrix to produce a <span className="font-mono">{cfg.D}</span>-dim embedding.
            In code this is usually a single <span className="font-mono">Conv2d(C, D, kernel={cfg.P}, stride={cfg.P})</span> —
            a convolution whose stride equals its kernel is exactly "one linear projection per patch".
          </p>

          <div className="grid md:grid-cols-2 gap-6 items-start">
            <div className="rounded-xl border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">The projection</div>
              <div className="flex flex-col gap-2 items-start">
                <Shape label="patches" color={C_PATCH} dims={[d.N, d.patchDim]} />
                <Arrow vertical label={`× W  [${d.patchDim} × ${cfg.D}]`} sub="shared across all patches" color={C_EMB} />
                <Shape label="patch embeddings" color={C_EMB} dims={[d.N, cfg.D]} big />
              </div>
              <div className="text-xs text-muted-foreground">
                Projection weight ≈ <span className="font-mono">{(d.patchEmbedParams).toLocaleString()}</span> params
                (<span className="font-mono">{d.patchDim}×{cfg.D} + {cfg.D}</span>).
              </div>
            </div>

            <div className="space-y-3">
              <div className="rounded-xl border p-4 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {cfg.cls ? 'Prepend CLS token, then add positions' : 'Add positional embeddings'}
                </div>
                <div className="flex flex-col gap-2 items-start">
                  <Shape label="patch embeddings" color={C_EMB} dims={[d.N, cfg.D]} />
                  {cfg.cls && <Arrow vertical label="prepend learnable [CLS]" color={C_ATTN} />}
                  {cfg.cls && <Shape label="with CLS" color={C_ATTN} dims={[d.seqIn, cfg.D]} sub={`${d.N} + 1`} />}
                  <Arrow vertical label="+ position embedding (same shape)" color={C_EMB} />
                  <Shape label="encoder input" color={C_EMB} dims={[d.seqIn, cfg.D]} big />
                </div>
              </div>
              <InfoBox color={C_EMB} title="Why add positions?">
                Self-attention is permutation-invariant — shuffle the tokens and you get the same answer. Positional
                embeddings (a <span className="font-mono">[{d.seqIn} × {cfg.D}]</span> table, learned or sinusoidal/RoPE)
                are <em>added</em> so the model can tell patch (0,0) from (3,7). Nothing about the shape changes; it's an
                elementwise add.
              </InfoBox>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="tokens in" value={d.N.toLocaleString()} color={C_PATCH} mono />
            <Stat label="dim per patch (in)" value={d.patchDim.toLocaleString()} color={C_PATCH} mono />
            <Stat label="hidden dim D (out)" value={cfg.D.toLocaleString()} color={C_EMB} mono />
            <Stat label={cfg.cls ? 'seq len (+CLS)' : 'seq len'} value={d.seqIn.toLocaleString()} color={C_EMB} mono />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 3 — Inside the ViT encoder block
// ═══════════════════════════════════════════════════════════════════════════════

function EncoderPage({ d, cfg }) {
  const N = d.seqIn;
  const mlpDim = cfg.mlpRatio * cfg.D;
  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_ATTN}>Step 3 — the transformer stack</SectionLabel>
          <CardTitle className="text-xl">Shapes through one of {cfg.layers} identical blocks</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            The encoder is <strong>{cfg.layers} identical blocks</strong> stacked back-to-back. The remarkable part:
            the tensor goes in as <span className="font-mono">[{N} × {cfg.D}]</span> and comes out as{' '}
            <span className="font-mono">[{N} × {cfg.D}]</span> — <strong>the shape never changes</strong>. Inside, though,
            it fans out into multi-head attention and a wide MLP.
          </p>

          <div className="grid lg:grid-cols-2 gap-6 items-start">
            {/* Attention sub-block */}
            <div className="rounded-xl border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: C_ATTN }}>Multi-head self-attention</div>
              <div className="flex flex-col gap-2 items-start">
                <Shape label="input (LayerNorm)" color={C_EMB} dims={[N, cfg.D]} />
                <Arrow vertical label={`Q,K,V = x·W  (3 × [${cfg.D}×${cfg.D}])`} color={C_ATTN} />
                <Shape label="Q, K, V each" color={C_ATTN} dims={[N, cfg.D]} />
                <Arrow vertical label={`split into ${cfg.heads} heads`} sub={`d_head = ${cfg.D}/${cfg.heads} = ${d.headDim}`} color={C_ATTN} />
                <Shape label="per-head Q,K,V" color={C_ATTN} dims={[cfg.heads, N, d.headDim]} />
                <Arrow vertical label={`scores = Q·Kᵀ / √${d.headDim}`} color={C_ATTN} />
                <Shape label="attention matrix" color={C_ATTN} dims={[cfg.heads, N, N]} sub={`softmax over the last axis · ${(cfg.heads * N * N).toLocaleString()} weights`} />
                <Arrow vertical label="× V, then concat heads + output proj" color={C_ATTN} />
                <Shape label="attention output" color={C_EMB} dims={[N, cfg.D]} big />
              </div>
            </div>

            {/* MLP sub-block */}
            <div className="space-y-3">
              <div className="rounded-xl border p-4 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: C_EMB }}>Feed-forward MLP</div>
                <div className="flex flex-col gap-2 items-start">
                  <Shape label="input (LayerNorm)" color={C_EMB} dims={[N, cfg.D]} />
                  <Arrow vertical label={`up-project × ${cfg.mlpRatio}`} sub={`W₁ [${cfg.D}×${mlpDim}]`} color={C_EMB} />
                  <Shape label="hidden (GELU)" color={C_PATCH} dims={[N, mlpDim]} sub={`${cfg.mlpRatio}× wider`} />
                  <Arrow vertical label="down-project" sub={`W₂ [${mlpDim}×${cfg.D}]`} color={C_EMB} />
                  <Shape label="block output" color={C_EMB} dims={[N, cfg.D]} big />
                </div>
              </div>
              <InfoBox color={C_ATTN} title="Where the cost lives">
                <ul className="space-y-1 text-xs list-disc pl-4">
                  <li>Attention matrix is <span className="font-mono">[{cfg.heads} × {N} × {N}]</span> — it grows with the <strong>square</strong> of sequence length. Doubling resolution 4×'s N and 16×'s this.</li>
                  <li>Each block ≈ <span className="font-mono">{((4 * cfg.D * cfg.D + 2 * mlpDim * cfg.D) / 1e6).toFixed(1)}M</span> params; ×{cfg.layers} blocks ≈ <span className="font-mono">{(d.encoderParams / 1e6).toFixed(0)}M</span>.</li>
                  <li>Residual + LayerNorm wrap each sub-block, so shape is preserved end-to-end.</li>
                </ul>
              </InfoBox>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="blocks" value={cfg.layers} color={C_ATTN} mono />
            <Stat label="heads / block" value={cfg.heads} color={C_ATTN} mono />
            <Stat label="d_head" value={d.headDim} color={C_ATTN} mono />
            <Stat label="attn weights / block" value={(cfg.heads * N * N).toLocaleString()} color={C_ATTN} mono />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 4 — 2×2 compression (pixel shuffle / patch merge)
// ═══════════════════════════════════════════════════════════════════════════════

function MergeGrid({ grid, merge, size = 260 }) {
  const cell = size / grid;
  const groups = [];
  const mg = Math.floor(grid / merge);
  for (let r = 0; r < mg; r++) for (let c = 0; c < mg; c++) groups.push({ r, c });
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="rounded-lg border" style={{ background: '#f8fafc', maxWidth: '100%' }}>
      {Array.from({ length: grid }).map((_, r) =>
        Array.from({ length: grid }).map((_, c) => (
          <rect key={`${r}-${c}`} x={c * cell} y={r * cell} width={cell} height={cell}
            fill={C_PATCH} fillOpacity={0.12 + ((r + c) % 2) * 0.08} stroke="#fff" strokeWidth={0.5} />
        ))
      )}
      {groups.map((g, i) => (
        <rect key={i} x={g.c * merge * cell} y={g.r * merge * cell}
          width={merge * cell} height={merge * cell}
          fill="none" stroke={C_COMP} strokeWidth={2} />
      ))}
    </svg>
  );
}

function CompressPage({ d, cfg }) {
  const m = cfg.merge;
  if (m === 1) {
    return (
      <div className="space-y-6">
        <PresetNote d={d} cfg={cfg} />
      </div>
    );
  }
  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_COMP}>Step 4 — visual-token compression</SectionLabel>
          <CardTitle className="text-xl">Merge each {m}×{m} neighborhood into one fatter token</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            After the encoder we still have <span className="font-mono">{d.N.toLocaleString()}</span> tokens — too many
            for the LLM. The fix (pixel-shuffle / patch-merge) reshapes the token grid back to 2D, groups it into{' '}
            <span className="font-mono">{m}×{m}</span> blocks, and <strong>concatenates each block's {m * m} vectors along
            the channel axis</strong>. Fewer tokens, but each is {m * m}× wider — no information is thrown away, it's moved
            from the sequence axis into the feature axis.
          </p>

          <div className="grid md:grid-cols-2 gap-6 items-start">
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {cfg.grid}×{cfg.grid} token grid → {d.mergedGrid}×{d.mergedGrid} groups (teal = one output token)
              </div>
              <MergeGrid grid={cfg.grid} merge={m} />
              <div className="text-xs text-muted-foreground text-center">
                {d.N.toLocaleString()} tokens → {d.visualTokens.toLocaleString()} tokens · {d.compressRatio}× reduction
              </div>
            </div>

            <div className="space-y-3">
              <div className="rounded-xl border p-4 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">The shape math</div>
                <div className="flex flex-col gap-2 items-start">
                  <Shape label="encoder output" color={C_EMB} dims={[d.N, cfg.D]} sub="patch tokens (CLS dropped)" />
                  <Arrow vertical label="reshape sequence → 2D grid" color={C_COMP} />
                  <Shape label="grid" color={C_PATCH} dims={[cfg.grid, cfg.grid, cfg.D]} />
                  <Arrow vertical label={`group ${m}×${m}, concat channels`} color={C_COMP} />
                  <Shape label="merged grid" color={C_COMP} dims={[d.mergedGrid, d.mergedGrid, d.mergedDim]} sub={`channels ×${m * m}`} />
                  <Arrow vertical label={`flatten + MLP → D (or LLM dim)`} color={C_COMP} />
                  <Shape label="compressed tokens" color={C_COMP} dims={[d.visualTokens, cfg.D]} big />
                </div>
              </div>
              <InfoBox color={C_COMP} title="Sequence-length is the whole point">
                <ul className="space-y-1 text-xs list-disc pl-4">
                  <li>Tokens: <span className="font-mono">{d.N.toLocaleString()} → {d.visualTokens.toLocaleString()}</span> ({d.compressRatio}× fewer)</li>
                  <li>Channels: <span className="font-mono">{cfg.D} → {d.mergedDim.toLocaleString()}</span> before the projection ({m * m}× wider)</li>
                  <li>Decoder attention cost scales with tokens²: this cuts it by <span className="font-mono">{(d.compressRatio * d.compressRatio)}×</span>.</li>
                </ul>
              </InfoBox>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function PresetNote({ d, cfg }) {
  return (
    <Card className="rounded-3xl shadow-sm">
      <CardHeader className="pb-2">
        <SectionLabel color={C_COMP}>Step 4 — visual-token compression</SectionLabel>
        <CardTitle className="text-xl">This preset uses no merge (factor = 1)</CardTitle>
      </CardHeader>
      <CardContent className="p-6 pt-2 space-y-4">
        <p className="text-sm opacity-85 leading-relaxed">
          The current config has <span className="font-mono">merge = 1</span>, so all{' '}
          <span className="font-mono">{d.N.toLocaleString()}</span> encoder tokens go straight to the projector with no
          spatial merging (this is how plain CLIP/SigLIP encoders behave). Switch to a <strong>Qwen2-VL</strong> or{' '}
          <strong>InternVL</strong> preset — or bump the merge factor in the playground — to see the 2×2 compression in action.
        </p>
        <div className="grid grid-cols-3 gap-3">
          <Stat label="tokens in" value={d.N.toLocaleString()} color={C_PATCH} mono />
          <Stat label="merge factor" value={`${cfg.merge}×${cfg.merge}`} color={C_COMP} mono />
          <Stat label="tokens out" value={d.visualTokens.toLocaleString()} color={C_COMP} mono />
        </div>
      </CardContent>
    </Card>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 5 — Into the text decoder
// ═══════════════════════════════════════════════════════════════════════════════

function DecoderPage({ d, cfg }) {
  const visW = Math.min(28, d.visualTokens);
  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_TEXT}>Step 5 — the text decoder</SectionLabel>
          <CardTitle className="text-xl">Visual tokens become just another part of the prompt</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 space-y-5">
          <p className="text-sm opacity-85 leading-relaxed">
            The compressed visual tokens still live in the vision hidden size (<span className="font-mono">{cfg.D}</span>).
            A small <strong>projector MLP</strong> maps them to the LLM's embedding size (<span className="font-mono">{cfg.Dllm}</span>),
            after which they are <strong>indistinguishable from text-token embeddings</strong> — concatenated into one
            sequence and fed to the decoder.
          </p>

          <div className="grid md:grid-cols-2 gap-6 items-start">
            <div className="rounded-xl border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Projector (modality bridge)</div>
              <div className="flex flex-col gap-2 items-start">
                <Shape label="compressed visual tokens" color={C_COMP} dims={[d.visualTokens, cfg.D]} />
                <Arrow vertical label={`MLP: ${cfg.D} → ${cfg.Dllm}`} sub="align to LLM embedding space" color={C_TEXT} />
                <Shape label="projected visual tokens" color={C_TEXT} dims={[d.visualTokens, cfg.Dllm]} big />
              </div>
              <div className="text-xs text-muted-foreground">Text tokens are embedded the usual way: <span className="font-mono">[{cfg.textTokens} × {cfg.Dllm}]</span>.</div>
            </div>

            <div className="rounded-xl border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Sequence assembly</div>
              {/* token strip */}
              <div className="flex flex-wrap gap-0.5">
                {Array.from({ length: visW }).map((_, i) => (
                  <div key={`v${i}`} className="h-5 w-2.5 rounded-sm" style={{ background: C_COMP, opacity: 0.5 + (i / visW) * 0.5 }} />
                ))}
                {d.visualTokens > visW && <span className="text-[10px] self-center px-1" style={{ color: C_COMP }}>…{d.visualTokens - visW} more</span>}
                {Array.from({ length: Math.min(cfg.textTokens, 12) }).map((_, i) => (
                  <div key={`t${i}`} className="h-5 w-2.5 rounded-sm" style={{ background: C_TEXT }} />
                ))}
              </div>
              <div className="flex flex-col gap-2 items-start">
                <Shape label="full decoder input" color={C_TEXT} dims={[d.seqLen, cfg.Dllm]} sub={`${d.visualTokens} visual + ${cfg.textTokens} text`} big />
                <Arrow vertical label="causal transformer decoder" sub="autoregressive: 1 new token per step" color={C_TEXT} />
                <Shape label="logits per step" color={C_ATTN} dims={['seq', 'vocab']} sub="argmax/sample → next token → append → repeat" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="visual tokens" value={d.visualTokens.toLocaleString()} color={C_COMP} mono />
            <Stat label="text tokens" value={cfg.textTokens} color={C_TEXT} mono />
            <Stat label="decoder seq length" value={d.seqLen.toLocaleString()} color={C_TEXT} mono />
            <Stat label="projector params ≈" value={`${(d.projectorParams / 1e6).toFixed(1)}M`} color={C_TEXT} mono />
          </div>

          <InfoBox color={C_TEXT} title="The full journey, in shapes">
            <span className="font-mono text-xs">
              [{d.H}×{d.W}×{cfg.C}] → [{d.N}×{d.patchDim}] → [{d.seqIn}×{cfg.D}] →(×{cfg.layers})→ [{d.seqIn}×{cfg.D}] → [{d.visualTokens}×{cfg.D}] → [{d.visualTokens}×{cfg.Dllm}] → prepend {cfg.textTokens} text → [{d.seqLen}×{cfg.Dllm}] → decode.
            </span>
          </InfoBox>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAPTER 6 — Shape playground
// ═══════════════════════════════════════════════════════════════════════════════

function divisorsOf(n) {
  const out = [];
  for (let i = 1; i <= n; i++) if (n % i === 0) out.push(i);
  return out;
}

function PlaygroundPage({ cfg, setCfg, d }) {
  const setK = (k, v) => setCfg(c => ({ ...c, [k]: v }));
  const mergeOptions = [1, 2, 3, 4].filter(m => cfg.grid % m === 0);
  const headOptions = divisorsOf(cfg.D).filter(h => h >= 4 && h <= 32 && cfg.D / h >= 32 && cfg.D / h <= 128);

  const rows = [
    { c: C_PIX, stage: 'Input image', shape: `${d.H} × ${d.W} × ${cfg.C}`, note: 'raw RGB pixels' },
    { c: C_PATCH, stage: 'Patch sequence', shape: `${d.N} × ${d.patchDim}`, note: `${cfg.grid}×${cfg.grid} flattened P·P·C` },
    { c: C_EMB, stage: 'Patch embeddings', shape: `${d.N} × ${cfg.D}`, note: 'linear projection to D' },
    { c: C_EMB, stage: cfg.cls ? 'Encoder input (+CLS +pos)' : 'Encoder input (+pos)', shape: `${d.seqIn} × ${cfg.D}`, note: 'positions added' },
    { c: C_ATTN, stage: 'Per-head Q/K/V', shape: `${cfg.heads} × ${d.seqIn} × ${d.headDim}`, note: `${cfg.heads} heads · d_head=${d.headDim}` },
    { c: C_ATTN, stage: 'Attention matrix', shape: `${cfg.heads} × ${d.seqIn} × ${d.seqIn}`, note: 'per block' },
    { c: C_EMB, stage: `Encoder output (×${cfg.layers})`, shape: `${d.seqIn} × ${cfg.D}`, note: 'shape preserved' },
    { c: C_COMP, stage: 'Merged (concat channels)', shape: `${d.mergedGrid} × ${d.mergedGrid} × ${d.mergedDim}`, note: `${cfg.merge}×${cfg.merge} groups` },
    { c: C_COMP, stage: 'Compressed visual tokens', shape: `${d.visualTokens} × ${cfg.D}`, note: `${d.compressRatio}× fewer tokens` },
    { c: C_TEXT, stage: 'Projected visual tokens', shape: `${d.visualTokens} × ${cfg.Dllm}`, note: 'mapped to LLM dim' },
    { c: C_TEXT, stage: 'Decoder input', shape: `${d.seqLen} × ${cfg.Dllm}`, note: `+${cfg.textTokens} text tokens` },
  ];

  return (
    <div className="space-y-6">
      <Card className="rounded-3xl shadow-sm">
        <CardHeader className="pb-2">
          <SectionLabel color={C_PATCH}>Playground</SectionLabel>
          <CardTitle className="text-xl">Turn every knob — every tensor recomputes live</CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-2 grid lg:grid-cols-[300px_1fr] gap-6 items-start">
          {/* Controls */}
          <div className="space-y-4">
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Patch size</div>
              <Seg options={[{ label: '14', value: 14 }, { label: '16', value: 16 }, { label: '32', value: 32 }]} value={cfg.P} onChange={v => setK('P', v)} />
            </div>
            <Slider label="Grid (patches / side)" value={cfg.grid} min={4} max={48} step={1} onChange={v => setCfg(c => ({ ...c, grid: v, merge: v % c.merge === 0 ? c.merge : 1 }))} suffix={` → ${cfg.grid * cfg.P}px`} />
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">ViT hidden dim D</div>
              <Seg options={[192, 384, 768, 1024, 1152, 1280].map(v => ({ label: String(v), value: v }))} value={cfg.D} onChange={v => setCfg(c => ({ ...c, D: v, heads: v % c.heads === 0 ? c.heads : 8 }))} />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Attention heads</div>
              <Seg options={headOptions.map(v => ({ label: String(v), value: v }))} value={headOptions.includes(cfg.heads) ? cfg.heads : headOptions[0]} onChange={v => setK('heads', v)} />
            </div>
            <Slider label="Encoder blocks" value={cfg.layers} min={2} max={48} onChange={v => setK('layers', v)} />
            <Slider label="MLP ratio" value={cfg.mlpRatio} min={2} max={8} onChange={v => setK('mlpRatio', v)} suffix="×" />
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Compression (merge)</div>
              <Seg options={mergeOptions.map(v => ({ label: `${v}×${v}`, value: v }))} value={mergeOptions.includes(cfg.merge) ? cfg.merge : 1} onChange={v => setK('merge', v)} />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">LLM hidden dim</div>
              <Seg options={[1024, 2048, 3584, 4096].map(v => ({ label: String(v), value: v }))} value={cfg.Dllm} onChange={v => setK('Dllm', v)} />
            </div>
            <Slider label="Text prompt tokens" value={cfg.textTokens} min={0} max={128} onChange={v => setK('textTokens', v)} />
            <div className="space-y-1">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">CLS token</div>
              <Seg options={[{ label: 'no', value: false }, { label: 'yes', value: true }]} value={cfg.cls} onChange={v => setK('cls', v)} />
            </div>
          </div>

          {/* Live shape table + budget */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="ViT params ≈" value={`${(d.vitTotal / 1e6).toFixed(0)}M`} color={C_EMB} mono />
              <Stat label="encoder GFLOPs ≈" value={d.gflops.toFixed(1)} color={C_ATTN} mono />
              <Stat label="patches → tokens" value={`${d.N}→${d.visualTokens}`} color={C_COMP} mono />
              <Stat label="decoder seq len" value={d.seqLen.toLocaleString()} color={C_TEXT} mono />
            </div>

            <div className="rounded-xl border overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-neutral-50 text-left">
                    <th className="p-2 font-semibold">Stage</th>
                    <th className="p-2 font-semibold font-mono">Shape</th>
                    <th className="p-2 font-semibold hidden md:table-cell">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-t" style={{ background: r.c + '06' }}>
                      <td className="p-2 font-medium" style={{ color: r.c }}>{r.stage}</td>
                      <td className="p-2 font-mono font-bold" style={{ color: r.c }}>[ {r.shape} ]</td>
                      <td className="p-2 text-muted-foreground hidden md:table-cell">{r.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <InfoBox color={C_GREY} title="Reading the budget">
              Parameter and FLOP figures are first-order estimates (biases/LayerNorm folded in loosely) meant to show
              <em> how the knobs trade off</em>, not to match a specific checkpoint to the last million. Notice how grid size
              hits GFLOPs hardest (quadratic through attention) while <span className="font-mono">D</span> and{' '}
              <span className="font-mono">layers</span> dominate parameter count.
            </InfoBox>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHELL
// ═══════════════════════════════════════════════════════════════════════════════

export default function VitPipelineVisualizer() {
  const [page, setPage] = useState(0);
  const [presetKey, setPresetKey] = useState('qwen');
  const [cfg, setCfg] = useState(PRESETS.qwen.cfg);

  const setPreset = (k) => { setPresetKey(k); setCfg(PRESETS[k].cfg); };
  // Any manual edit means we're no longer on a named preset.
  const editCfg = (updater) => { setPresetKey('custom'); setCfg(updater); };

  const d = useMemo(() => derive(cfg), [cfg]);

  const pageComponent = useMemo(() => {
    switch (page) {
      case 0: return <OverviewPage d={d} cfg={cfg} />;
      case 1: return <PatchifyPage d={d} cfg={cfg} />;
      case 2: return <EmbedPage d={d} cfg={cfg} />;
      case 3: return <EncoderPage d={d} cfg={cfg} />;
      case 4: return <CompressPage d={d} cfg={cfg} />;
      case 5: return <DecoderPage d={d} cfg={cfg} />;
      case 6: return <PlaygroundPage cfg={cfg} setCfg={editCfg} d={d} />;
      default: return null;
    }
  }, [page, d, cfg]);

  return (
    <div className="min-h-screen bg-white p-4 text-black md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight md:text-4xl">A Vision Transformer, Shape by Shape</h1>
          <p className="mt-1 text-sm opacity-60">
            Pixels → patches → ViT → 2×2 compression → text decoder · every dimension, every transformation
          </p>
        </div>

        {/* Global config bar (hidden on playground, which has its own controls) */}
        {page !== 6 && <PresetBar presetKey={presetKey} setPreset={setPreset} d={d} cfg={cfg} />}

        {/* Navigation */}
        <Card className="rounded-3xl shadow-sm">
          <CardContent className="flex items-center justify-between gap-4 p-4">
            <Button variant="outline" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Previous</Button>
            <div className="text-center min-w-0 flex-1">
              <div className="font-semibold text-sm truncate">{pages[page].title}</div>
              <div className="text-xs text-muted-foreground truncate">{pages[page].subtitle}</div>
            </div>
            <Button disabled={page === pages.length - 1} onClick={() => setPage(p => p + 1)}>Next →</Button>
          </CardContent>
        </Card>

        {/* Progress dots */}
        <div className="flex justify-center gap-2 flex-wrap">
          {pages.map((p, i) => (
            <button
              key={p.id} onClick={() => setPage(i)} title={p.title}
              className="rounded-full transition-all text-xs font-medium"
              style={{
                background: page === i ? '#0a0a0a' : '#e5e5e5',
                color: page === i ? 'white' : '#737373',
                padding: page === i ? '4px 12px' : '4px 8px',
                minWidth: page === i ? 80 : 24,
              }}
            >
              {page === i ? `${i + 1}. ${p.title.split(' ').slice(0, 3).join(' ')}` : i + 1}
            </button>
          ))}
        </div>

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
