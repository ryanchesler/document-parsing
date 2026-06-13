# Document Parsing — From OCR to Vision-Language Models

An interactive talk site exploring the modern document parsing landscape, built for the San Diego Machine Learning meetup.

7 chapters covering the journey from classical OCR pipelines through the current vision-language model wave:

1. **What can you do with a document?** — taxonomy of doc understanding tasks
2. **Classical OCR** — detection → cropping → recognition pipeline (using a real Chinese financial report as the example)
3. **Enter the VLM** — architecture, patch resolution, visual-token compression, schema-prompted extraction
4. **Where VLMs still fail** — hallucination, unparseable outputs, repetition loops, latency (with real bug reports cited)
5. **Training & data** — five data acquisition methods, synthetic generation (tables + charts), pseudo-labeling, CE vs RL with the reading-order alignment problem
6. **Inference & architecture** — autoregressive bottleneck, VRFM coarse-to-fine, MinerU-Diffusion, PA-BDM block diffusion
7. **The current landscape** — benchmarks (OmniDocBench, olmOCR-Bench, ParseBench), multilingual capability comparison, model spotlights (MinerU, olmOCR-2, PaddleOCR-VL-1.6, Nemotron Parse, Chandra OCR 2, HunyuanOCR)

## Tech stack

- Vite + React 19
- Tailwind CSS v4
- Framer Motion (page transitions)
- Inline SVG diagrams, no chart libraries
- Pre-rendered demo images for the patch-resolution explorer

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:5173/document-parsing/`.

## Deployment

This site auto-deploys to GitHub Pages on every push to `main` via the workflow in `.github/workflows/deploy.yml`.

The `base` path in `vite.config.js` is set to `/document-parsing/` to match the repo name. If you fork this and rename the repo, update that value to match.

## Sources & citations

All quantitative claims in the site link directly to the source paper, repo, or benchmark. The notable ones:

- MinerU-Diffusion paper — [arxiv:2603.22458](https://arxiv.org/abs/2603.22458)
- PA-BDM block diffusion paper — [arxiv:2605.16861](https://arxiv.org/abs/2605.16861)
- PaddleOCR-VL VRFM paper — [arxiv:2510.14528](https://arxiv.org/abs/2510.14528)
- olmOCR-2 RL paper — [arxiv:2510.19817](https://arxiv.org/abs/2510.19817)
- HunyuanOCR technical report — [arxiv:2511.19575](https://arxiv.org/abs/2511.19575)
- OmniDocBench (CVPR 2025) — [arxiv:2412.07626](https://arxiv.org/abs/2412.07626)
- ParseBench — [arxiv:2604.08538](https://arxiv.org/abs/2604.08538)
- Chandra 90-language benchmark — [GitHub](https://github.com/datalab-to/chandra/blob/master/FULL_BENCHMARKS.md)

## License

Code under Apache 2.0. Real-document samples in `public/resources/` are illustrative — replace with your own examples before redistribution.
