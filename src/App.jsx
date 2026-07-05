import { useState } from 'react';
import DocumentParsingVisualizer from '../document_parsing_visualizer';
import VitPipelineVisualizer from '../vit_pipeline_visualizer';

const APPS = [
  { id: 'vit', label: 'ViT, shape by shape', Component: VitPipelineVisualizer },
  { id: 'doc', label: 'Document parsing', Component: DocumentParsingVisualizer },
];

export default function App() {
  const [appId, setAppId] = useState('vit');
  const Active = APPS.find(a => a.id === appId).Component;

  return (
    <div>
      <div className="fixed right-3 top-3 z-50 flex rounded-lg border bg-white/90 shadow-sm backdrop-blur">
        {APPS.map(a => (
          <button
            key={a.id}
            onClick={() => setAppId(a.id)}
            className="px-3 py-1.5 text-xs font-medium transition-colors first:rounded-l-lg last:rounded-r-lg"
            style={{
              background: appId === a.id ? '#0a0a0a' : 'transparent',
              color: appId === a.id ? 'white' : '#525252',
            }}
          >
            {a.label}
          </button>
        ))}
      </div>
      <Active />
    </div>
  );
}
