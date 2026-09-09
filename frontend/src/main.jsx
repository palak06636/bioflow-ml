import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity, BrainCircuit, Database, GripVertical, Play, UploadCloud, X} from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_URL || '';
const blocks = [
  {id: 'impute', label: 'Fill Missing Values', type: 'prepare', description: 'Median and most-frequent imputation'},
  {id: 'scale', label: 'Standardize Features', type: 'prepare', description: 'Zero mean and unit variance'},
  {id: 'logistic_regression', label: 'Logistic Regression', type: 'model', description: 'Fast, interpretable classifier'},
  {id: 'random_forest', label: 'Random Forest', type: 'model', description: 'Ensemble of decision trees'},
  {id: 'knn', label: 'K-Nearest Neighbours', type: 'model', description: 'Similarity-based classifier'},
];

function App() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState(null);
  const [target, setTarget] = useState('');
  const [pipeline, setPipeline] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function inspect(selected) {
    if (!selected) return;
    setFile(selected); setError(''); setResult(null);
    const body = new FormData(); body.append('file', selected);
    const response = await fetch(`${API}/api/dataset/inspect`, {method: 'POST', body});
    const data = await response.json();
    if (!response.ok) return setError(data.error);
    setSummary(data); setTarget(data.columns.at(-1));
  }

  function addBlock(id) {
    const block = blocks.find(item => item.id === id);
    if (!block) return;
    setPipeline(current => block.type === 'model'
      ? [...current.filter(item => item.type !== 'model'), block]
      : current.some(item => item.id === id) ? current : [...current, block]);
  }

  async function train() {
    const model = pipeline.find(item => item.type === 'model');
    if (!file || !target || !model) return setError('Upload data, select a target, and add a model block.');
    setBusy(true); setError('');
    const body = new FormData();
    body.append('file', file); body.append('target', target); body.append('model', model.id);
    body.append('scale', pipeline.some(item => item.id === 'scale'));
    try {
      const response = await fetch(`${API}/api/train`, {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  return <div className="app">
    <header><div className="brand"><BrainCircuit size={30}/><span>BioFlow <b>ML</b></span></div><div className="status"><i/> Local workspace</div></header>
    <main>
      <section className="hero"><p className="eyebrow">NO-CODE MACHINE LEARNING</p><h1>Turn your dataset into an<br/><em>intelligent pipeline.</em></h1><p>Upload health or biological data, design a workflow visually, and train a model in seconds.</p></section>
      <div className="workspace">
        <aside>
          <h3><Database size={17}/> Data</h3>
          <label className="upload" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault(); inspect(e.dataTransfer.files[0])}}>
            <UploadCloud/><strong>{file ? file.name : 'Drop your CSV here'}</strong><span>{file ? `${summary?.rows || '...'} rows loaded` : 'or click to browse · max 10 MB'}</span>
            <input type="file" accept=".csv,text/csv" onChange={e=>inspect(e.target.files[0])}/>
          </label>
          {summary && <label className="field">Prediction target<select value={target} onChange={e=>setTarget(e.target.value)}>{summary.columns.map(c=><option key={c}>{c}</option>)}</select></label>}
          <h3><BrainCircuit size={17}/> Pipeline blocks</h3>
          <p className="hint">Drag a block onto the canvas</p>
          {blocks.map(block=><div className={`block ${block.type}`} key={block.id} draggable onDragStart={e=>e.dataTransfer.setData('block', block.id)} onClick={()=>addBlock(block.id)}><GripVertical/><div><b>{block.label}</b><small>{block.description}</small></div></div>)}
        </aside>
        <section className="canvas" onDragOver={e=>e.preventDefault()} onDrop={e=>addBlock(e.dataTransfer.getData('block'))}>
          <div className="canvas-head"><div><span>WORKFLOW 01</span><h2>Classification Pipeline</h2></div><button onClick={train} disabled={busy}><Play size={16}/>{busy?'Training…':'Run pipeline'}</button></div>
          <div className="flow">
            <div className="dataset-node"><Database/><div><b>{file ? file.name : 'Your dataset'}</b><small>{summary ? `${summary.rows} rows · ${summary.columns.length} columns` : 'Upload a CSV to begin'}</small></div></div>
            {pipeline.length === 0 && <div className="empty"><span>+</span><b>Drop pipeline blocks here</b><small>Start with preprocessing, then add a model</small></div>}
            {pipeline.map((block,index)=><React.Fragment key={block.id}><div className="connector"/><div className={`node ${block.type}`}><span>{index+1}</span><div><b>{block.label}</b><small>{block.description}</small></div><button onClick={()=>setPipeline(pipeline.filter(x=>x.id!==block.id))}><X size={15}/></button></div></React.Fragment>)}
          </div>
          {error && <div className="error">{error}</div>}
          {result && <div className="results"><div className="result-title"><Activity/><div><span>TRAINING COMPLETE</span><h2>Model performance</h2></div></div><div className="metrics">{Object.entries(result.metrics).map(([name,value])=><div key={name}><span>{name}</span><strong>{(value*100).toFixed(1)}%</strong><i style={{width:`${value*100}%`}}/></div>)}</div><p>Trained on {result.training_rows} rows and evaluated on {result.testing_rows} unseen rows.</p></div>}
        </section>
      </div>
    </main>
    <footer>Built for transparent, reproducible machine learning.</footer>
  </div>
}

createRoot(document.getElementById('root')).render(<App/>);

