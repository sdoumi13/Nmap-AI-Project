import { useState } from 'react';
import { Network, Loader2, CheckCircle, XCircle, Play, ArrowRight, Sparkles, Code, Brain } from 'lucide-react';
import { analyzeApi, executionApi, AnalyzeResponse, ExecutionReport } from '../services/api';

export default function RouterPage() {
  const [query, setQuery] = useState('');
  const [target, setTarget] = useState('192.168.188.128');
  const [loading, setLoading] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState<AnalyzeResponse | null>(null);
  const [executionResult, setExecutionResult] = useState<ExecutionReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<'input' | 'analyzing' | 'analyzed' | 'executing' | 'completed'>('input');

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setAnalyzeResult(null);
    setExecutionResult(null);
    setCurrentStep('analyzing');

    try {
      // FIXED: Call analyzeApi directly (it's a function, not an object)
      const result = await analyzeApi({ query, target });
      setAnalyzeResult(result);
      setCurrentStep('analyzed');
    } catch (err) {
      const errorMessage = (err instanceof Error) ? err.message : 'Erreur lors de l\'analyse';
      setError(errorMessage);
      setCurrentStep('input');
    } finally {
      setLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!analyzeResult || !analyzeResult.entry_id) {
      setError('Aucune analyse disponible pour l\'exécution');
      return;
    }

    setLoading(true);
    setError(null);
    setExecutionResult(null);
    setCurrentStep('executing');

    try {
      const command = analyzeResult.generated_command || analyzeResult.best_match_command || '';
      const result = await executionApi.execute({
        entry_id: analyzeResult.entry_id,
        intent: query,
        command: command,
        target: target,
        agent_name: analyzeResult.analysis.target_agent || 'UNKNOWN'
      });
      setExecutionResult(result);
      setCurrentStep('completed');
    } catch (err) {
      const errorMessage = (err instanceof Error) ? err.message : 'Erreur lors de l\'exécution';
      setError(errorMessage);
      setCurrentStep('analyzed');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setQuery('');
    setTarget('192.168.188.128');
    setAnalyzeResult(null);
    setExecutionResult(null);
    setError(null);
    setCurrentStep('input');
  };

  const getAgentIcon = (agent: string) => {
    return agent === 'RAG' ? Sparkles : Code;
  };

  const getAgentColor = (agent: string) => {
    return agent === 'RAG' ? 'text-blue-400' : 'text-purple-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Network className="w-8 h-8 text-purple-400" />
          Router IA - Nouvelle Analyse
        </h1>
        <p className="mt-2 text-slate-400">
          Entrez votre requête pour générer et exécuter une commande Nmap via les agents IA
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-gradient-to-br from-slate-900/95 via-slate-900/90 to-slate-900/95 border border-slate-800/50 rounded-2xl p-8 shadow-2xl relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

        <form onSubmit={handleAnalyze} className="space-y-6 relative z-10">
          {/* Query Input */}
          <div className="space-y-2">
            <label htmlFor="query" className="flex items-center gap-2 text-sm font-semibold text-slate-300 uppercase tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
              Requête utilisateur
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ex: Scan les ports ouverts sur la machine cible"
              className="w-full px-5 py-4 bg-slate-800/60 border-2 border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all duration-300 font-mono text-sm resize-none"
              rows={5}
              required
              disabled={loading}
            />
            {query && (
              <p className="text-xs text-slate-500 flex items-center gap-1">
                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                {query.length} caractères
              </p>
            )}
          </div>

          {/* Target Input */}
          <div className="space-y-2">
            <label htmlFor="target" className="flex items-center gap-2 text-sm font-semibold text-slate-300 uppercase tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              Cible (IP)
            </label>
            <input
              id="target"
              type="text"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="192.168.188.128"
              className="w-full px-5 py-4 bg-slate-800/60 border-2 border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all duration-300 font-mono"
              required
              disabled={loading}
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 via-purple-500 to-blue-600 text-white font-bold rounded-xl hover:from-purple-700 hover:via-purple-600 hover:to-blue-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg"
          >
            {loading ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                <span>Analyse en cours...</span>
              </>
            ) : (
              <>
                <Brain className="w-6 h-6" />
                <span>Analyser et Générer</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 animate-fade-in">
          <div className="flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-400">Erreur</h3>
              <p className="text-sm text-red-300 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Result */}
      {analyzeResult && (
        <div className="space-y-4 animate-fade-in">
          <div className="bg-gradient-to-br from-slate-900/95 via-slate-900/90 to-slate-900/95 border border-slate-800/50 rounded-2xl p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <CheckCircle className="w-6 h-6 text-emerald-400" />
                Résultat de l'Analyse
              </h2>
              {analyzeResult.relevant && (
                <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full text-xs font-semibold">
                  Requête pertinente
                </span>
              )}
            </div>

            {!analyzeResult.relevant ? (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
                <p className="text-yellow-400 text-sm">{analyzeResult.reason || 'Requête non pertinente pour Nmap'}</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Complexity & Agent */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50">
                    <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Complexité</p>
                    <p className="text-2xl font-bold text-white mb-1">{analyzeResult.analysis.level}</p>
                    <p className="text-xs text-slate-500 mt-1">{analyzeResult.analysis.reason}</p>
                  </div>
                  <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50">
                    <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Agent sélectionné</p>
                    <div className="flex items-center gap-3 mb-1">
                      {(() => {
                        const Icon = getAgentIcon(analyzeResult.analysis.target_agent);
                        return <Icon className={`w-6 h-6 ${getAgentColor(analyzeResult.analysis.target_agent)}`} />;
                      })()}
                      <p className="text-2xl font-bold text-white">{analyzeResult.analysis.target_agent}</p>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">
                      {analyzeResult.generation_method === 'RAG' ? 'Génération via RAG' : 'Génération via Diffusion'}
                    </p>
                  </div>
                </div>

                {/* Generated Command */}
                {analyzeResult.generated_command && (
                  <div className="bg-slate-800/40 rounded-xl p-5 border border-slate-700/50">
                    <p className="text-xs text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      Commande générée
                    </p>
                    <code className="block text-sm text-slate-200 font-mono bg-slate-900/70 px-4 py-3 rounded-lg border border-slate-700/50">
                      {analyzeResult.generated_command}
                    </code>
                  </div>
                )}

                {/* Execute Button */}
                {currentStep === 'analyzed' && (
                  <button
                    onClick={handleExecute}
                    disabled={loading || !analyzeResult.generated_command}
                    className="w-full px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-700 hover:to-teal-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-6 h-6 animate-spin" />
                        <span>Exécution en cours...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-6 h-6" />
                        <span>Exécuter avec Agent 5 (Validation + Sandbox + VM)</span>
                      </>
                    )}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Execution Result */}
      {executionResult && (
        <div className="bg-slate-900/90 border border-slate-800/50 rounded-2xl p-6 shadow-xl animate-fade-in">
          <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
            <Play className="w-6 h-6 text-blue-400" />
            Résultat de l'Exécution
          </h2>

          <div className="space-y-4">
            {/* Final Status */}
            <div className={`rounded-xl p-4 border ${
              executionResult.final_status === 'success' 
                ? 'bg-emerald-500/10 border-emerald-500/30' 
                : 'bg-red-500/10 border-red-500/30'
            }`}>
              <div className="flex items-center gap-2">
                {executionResult.final_status === 'success' ? (
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <p className={`font-semibold ${
                  executionResult.final_status === 'success' ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  Statut: {executionResult.final_status}
                </p>
              </div>
            </div>

            {/* Execution Stages */}
            <div className="space-y-3">
              {/* Validation Stage */}
              {executionResult.stages.validation && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Validation</p>
                  <p className="text-sm text-white">Score: {executionResult.stages.validation.score}/100</p>
                  <p className="text-xs text-slate-500">Méthode: {executionResult.stages.validation.method}</p>
                </div>
              )}

              {/* Self-Correction Stage */}
              {executionResult.stages.self_correction && executionResult.stages.self_correction.applied && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Auto-correction</p>
                  <p className="text-sm text-white">
                    Commande corrigée: {executionResult.stages.self_correction.final_command || executionResult.stages.self_correction.corrected_command}
                  </p>
                  {(executionResult.stages.self_correction.history || executionResult.stages.self_correction.attempts) && (
                    <p className="text-xs text-slate-500">
                      Historique: {(executionResult.stages.self_correction.history?.length || executionResult.stages.self_correction.attempts?.length || 0)} tentatives
                    </p>
                  )}
                </div>
              )}

              {/* Sandbox Stage - Handle both 'sandbox' and 'sandbox_execution' */}
              {(executionResult.stages.sandbox || executionResult.stages.sandbox_execution) && (() => {
                const sandbox = executionResult.stages.sandbox || executionResult.stages.sandbox_execution!;
                return (
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                    <p className="text-xs text-slate-400 mb-1">Sandbox Test</p>
                    <p className={`text-sm ${sandbox.success ? 'text-emerald-400' : 'text-red-400'}`}>
                      {sandbox.success ? '✅ Réussi' : '❌ Échoué'}
                    </p>
                  </div>
                );
              })()}

              {/* VM Execution Stage */}
              {executionResult.stages.vm_execution && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Exécution VM</p>
                  <p className={`text-sm ${executionResult.stages.vm_execution.success ? 'text-emerald-400' : 'text-red-400'}`}>
                    {executionResult.stages.vm_execution.success ? '✅ Réussi' : '❌ Échoué'}
                  </p>
                  {executionResult.stages.vm_execution.output && (
                    <pre className="text-xs text-slate-400 mt-2 bg-slate-900/50 p-2 rounded overflow-auto max-h-40">
                      {executionResult.stages.vm_execution.output.substring(0, 500)}
                    </pre>
                  )}
                </div>
              )}
            </div>

            {/* Reset Button */}
            <button
              onClick={handleReset}
              className="w-full px-6 py-3 bg-slate-800/50 text-slate-300 font-semibold rounded-xl hover:bg-slate-800 transition-all flex items-center justify-center gap-2 border border-slate-700/50"
            >
              <ArrowRight className="w-5 h-5 rotate-180" />
              <span>Nouvelle analyse</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}