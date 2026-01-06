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
      const result = await analyzeApi.analyze(query, target);
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

      {/* Input Form - Enhanced with animations */}
      <div className="bg-gradient-to-br from-slate-900/95 via-slate-900/90 to-slate-900/95 border border-slate-800/50 rounded-2xl p-8 shadow-2xl relative overflow-hidden animate-fade-in-up group">
        {/* Animated background gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
        
        {/* Shimmer effect */}
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
        </div>

        <form onSubmit={handleAnalyze} className="space-y-6 relative z-10">
          {/* Query Input - Enhanced */}
          <div className="space-y-2 animate-fade-in-up animate-stagger-1">
            <label htmlFor="query" className="flex items-center gap-2 text-sm font-semibold text-slate-300 uppercase tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
              Requête utilisateur
            </label>
            <div className="relative">
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ex: Scan les ports ouverts sur la machine cible"
                className="w-full px-5 py-4 bg-slate-800/60 border-2 border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all duration-300 font-mono text-sm resize-none backdrop-blur-sm hover:border-slate-600/70 hover:bg-slate-800/70"
                rows={5}
                required
                disabled={loading}
              />
              {/* Animated border on focus */}
              <div className="absolute inset-0 rounded-xl border-2 border-purple-500/0 pointer-events-none transition-all duration-300 focus-within:border-purple-500/30" />
            </div>
            {query && (
              <p className="text-xs text-slate-500 flex items-center gap-1 animate-fade-in">
                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                {query.length} caractères
              </p>
            )}
          </div>

          {/* Target Input - Enhanced */}
          <div className="space-y-2 animate-fade-in-up animate-stagger-2">
            <label htmlFor="target" className="flex items-center gap-2 text-sm font-semibold text-slate-300 uppercase tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              Cible (IP)
            </label>
            <div className="relative">
              <input
                id="target"
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="192.168.188.128"
                className="w-full px-5 py-4 bg-slate-800/60 border-2 border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all duration-300 font-mono backdrop-blur-sm hover:border-slate-600/70 hover:bg-slate-800/70"
                required
                disabled={loading}
              />
              {/* Animated border on focus */}
              <div className="absolute inset-0 rounded-xl border-2 border-blue-500/0 pointer-events-none transition-all duration-300 focus-within:border-blue-500/30" />
            </div>
            {target && (
              <p className="text-xs text-slate-500 flex items-center gap-1 animate-fade-in">
                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                IP cible configurée
              </p>
            )}
          </div>

          {/* Submit Button - Enhanced */}
          <div className="pt-2 animate-fade-in-up animate-stagger-3">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 via-purple-500 to-blue-600 text-white font-bold rounded-xl hover:from-purple-700 hover:via-purple-600 hover:to-blue-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 hover:scale-[1.02] active:scale-[0.98] relative overflow-hidden group/btn"
            >
              {/* Button shimmer effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover/btn:translate-x-full transition-transform duration-700" />
              
              {/* Button content */}
              <div className="relative z-10 flex items-center gap-3">
                {loading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span className="text-base">Analyse en cours...</span>
                  </>
                ) : (
                  <>
                    <Brain className="w-6 h-6 group-hover/btn:rotate-12 transition-transform duration-300" />
                    <span className="text-base">Analyser et Générer</span>
                    <ArrowRight className="w-5 h-5 opacity-0 group-hover/btn:opacity-100 group-hover/btn:translate-x-1 transition-all duration-300" />
                  </>
                )}
              </div>
            </button>
            
            {/* Helper text */}
            {!query.trim() && (
              <p className="text-xs text-slate-500 text-center mt-3 animate-fade-in">
                Entrez votre requête pour commencer l'analyse
              </p>
            )}
          </div>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-400">Erreur</h3>
              <p className="text-sm text-red-300 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Result - Enhanced */}
      {analyzeResult && (
        <div className="space-y-4 animate-fade-in-up">
          <div className="bg-gradient-to-br from-slate-900/95 via-slate-900/90 to-slate-900/95 border border-slate-800/50 rounded-2xl p-6 shadow-2xl relative overflow-hidden group">
            {/* Animated background */}
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 via-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
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
                {/* Complexity & Agent - Enhanced */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-800/40 rounded-xl p-5 border border-slate-700/50 hover:border-purple-500/30 transition-all duration-300 hover-lift relative overflow-hidden group/card">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-500/5 to-purple-500/0 opacity-0 group-hover/card:opacity-100 transition-opacity duration-300" />
                    <div className="relative z-10">
                      <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Complexité</p>
                      <p className="text-2xl font-bold text-white mb-1">{analyzeResult.analysis.level}</p>
                      <p className="text-xs text-slate-500 mt-1">{analyzeResult.analysis.reason}</p>
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-800/40 rounded-xl p-5 border border-slate-700/50 hover:border-blue-500/30 transition-all duration-300 hover-lift relative overflow-hidden group/card">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/5 to-blue-500/0 opacity-0 group-hover/card:opacity-100 transition-opacity duration-300" />
                    <div className="relative z-10">
                      <p className="text-xs text-slate-400 mb-2 uppercase tracking-wider">Agent sélectionné</p>
                      <div className="flex items-center gap-3 mb-1">
                        {(() => {
                          const Icon = getAgentIcon(analyzeResult.analysis.target_agent);
                          return <Icon className={`w-6 h-6 ${getAgentColor(analyzeResult.analysis.target_agent)} group-hover/card:scale-110 transition-transform duration-300`} />;
                        })()}
                        <p className="text-2xl font-bold text-white">{analyzeResult.analysis.target_agent}</p>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        {analyzeResult.generation_method === 'RAG' ? 'Génération via RAG' : 'Génération via Diffusion'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Generated Command - Enhanced */}
                {analyzeResult.generated_command && (
                  <div className="bg-gradient-to-br from-slate-800/60 to-slate-800/40 rounded-xl p-5 border border-slate-700/50 hover:border-emerald-500/30 transition-all duration-300 relative overflow-hidden group/cmd">
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-emerald-500/5 to-emerald-500/0 opacity-0 group-hover/cmd:opacity-100 transition-opacity duration-300" />
                    <div className="relative z-10">
                      <p className="text-xs text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        Commande générée
                      </p>
                      <div className="flex items-center gap-2">
                        <code className="flex-1 text-sm text-slate-200 font-mono bg-slate-900/70 px-4 py-3 rounded-lg border border-slate-700/50 hover:border-emerald-500/30 transition-all duration-300">
                          {analyzeResult.generated_command}
                        </code>
                      </div>
                    </div>
                  </div>
                )}

                {/* Execute Button */}
                {currentStep === 'analyzed' && (
                  <button
                    onClick={handleExecute}
                    disabled={loading || !analyzeResult.generated_command}
                    className="w-full px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-700 hover:to-teal-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    <div className="relative z-10 flex items-center gap-3">
                      {loading ? (
                        <>
                          <Loader2 className="w-6 h-6 animate-spin" />
                          <span className="text-base">Exécution en cours...</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-6 h-6 group-hover/exec:scale-110 transition-transform duration-300" />
                          <span className="text-base">Exécuter avec Agent 5 (Validation + Sandbox + VM)</span>
                          <ArrowRight className="w-5 h-5 opacity-0 group-hover/exec:opacity-100 group-hover/exec:translate-x-1 transition-all duration-300" />
                        </>
                      )}
                    </div>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Execution Result */}
      {executionResult && (
        <div className="bg-slate-900/90 border border-slate-800/50 rounded-2xl p-6 shadow-xl">
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
              {executionResult.stages.validation && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Validation</p>
                  <p className="text-sm text-white">Score: {executionResult.stages.validation.score}/100</p>
                  <p className="text-xs text-slate-500">Méthode: {executionResult.stages.validation.method}</p>
                </div>
              )}

              {executionResult.stages.self_correction && executionResult.stages.self_correction.applied && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Auto-correction</p>
                  <p className="text-sm text-white">Commande corrigée: {executionResult.stages.self_correction.final_command}</p>
                  {executionResult.stages.self_correction.history && (
                    <p className="text-xs text-slate-500">Historique: {executionResult.stages.self_correction.history.length} tentatives</p>
                  )}
                </div>
              )}

              {executionResult.stages.sandbox && (
                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                  <p className="text-xs text-slate-400 mb-1">Sandbox Test</p>
                  <p className={`text-sm ${executionResult.stages.sandbox.success ? 'text-emerald-400' : 'text-red-400'}`}>
                    {executionResult.stages.sandbox.success ? '✅ Réussi' : '❌ Échoué'}
                  </p>
                </div>
              )}

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

